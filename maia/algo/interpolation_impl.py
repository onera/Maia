from   mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from   maia.utils import np_utils
from   maia.utils import vstride as vs

import Pypdm.Pypdm as PDM

from maia.typing import *

def _cell_tgt_to_vtx_tgt(cell_vtx, cell_tgt, cell_vtx_weight, n_vtx):
  """
  Transform cell->tgt (src_to_tgt, src_vtx_weight) information from mesh_location
  onto vtx->tgt information.
  """
  # Id of cell having some tgt, duplicated if multiple tgt
  active_cell = np_utils.repeated_arange(cell_tgt.counts, dtype=np.int32)
  # Cell vtx only for these cells, still duplicated
  # len of active_cell_vtx is == len(active_cell) == cell_tgt.dsize (because of reps)
  active_cell_vtx = vs.take(cell_vtx, active_cell)

  # Number of vertex counted with reps
  # Attention n_vtx différent
  vtx_to_tgt_n = np.zeros(n_vtx, dtype=np.int32)
  np.add.at(vtx_to_tgt_n, active_cell_vtx.values-1, 1)

  # Cell_tgt -> id points shifté ; on les etend 
  cell_tgt_extended = np.repeat(cell_tgt.values, active_cell_vtx.counts) # cell_vtx->tgt

  sort_idx = np.argsort(active_cell_vtx.values)
  vtx_to_tgt     = cell_tgt_extended[sort_idx] # vtx->tgt
  vtx_to_tgt_wgt = cell_vtx_weight[sort_idx]

  vtx_to_tgt_vs = vs.from_counts(vtx_to_tgt_n, vtx_to_tgt)
  vtx_to_weight = vs.from_counts(vtx_to_tgt_n, vtx_to_tgt_wgt)

  return vtx_to_tgt_vs, vtx_to_weight

def _combine_geo_results(all_located_inv, all_closest_inv, strategy, src_loc):
  """
  Extract the srt to tgt data from location and closest point results depending on
  strategy (closest or location or both)
  """
  dist2weight = lambda V : vs.from_displs(V.displs, 1. / np.maximum(V.values, 1E-20))
  if strategy == 'Location':
    if src_loc=="CellCenter":
      tgt_in_src_gnum = [data['points_gnum_shifted'] for data in all_located_inv]
      tgt_in_src_wght = None
    elif src_loc=="Vertex":
      tgt_in_src_gnum = [data['points_gnum_shifted@VTX'] for data in all_located_inv]
      tgt_in_src_wght = [data['points_weights@VTX'] for data in all_located_inv]
        
  elif strategy == 'Closest':
    tgt_in_src_gnum = [data['tgt_in_src_shifted'] for data in all_closest_inv]
    tgt_in_src_wght = [dist2weight(data['tgt_in_src_dist2']) for data in all_closest_inv]
    
  else:
    tgt_in_src_gnum = []
    tgt_in_src_wght = []

    for res_loc, res_clo in zip(all_located_inv, all_closest_inv):
      clo_tgt_in_src_gnum = res_clo['tgt_in_src_shifted']
      clo_tgt_in_scr_wght = dist2weight(res_clo['tgt_in_src_dist2'])

      if src_loc=="CellCenter":
        loc_src_to_tgt_gnum = res_loc['points_gnum_shifted']
        loc_tgt_in_src_wght = vs.from_displs(loc_src_to_tgt_gnum.displs, np.ones(loc_src_to_tgt_gnum.dsize))
      elif src_loc=="Vertex":
        loc_src_to_tgt_gnum = res_loc['points_gnum_shifted@VTX']
        loc_tgt_in_src_wght = res_loc['points_weights@VTX']

      tgt_in_src_gnum.append(vs.concatenate([loc_src_to_tgt_gnum, clo_tgt_in_src_gnum], vs.INNER_AXIS))
      tgt_in_src_wght.append(vs.concatenate([loc_tgt_in_src_wght, clo_tgt_in_scr_wght], vs.INNER_AXIS))

  return tgt_in_src_gnum, tgt_in_src_wght

def _expected_single_val(l:Sequence):
  assert l.count(l[0]) == len(l)
  return l[0]

def discover_fields_name(zones, container_name, root, comm):
  if len(zones) > 0:
    fields_name_l = list()
    label_l = list()
    loc_l = list()
    for zone in zones:
      container = PT.find_node_from_path(zone, container_name)
      fields_name = sorted([PT.get_name(array) for array in PT.iter_children_from_label(container, 'DataArray_t')])
      label_l.append(PT.get_label(container))
      loc_l.append(PT.Container.GridLocation(container))
      fields_name_l.append(fields_name)
    fields_name = _expected_single_val(fields_name_l)
    label = _expected_single_val(label_l)
    loc = _expected_single_val(loc_l)
  else:
    fields_name = label = loc = None

  if root is not None: # Some rank have no src partitions, share field names
    fields_name, label, loc = comm.bcast((fields_name, label, loc), root=root)

  return fields_name, label, loc

class Interpolator:
  """ Low level class to perform interpolations.

  This class instanciate the PartToPart object and run the exchanges from a 
  precomputed src -> target indirection.

  src_parts and tgt_parts, as well as the fields stored in src_to_tgt, are expected as flat lists
  of partitions. In multidomain case, the shifts must have been already performed.
  """
  def __init__(self, src_parts, tgt_parts, src_to_tgt, input_loc, output_loc, comm):
    self.src_parts = src_parts
    self.tgt_parts = tgt_parts
    
    self.output_loc = output_loc
    self.input_loc = input_loc
    self.comm = comm

    # If some rank have no partitions, store a rank used as root to share FS names
    self.root = None
    if comm.allreduce(len(src_parts) == 0, MPI.LOR):
      self.root = self.comm.allreduce(-1 if len(src_parts) == 0 else comm.rank, MPI.MAX)

    self.target_part_size = [gn.size for gn in src_to_tgt['tgt_gnum']]
    
    self.PTP = PDM.PartToPart(comm,
                              src_to_tgt['src_gnum'],
                              src_to_tgt['tgt_gnum'],
                              [a.displs for a in src_to_tgt['src_to_tgt']],
                              [a.values for a in src_to_tgt['src_to_tgt']])

    self.referenced_nums = self.PTP.get_referenced_lnum2()
    self.sending_gnums = self.PTP.get_gnum1_come_from()

    # Send weight to targets partitions (if available, some strategy does not use weights)
    if 'src_to_tgt_weight' in src_to_tgt:
      _weight = [a.values for a in src_to_tgt['src_to_tgt_weight']]
      request = self.PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                               _weight)
      _, self.tgt_weight = self.PTP.wait(request)


  def _reduce_single_val(self, i_part, data):
    """
    A basic reduce function who take the first received value for each target
    """
    come_from_idx = self.sending_gnums[i_part]['come_from_idx']
    return data[come_from_idx[:-1]]

  def _reduce_weighted_mean(self, i_part, data):
    """
    Compute a weighted mean of the received values.
    Usable only if weight are available in src_to_tgt dict
    (eg not Location method used on a CellCenter source).
    """
    come_from_idx = self.sending_gnums[i_part]['come_from_idx']
    reduced_data   = np.add.reduceat(data*self.tgt_weight[i_part], come_from_idx[:-1])
    reduced_factor = np.add.reduceat(     self.tgt_weight[i_part], come_from_idx[:-1])
    assert reduced_data.size == come_from_idx.size - 1
    return reduced_data / reduced_factor


  def exchange_fields(self, container_name, reduce_func=_reduce_weighted_mean):
    """
    For all fields found under container_name node,
    - Perform a part to part exchange
    - Reduce the received data using reduce_func (because tgt elements can receive multiple data)
    - Fill the target sol with a default value + the reduced value
    """

    #Check that solutions are known on each source partition
    fields_names, container_label, _ = discover_fields_name(self.src_parts, container_name, self.root, self.comm)

    for src_part in self.src_parts:
      container = PT.find_node_from_path(src_part, container_name)
      assert PT.Container.GridLocation(container) == self.input_loc

    #Cleanup target partitions
    for tgt_part in self.tgt_parts:
      PT.rm_children_from_name(tgt_part, container_name)
      fs = PT.new_FlowSolution(container_name, loc=self.output_loc, parent=tgt_part)
      PT.set_label(fs, container_label)

    #Collect src sol
    src_field_dic = dict()
    for field_name in fields_names:
      field_path = container_name + '/' + field_name
      src_field_dic[field_name] = [PT.get_node_from_path(part, field_path)[1] for part in self.src_parts]

    #Exchange
    for field_name, src_sol in src_field_dic.items():
      request = self.PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1,
                               src_sol)
      _, lnp_part_data = self.PTP.wait(request)

      for i_part, tgt_part in enumerate(self.tgt_parts):
        fs = PT.get_node_from_path(tgt_part, container_name)
        data_size = self.target_part_size[i_part]
        data = np.nan * np.ones(data_size)
        come_from_idx = self.sending_gnums[i_part]['come_from_idx']
        if (np.diff(come_from_idx) == 1).all():
          reduced_data = lnp_part_data[i_part]
        else:
          reduced_data = reduce_func(self, i_part, lnp_part_data[i_part])
        data[self.referenced_nums[i_part]-1] = reduced_data #Use referenced ids to erase default value

        # Reshape data only for partitioned / structured zones
        if PT.Zone.Type(tgt_part) == 'Structured' and MT.get_Distribution(tgt_part) is None:
          shape = PT.Zone.CellSize(tgt_part) if self.output_loc == 'CellCenter' else PT.Zone.VertexSize(tgt_part)
          PT.update_child(fs, field_name, 'DataArray_t', data.reshape(shape, order='F'))
        else:
          PT.update_child(fs, field_name, 'DataArray_t', data)

def extract_sub_cloud_from_flag(cloud: Tuple[NDArray, NDArray],
                                flag: NDArray) -> Tuple[NDArray, NDArray]:
  sub_coords = cloud[0][np.repeat(flag, 3)]
  sub_lngn   = cloud[1][flag]
  return sub_coords, sub_lngn

class ConservativeInterpolator:
  """
  This base class factorize conservative interpolation pattern for distributed
  or partitioned meshes.

  It is intended to be inherited by specialized classes, which
  must overide the following methods:
  """
  @staticmethod
  def get_native_measure(zone:CGNSTree, comm:MPIComm) -> NDArray:
    """ Return the main measure (volume or area) of the input zone """
    raise NotImplementedError
  @staticmethod
  def get_cell_clouds(zone:Sequence[CGNSTree], comm:MPIComm) -> List[Tuple[NDArray, NDArray]]:
    """ Return point clouds (cell_centers, cell_lngn) of the input zones """
    raise NotImplementedError
  @staticmethod
  def compute_mesh_intersection(src_parts:Sequence[CGNSTree], tgt_parts:Sequence[CGNSTree], comm:MPIComm):
    """ Return the PartToPart and the tgt_to_src dictionnary of mesh intersection """
    raise NotImplementedError

  def VertexToCell(self):
    """ Create a VertexToCell conservative local interpolator for source mesh """
    raise NotImplementedError
  def CellToVertex(self):
    """ Create a CellToVertex conservative local interpolator for target mesh """
    raise NotImplementedError

  """ Note - Multidomain : this class works with a flat view of source and
  target partition. Implementors must manage multidomain offsets.  """

  def __init__(self,
               src_parts:List[CGNSTree],
               tgt_parts:List[CGNSTree],
               comm:MPIComm,
               **kwargs):

    # In the init part of the interpolator we build the part to part and weights
    # used to exchange data from source to target cells.
    # On each partition, weights are a strided array of size len(tgt_cells),
    # containing for each tgt cell a weight w_i for each of its related src cells
    #
    # Assuming that the exchanged fields will be in integrated from (eg mass),
    # the weigth from a source cell I to a tgt cell J is
    # 
    #   Volume_(I∩J) / Volume_I    for standard cells
    #   Volume_J / Volume_I        for tgt cells outside src mesh, where I is the closest cell
    #
    # Target cells that are partially outside src mesh are corrected with the
    # coefficient (1/r) where r = \sum_I Volume_(I∩J)  / Volume_J

    # Since the exchange fields will be usually in conservative form (eg density), we can
    # report the conservative <-> integrated factor (Volume_I / Volume_J) directly in weights
    # which become
    #   Volume_(I∩J) / Volume_J    for standard cells
    #   1                          for tgt cells outside src mesh, where I is the closest cell
    # and correction by (1/r) is unchanged


    vol_src = [self.get_native_measure(zone, comm) for zone in src_parts]
    vol_tgt = [self.get_native_measure(zone, comm) for zone in tgt_parts]

    # Compute intersection between src (part 2) and tgt (part 1)
    ptp, tgt_to_src = self.compute_mesh_intersection(src_parts, tgt_parts, comm)

    src_weights_l = [vs.from_displs(r['a_to_b_idx'], r['a_to_b_weight']) / vol \
                     for r,vol in zip(tgt_to_src, vol_tgt)]
    # Detect tgt cell not completly covered by src cells:
    #   - outside_mask is True if tgt cell is totally   outside src mesh
    #   - partial_mask is True       "        partially        "
    # vol_ratio is the fraction of tgt cell covered by src mesh, since a_to_b_weight already
    # include src vol and we divided by tgt_vol, we just have to sum
    tol = kwargs.get('measure_ratio_tol', 1E-12)
    vol_ratio = [src_weights.reduce(vs.ReduceOp.SUM) for src_weights in src_weights_l]
    partial_mask = [r*(1-r) > tol for r in vol_ratio]
    outside_mask = [r < tol       for r in vol_ratio]

    # Incorporate cut-cell correction for partial cells
    for i, weight in enumerate(src_weights_l):
      vol_ratio_rep    = np.repeat(vol_ratio[i], weight.counts)
      partial_mask_rep = np.repeat(partial_mask[i], weight.counts)
      np.divide(weight.values, vol_ratio_rep, where=partial_mask_rep, out=weight._values)

    # For outside cells, detect the closest cell in src mesh.
    # Incorporate it in PartToPart (update it) with a weight equal to tgt cell volume
    if comm.allreduce(any([outside.any() for outside in outside_mask]), MPI.LOR):
      # Perform closest point on outside cells only
      src_clouds = self.get_cell_clouds(src_parts, comm)
      tgt_clouds = self.get_cell_clouds(tgt_parts, comm)
      _tgt_clouds = [extract_sub_cloud_from_flag(cloud, flag) for cloud,flag in zip(tgt_clouds, outside_mask)]

      from maia.algo.part import closest_points as CLO
      closest_out = CLO._closest_points(src_clouds, _tgt_clouds, comm, False, n_pts=1, need_shift=True)

      a_to_b_cat = []
      weights_cat = []
      for i in range(len(tgt_clouds)):
        a_to_b_mi = vs.from_displs(tgt_to_src[i]['a_to_b_idx'], tgt_to_src[i]['a_to_b'])
        weight_mi = src_weights_l[i]

        if closest_out[i]['closest_src_gnum'].size > 0:
          a_to_b_clo = vs.from_counts(outside_mask[i].astype(np.int32), closest_out[i]['closest_src_gnum'])
          weight_clo = vs.from_counts(a_to_b_clo.counts, np.ones(a_to_b_clo.dsize))
          
          a_to_b_cat.append(vs.concatenate([a_to_b_mi, a_to_b_clo], vs.INNER_AXIS))
          weights_cat.append(vs.concatenate([weight_mi, weight_clo], vs.INNER_AXIS))
        else:
          a_to_b_cat.append(a_to_b_mi)
          weights_cat.append(weight_mi)

      # Override PartToPart and weights
      ptp = PDM.PartToPart(comm,
                           [cloud[1] for cloud in tgt_clouds], # Part 1 is tgt
                           [cloud[1] for cloud in src_clouds], # Part 2 is src
                           [a.displs for a in a_to_b_cat],
                           [a.values for a in a_to_b_cat])
      src_weights_l = weights_cat

    self.ptp = ptp
    self.src_weights_l = src_weights_l
    self.src_parts = src_parts
    self.tgt_parts = tgt_parts
    self.src_vol = vol_src
    self.tgt_vol = vol_tgt
    self.comm = comm

    # Caching
    self._vtx_to_cell_src = None
    self._cell_to_vtx_tgt = None

    # If some rank have no partitions, store a rank used as root to share FS names
    self.root = None
    if comm.allreduce(len(src_parts) == 0, MPI.LOR):
      self.root = self.comm.allreduce(-1 if len(src_parts) == 0 else comm.rank, MPI.MAX)


  @property
  def vtx_to_cell_src(self):
    if self._vtx_to_cell_src is None:
      self._vtx_to_cell_src = self.VertexToCell()
    return self._vtx_to_cell_src
  @property
  def cell_to_vtx_tgt(self):
    if self._cell_to_vtx_tgt is None:
      self._cell_to_vtx_tgt = self.CellToVertex()
    return self._cell_to_vtx_tgt

  def cell_data_transfer(self, 
                         src_fields_l:Dict[str, List[NDArray]],
                         is_conservative:bool) -> Dict[str, List[NDArray]]:

    # This function is relevant for integrated fields (such as mass) :
    # if data is in conservative form, we must multiply it by Density
    # Reminder : in ptp, part1 is target mesh, part2 is src mesh
    rq_dict = dict()
    for name, src_fields in src_fields_l.items():
      
      # Integrated to conservative, if needed
      if not is_conservative:
        src_fields = [f / vol for f,vol in zip(src_fields, self.src_vol)]

      rq_dict[name] = self.ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                                             PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2,
                                             src_fields)
    tgt_fields_l = {}
    for name, rq in rq_dict.items():
      _, recv_datas = self.ptp.reverse_wait(rq)
  
      # Ponderate by weights
      tgt_fields = [
        vs.from_displs(src_weights.displs, data*src_weights.values).reduce(vs.ReduceOp.SUM)
        for data, src_weights in zip(recv_datas, self.src_weights_l)]

      # Conservative to integrated, if needed
      if not is_conservative:
        for f, vol in zip(tgt_fields, self.tgt_vol):
          f *= vol
      
      tgt_fields_l[name] = tgt_fields

    return tgt_fields_l

  def exchange_fields(self, container_name:str, tgt_loc:str, is_conservative=True):

    field_names, cnt_label, src_loc = discover_fields_name(self.src_parts, container_name, self.root, self.comm)

    src_fields_l:Dict[str, List[NDArray]] = {key: [] for key in field_names}
    for src_zone in self.src_parts:
      container = PT.find_node_from_path(src_zone, container_name)
      for key, val in PT.Container.fields(container).items():
        src_fields_l[key].append(val.reshape(-1, order='F')) # Flatten if src zone is S

    if src_loc == 'Vertex':
      src_fields_l = self.vtx_to_cell_src._exchange_fields(src_fields_l, is_conservative)
    elif src_loc != 'CellCenter':
      raise ValueError(f"Unsupported location for input container: {src_loc}")
      
    tgt_fields_l = self.cell_data_transfer(src_fields_l, is_conservative)

    # Back to vertex
    if tgt_loc == 'Vertex':
      tgt_fields_l = self.cell_to_vtx_tgt._exchange_fields(tgt_fields_l, is_conservative)
    elif tgt_loc != 'CellCenter':
      raise ValueError(f"Unsupported location for output container: {tgt_loc}")

    # Update target partitions
    for i,tgt_part in enumerate(self.tgt_parts):
      # Reshape data only for partitioned / structured zones
      PT.rm_children_from_name(tgt_part, container_name)
      if PT.Zone.Type(tgt_part) == 'Structured' and MT.get_Distribution(tgt_part) is None:
        shape = PT.Zone.CellSize(tgt_part) if tgt_loc == 'CellCenter' else PT.Zone.VertexSize(tgt_part)
        fields = {key: vals[i].reshape(shape, order='F') for key,vals in tgt_fields_l.items()}
      else:
        fields = {key: vals[i] for key,vals in tgt_fields_l.items()}
      fs = PT.new_FlowSolution(container_name, loc=tgt_loc, fields=fields, parent=tgt_part)
      PT.set_label(fs, cnt_label)
