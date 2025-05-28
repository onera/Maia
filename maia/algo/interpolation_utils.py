from   mpi4py import MPI
import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from   maia.utils import np_utils
from   maia.utils import vstride as vs

import Pypdm.Pypdm as PDM

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
    fields_per_part = list()
    container_label = ''
    for src_part in self.src_parts:
      container = PT.find_node_from_path(src_part, container_name)
      container_label = PT.get_label(container)
      assert PT.Subset.GridLocation(container) == self.input_loc
      fields_name = sorted([PT.get_name(array) for array in PT.iter_children_from_label(container, 'DataArray_t')])
      fields_per_part.append(fields_name)
    if len(fields_per_part) > 0:
      assert fields_per_part.count(fields_per_part[0]) == len(fields_per_part)

    fields_names = fields_per_part[0] if len(fields_per_part) > 0 else None
    if self.root is not None: # Some rank have no src partitions, share field names
      container_label = self.comm.bcast(container_label, root=self.root)
      fields_names = self.comm.bcast(fields_names, root=self.root)

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