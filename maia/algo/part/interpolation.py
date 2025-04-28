import numpy as np
from mpi4py import MPI
import Pypdm.Pypdm as PDM

from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils                  import py_utils, np_utils
from maia.utils                  import logging as mlog
from maia.utils                  import vstride as vs
from maia.transfer               import utils as te_utils
from maia.factory.dist_from_part import get_parts_per_blocks
from maia.pytree.maia.check_tree import check_cgns_part_tree

from .import point_cloud_utils as PCU
from .import multidom_gnum     as MDG
from .import localize as LOC
from .import closest_points as CLO

class Interpolator:
  """ Low level class to perform interpolations """
  def __init__(self, 
               src_parts_per_dom: List[List[CGNSPartTree]], 
               tgt_parts_per_dom: List[List[CGNSPartTree]], 
               src_to_tgt: Any,
               input_loc: str,
               output_loc: str,
               comm: MPIComm) -> None:
    self.src_parts = py_utils.to_flat_list(src_parts_per_dom) 
    self.tgt_parts = py_utils.to_flat_list(tgt_parts_per_dom) 
    
    self.output_loc = output_loc
    self.input_loc = input_loc
    self.comm = comm

    # If some rank have no partitions, store a rank used as root to share FS names
    self.root = None
    if comm.allreduce(len(self.src_parts) == 0, MPI.LOR):
      self.root = self.comm.allreduce(-1 if len(self.src_parts) == 0 else comm.rank, MPI.MAX)

    _, src_lngn_per_dom = MDG.get_shifted_ln_to_gn_from_loc(src_parts_per_dom, self.input_loc, comm)
    all_src_lngn = py_utils.to_flat_list(src_lngn_per_dom)

    _, tgt_lngn_per_dom = MDG.get_shifted_ln_to_gn_from_loc(tgt_parts_per_dom, self.output_loc, comm)
    all_tgt_lngn = py_utils.to_flat_list(tgt_lngn_per_dom)

    self.src_to_tgt_idx = [data['target_gnum'].displs for data in src_to_tgt]
    _src_to_tgt = [data['target_gnum'].values for data in src_to_tgt]
    self.PTP = PDM.PartToPart(comm,
                              all_src_lngn,
                              all_tgt_lngn,
                              self.src_to_tgt_idx,
                              _src_to_tgt)

    self.referenced_nums = self.PTP.get_referenced_lnum2()
    self.sending_gnums = self.PTP.get_gnum1_come_from()

    # Send weight to targets partitions (if available)
    try:
      _weight = [data['target_weight'].values for data in src_to_tgt]
      request = self.PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                               _weight)
      _, self.tgt_weight = self.PTP.wait(request)
    except KeyError:
      pass


  def _reduce_single_val(self, i_part: int, data: NDArray) -> NDArray:
    """
    A basic reduce function who take the first received value for each target
    """
    come_from_idx = self.sending_gnums[i_part]['come_from_idx']
    return data[come_from_idx[:-1]]

  def _reduce_weighted_mean(self, i_part: int, data: NDArray) -> NDArray:
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


  def exchange_fields(self, 
                      container_name: str, 
                      reduce_func: Callable[['Interpolator', int, NDArray], NDArray] = _reduce_weighted_mean) -> None:
    """
    For all fields found under container_name node,
    - Perform a part to part exchange
    - Reduce the received data using reduce_func (because tgt elements can receive multiple data)
    - Fill the target sol with a default value + the reduced value
    """

    #Check that solutions are known on each source partition
    fields_per_part:List[List[str]] = list()
    for src_part in self.src_parts:
      container = PT.request_node_from_path(src_part, container_name)
      assert PT.Subset.GridLocation(container) == self.input_loc
      fields_name = sorted([PT.get_name(array) for array in PT.iter_children_from_label(container, 'DataArray_t')])
      fields_per_part.append(fields_name)
    if len(fields_per_part) > 0:
      assert fields_per_part.count(fields_per_part[0]) == len(fields_per_part)

    fields_names = fields_per_part[0] if len(fields_per_part) > 0 else None
    if self.root is not None: # Some rank have no src partitions, share field names
      fields_names = self.comm.bcast(fields_names, root=self.root)
    assert fields_names is not None

    #Cleanup target partitions
    for tgt_part in self.tgt_parts:
      PT.rm_children_from_name(tgt_part, container_name)
      fs = PT.new_FlowSolution(container_name, loc=self.output_loc, parent=tgt_part)

    #Collect src sol
    src_field_dic = dict()
    for field_name in fields_names:
      field_path = container_name + '/' + field_name
      src_field_dic[field_name] = [PT.request_node_from_path(part, field_path)[1] for part in self.src_parts]

    #Exchange
    for field_name, src_sol in src_field_dic.items():
      request = self.PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1,
                               src_sol)
      strides, lnp_part_data = self.PTP.wait(request)

      for i_part, tgt_part in enumerate(self.tgt_parts):
        fs = PT.get_node_from_path(tgt_part, container_name)
        data_size = PT.Zone.n_cell(tgt_part) if self.output_loc == 'CellCenter' else PT.Zone.n_vtx(tgt_part)
        data = np.nan * np.ones(data_size)
        come_from_idx = self.sending_gnums[i_part]['come_from_idx']
        if (np.diff(come_from_idx) == 1).all():
          reduced_data = lnp_part_data[i_part]
        else:
          reduced_data = reduce_func(self, i_part, lnp_part_data[i_part])
        data[self.referenced_nums[i_part]-1] = reduced_data #Use referenced ids to erase default value
        if PT.Zone.Type(tgt_part) == 'Unstructured':
          PT.update_child(fs, field_name, 'DataArray_t', data)
        else:
          shape = PT.Zone.CellSize(tgt_part) if self.output_loc == 'CellCenter' else PT.Zone.VertexSize(tgt_part)
          PT.update_child(fs, field_name, 'DataArray_t', data.reshape(shape, order='F'))


def _cell_tgt_to_vtx_tgt(cell_vtx: vs.VStrideArray,
                         cell_tgt: vs.VStrideArray,
                         cell_vtx_weight:NDArray,
                         n_vtx:int) -> Tuple[vs.VStrideArray, vs.VStrideArray]:
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


def create_src_to_tgt(src_parts_per_dom:List[List[CGNSPartTree]],
                      tgt_parts_per_dom:List[List[CGNSPartTree]],
                      comm:MPIComm,
                      src_loc:Literal['CellCenter', 'Vertex'] = 'CellCenter',
                      tgt_loc:Literal['CellCenter', 'Vertex'] = 'CellCenter',
                      strategy:str = 'Closest',
                      loc_tolerance:float = 1E-6,
                      n_closest_pt:int = 1):
  """ Create a source to target indirection depending of the choosen strategy.

  This indirection can then be used to create an interpolator object.
  """

  assert strategy in ['LocationAndClosest', 'Location', 'Closest']


  #Phase 1 -- localisation
  if strategy != 'Closest':
    all_n_vtx = [PT.Zone.n_vtx(zone) for src_parts in src_parts_per_dom for zone in src_parts]

    location_out, location_out_inv = LOC._localize_points(src_parts_per_dom, tgt_parts_per_dom, \
        tgt_loc, comm, True, loc_tolerance)

    # output is nested by domain so we need to flatten it
    all_unlocated = [data['unlocated_ids'] for domain in location_out for data in domain]
    all_located_inv = py_utils.to_flat_list(location_out_inv)
    n_unlocated = sum([t.size for t in all_unlocated])
    n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.SUM)
    if comm.Get_rank() == 0:
      mlog.stat(f"[interpolation] Number of unlocated points for Location method is {n_tot_unlocated}")


  all_closest_inv:List[Dict[str, vs.VStrideArray]] = list()
  if strategy == 'Closest' or (strategy == 'LocationAndClosest' and n_tot_unlocated > 0):

    # > Setup source for closest point (with shift to manage multidomain)
    _, src_clouds_nested = PCU.get_shifted_point_clouds(src_parts_per_dom, src_loc, comm)
    src_clouds = py_utils.to_flat_list(src_clouds_nested)

    # > Setup target for closest point (with shift to manage multidomain)
    _, tgt_clouds_nested = PCU.get_shifted_point_clouds(tgt_parts_per_dom, tgt_loc, comm)
    tgt_clouds = py_utils.to_flat_list(tgt_clouds_nested)

    # > If we previously did a mesh location, we only treat unlocated points : create a sub global numbering
    if strategy != 'Closest':
      assert len(all_unlocated) == len(tgt_clouds)
      sub_clouds = [PCU.extract_sub_cloud(*tgt_cloud, all_unlocated[i]) for i,tgt_cloud in enumerate(tgt_clouds)]
      all_extracted_lngn = [sub_cloud[1] for sub_cloud in sub_clouds]
      all_sub_lngn = PCU.create_sub_numbering(all_extracted_lngn, comm) #This one is collective
      tgt_clouds = [(tgt_cloud[0], sub_lngn) for tgt_cloud, sub_lngn in zip(sub_clouds, all_sub_lngn)]

    n_clo = n_closest_pt
    all_closest, all_closest_inv = CLO._closest_points(src_clouds, tgt_clouds, comm, n_clo, reverse=True)

    #If we worked on sub gnum, we must go back to original numbering
    if strategy != 'Closest':
      gnum_to_transform = [results["tgt_in_src"].values for results in all_closest_inv]
      PDM.transform_to_parent_gnum(gnum_to_transform, all_sub_lngn, all_extracted_lngn, comm)

  dist2weight = lambda V : vs.from_displs(V.displs, 1. / np.maximum(V.values, 1E-20))
  # Combine Location & Closest results if both method were used
  if strategy == 'Location' or (strategy == 'LocationAndClosest' and n_tot_unlocated == 0):
    if src_loc=="CellCenter":
      
      src_to_tgt = [{'target_gnum' : data['points_gnum_shifted']} 
                    for data in all_located_inv]
    elif src_loc=="Vertex":
      src_to_tgt = list()
      for data, n_vtx in zip(all_located_inv, all_n_vtx):

        vtx_to_tgt, vtx_to_weight = _cell_tgt_to_vtx_tgt(data['cell_vtx'],
                                                         data['points_gnum_shifted'],
                                                         data['points_weights'].values,      
                                                         n_vtx)
        src_to_tgt.append({'target_gnum' :vtx_to_tgt, 'target_weight' : vtx_to_weight})
        
  elif strategy == 'Closest':
    src_to_tgt = [{'target_gnum' : data['tgt_in_src'],
                   'target_weight' : dist2weight(data['tgt_in_src_dist2'])}
                   for data in all_closest_inv]
  else:
    src_to_tgt = []

    for res_loc, n_vtx, res_clo in zip(all_located_inv, all_n_vtx, all_closest_inv):
      clo_tgt_in_src = res_clo['tgt_in_src']
      clo_weight     = dist2weight(res_clo['tgt_in_src_dist2'])

      if src_loc=="CellCenter":
        loc_src_to_tgt = res_loc['points_gnum_shifted']
        loc_weight     = vs.from_displs(loc_src_to_tgt.displs, np.ones(loc_src_to_tgt.dsize))
      elif src_loc=="Vertex": # Move results of mesh location from cell to vtx
        loc_src_to_tgt, loc_weight = _cell_tgt_to_vtx_tgt(res_loc['cell_vtx'],
                                                          res_loc['points_gnum_shifted'],
                                                          res_loc['points_weights'].values,      #cell_vtx_weight
                                                          n_vtx)

        
      tgt_in_src_vs = vs.concatenate([loc_src_to_tgt, clo_tgt_in_src], vs.INNER_AXIS)
      tgt_weight_vs = vs.concatenate([loc_weight, clo_weight],         vs.INNER_AXIS)

      src_to_tgt.append({'target_gnum' :tgt_in_src_vs, 'target_weight':tgt_weight_vs})

  return src_to_tgt



def interpolate_from_parts_per_dom(src_parts_per_dom,
                                   tgt_parts_per_dom,
                                   comm,
                                   containers_name,
                                   location,
                                   **options):
  """
  Low level interface for interpolation
  Input are a list of partitioned zones for each src domain, and a list of partitioned zone for each tgt
  domain. Lists mush be coherent across procs, ie we must have an empty entry if a proc does not know a domain.

  containers_name is the list of FlowSolution containers to be interpolated
  location is the output location (CellCenter or Vertex); input location can be Vertex only if 
  strategy is 'Closest', otherwise it must be CellCenter
  **options are passed to interpolator creationg function, see create_src_to_tgt
  """
  # Guess location of input fields
  if len(containers_name) == 0:
    return
  try:
    first_part = next(part for dom in src_parts_per_dom for part in dom)
    input_loc = PT.Subset.GridLocation(PT.request_child_from_name(first_part, containers_name[0]))
  except StopIteration:
    input_loc = ''
  input_loc = comm.allreduce(input_loc, op=MPI.MAX)

  src_to_tgt = create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, comm, input_loc, location, **options)

  interpolator = Interpolator(src_parts_per_dom, tgt_parts_per_dom, src_to_tgt, input_loc, location, comm)
  for container_name in containers_name:
    interpolator.exchange_fields(container_name)

def interpolate(src_tree: CGNSPartTree,
                tgt_tree: CGNSPartTree, 
                comm: MPIComm,
                containers_name: List[str], 
                location: Literal['CellCenter', 'Vertex'],
                **options) -> None:
  """Interpolate fields between two partitionned trees.

  This function can transfer CellCenter or Vertex located fields, but not both
  at the same time.
  Target tree is modified inplace: the requested FlowSolution_t containers are transfered
  from the source tree.

  Interpolation strategy can be controled thought the options kwargs:

  - ``strategy`` (default = 'Closest') -- control interpolation method

    - 'Closest' : Target points use the inverse distance weighting on the ``n_closest_pt`` source point values.
    - 'Location' : For ``CellCenter`` fields, target points take the value of the cell in which they are located.
      For ``Vertex`` fields, target points use finite element weights of source cell vertices to compute interpolation.
      In both cases, unlocated points take the value ``NaN``.
    - 'LocationAndClosest' : Use 'Location' method and then 'ClosestPoint' method
      for the unlocated points.

  - ``n_closest_pt`` (default = 1) -- If strategy is 'Closest' or 'LocationAndClosest', 
    specify the number of closest points used for interpolation.

  - ``loc_tolerance`` (default = 1E-6) -- Geometric tolerance for Location method.

  See also:
    :func:`create_interpolator` takes the same parameters (excepted ``containers_name``,
    which must be replaced by ``src_location``), and returns an Interpolator object which can be used
    to exchange containers more than once through its ``Interpolator.exchange_fields(container_name)`` method.

  Args:
    src_tree (CGNSPartTree): Source tree, partitionned. Only 3D unstructured connectivities are managed.
    tgt_tree (CGNSPartTree): Target tree, partitionned. Structured or unstructured connectivities are managed.
    comm       (MPIComm)   : MPI communicator
    containers_name (list of str) : List of the names of the source FlowSolution_t nodes to transfer.
    location ({'CellCenter', 'Vertex'}) : Expected target location of the fields.
    **options: Options related to interpolation strategy

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #interpolate@start
        :end-before: #interpolate@end
        :dedent: 2
  """
  check_cgns_part_tree(src_tree)
  check_cgns_part_tree(tgt_tree)
  src_parts_per_dom = list(get_parts_per_blocks(src_tree, comm).values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  interpolate_from_parts_per_dom(src_parts_per_dom, tgt_parts_per_dom, comm, containers_name, location, **options)


def create_interpolator(
  src_tree: CGNSPartTree, 
  tgt_tree: CGNSPartTree,
  comm: MPIComm, 
  src_location: Literal['CellCenter', 'Vertex'],
  location: Literal['CellCenter', 'Vertex'],
  **options
) -> Interpolator:
  """Same as interpolate, but return the interpolator object instead
  of doing interpolations. Interpolator can be called multiple time to exchange
  fields without recomputing the src_to_tgt indirection (geometry must remain the same).
  """
  check_cgns_part_tree(src_tree)
  check_cgns_part_tree(tgt_tree)
  src_parts_per_dom = list(get_parts_per_blocks(src_tree, comm).values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  src_to_tgt = create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, comm, src_location, location, **options)
  return Interpolator(src_parts_per_dom, tgt_parts_per_dom, src_to_tgt, src_location, location, comm)
