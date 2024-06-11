from mpi4py import MPI
import numpy as np
import warnings

import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils                  import py_utils, np_utils
from maia.utils                  import logging as mlog
from maia.transfer               import utils as te_utils
from maia.factory.dist_from_part import get_parts_per_blocks

from .import point_cloud_utils as PCU
from .import multidom_gnum     as MDG
from .import localize as LOC
from .import closest_points as CLO

class Interpolator:
  """ Low level class to perform interpolations """
  def __init__(self, src_parts_per_dom, tgt_parts_per_dom, src_to_tgt, input_loc, output_loc, comm):
    self.src_parts = py_utils.to_flat_list(src_parts_per_dom) 
    self.tgt_parts = py_utils.to_flat_list(tgt_parts_per_dom) 
    
    self.output_loc = output_loc
    self.input_loc = input_loc

    _, src_lngn_per_dom = MDG.get_shifted_ln_to_gn_from_loc(src_parts_per_dom, self.input_loc, comm)
    all_src_lngn = py_utils.to_flat_list(src_lngn_per_dom)

    _, tgt_lngn_per_dom = MDG.get_shifted_ln_to_gn_from_loc(tgt_parts_per_dom, self.output_loc, comm)
    all_tgt_lngn = py_utils.to_flat_list(tgt_lngn_per_dom)

    self.src_to_tgt_idx = [data['target_idx'] for data in src_to_tgt]
    _src_to_tgt = [data['target'] for data in src_to_tgt]
    self.PTP = PDM.PartToPart(comm,
                              all_src_lngn,
                              all_tgt_lngn,
                              self.src_to_tgt_idx,
                              _src_to_tgt)

    self.referenced_nums = self.PTP.get_referenced_lnum2()
    self.sending_gnums = self.PTP.get_gnum1_come_from()

    # Send weight to targets partitions (if available)
    try:
      _weight = [data['target_weight'] for data in src_to_tgt]
      request = self.PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                               PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                               _weight)
      _, self.tgt_weight = self.PTP.wait(request)
    except KeyError:
      pass


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
    weight_threshold = np.minimum(self.tgt_weight[i_part], 1E+20)
    reduced_data   = np.add.reduceat(data*weight_threshold, come_from_idx[:-1])
    reduced_factor = np.add.reduceat(     weight_threshold, come_from_idx[:-1])
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
    for src_part in self.src_parts:
      container = PT.get_node_from_path(src_part, container_name)
      assert PT.Subset.GridLocation(container) == self.input_loc
      fields_name = sorted([PT.get_name(array) for array in PT.iter_children_from_label(container, 'DataArray_t')])
    fields_per_part.append(fields_name)
    assert fields_per_part.count(fields_per_part[0]) == len(fields_per_part)

    #Cleanup target partitions
    for tgt_part in self.tgt_parts:
      PT.rm_children_from_name(tgt_part, container_name)
      fs = PT.new_FlowSolution(container_name, loc=self.output_loc, parent=tgt_part)

    #Collect src sol
    src_field_dic = dict()
    for field_name in fields_per_part[0]:
      field_path = container_name + '/' + field_name
      src_field_dic[field_name] = [PT.get_node_from_path(part, field_path)[1] for part in self.src_parts]

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


def _cell_tgt_to_vtx_tgt(cell_vtx_idx, cell_vtx, cell_tgt_idx, cell_tgt, cell_vtx_weight, n_vtx):
  '''
  Transform cell->tgt (src_to_tgt, src_vtx_weight) information from mesh_location
  onto vtx->tgt information.
  '''
  # > Generate cell_vtx_idx + cell_vtx of cell which have tgt (duplicated if multiple tgt)
  cell_id     = np.arange(0, cell_vtx_idx.size-1, dtype=np.int32)
  cell_n_tgt  = np.diff(cell_tgt_idx) # number of tgt in cell
  active_cell = np.repeat(cell_id, cell_n_tgt) # id of cell having some tgt (duplicated if multiple tgt)
  active_cell_vtx_ids = np_utils.multi_arange(cell_vtx_idx[active_cell], cell_vtx_idx[active_cell+1]) # id of vtx in cell_vtx connectivity
  
  cell_n_vtx = np.diff(cell_vtx_idx) # number of vtx in cell
  active_cell_n_vtx = cell_n_vtx[active_cell]
  active_cell_vtx_idx = np_utils.sizes_to_indices(active_cell_n_vtx) # cell_vtx_idx of cells with tgt
  active_cell_vtx     = cell_vtx[active_cell_vtx_ids] # cell_vtx of cells with tgt

  cell_tgt_extended = np.repeat(cell_tgt, np.diff(active_cell_vtx_idx)) # cell_vtx->tgt
  sort_idx = np.argsort(active_cell_vtx)
  vtx_to_tgt_n = np.zeros(n_vtx, dtype=np.int32)
  np.add.at(vtx_to_tgt_n, active_cell_vtx-1, 1)

  vtx_to_tgt_idx = np_utils.sizes_to_indices(vtx_to_tgt_n) 
  vtx_to_tgt     = cell_tgt_extended[sort_idx] # vtx->tgt
  vtx_to_weight  = cell_vtx_weight[sort_idx]

  return vtx_to_tgt_idx, vtx_to_tgt, vtx_to_weight


def create_src_to_tgt(src_parts_per_dom,
                      tgt_parts_per_dom,
                      comm,
                      src_loc = 'CellCenter',
                      tgt_loc = 'CellCenter',
                      strategy = 'Closest',
                      loc_tolerance = 1E-6,
                      n_closest_pt = 1):
  """ Create a source to target indirection depending of the choosen strategy.

  This indirection can then be used to create an interpolator object.
  """

  assert strategy in ['LocationAndClosest', 'Location', 'Closest']


  #Phase 1 -- localisation
  if strategy != 'Closest':
    all_n_vtx = list()
    for i_domain, src_part_zones in enumerate(src_parts_per_dom):
      all_n_vtx.append([PT.Zone.n_vtx(zone) for zone in src_part_zones])
    all_n_vtx = py_utils.to_flat_list(all_n_vtx)

    location_out, location_out_inv, location_cell_vtx = LOC._localize_points(src_parts_per_dom, tgt_parts_per_dom, \
        tgt_loc, comm, True, loc_tolerance)

    # output is nested by domain so we need to flatten it
    all_unlocated = [data['unlocated_ids'] for domain in location_out for data in domain]
    all_located_inv = py_utils.to_flat_list(location_out_inv)
    all_located_cell_vtx = py_utils.to_flat_list(location_cell_vtx)
    n_unlocated = sum([t.size for t in all_unlocated])
    n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.SUM)
    if comm.Get_rank() == 0:
      mlog.stat(f"[interpolation] Number of unlocated points for Location method is {n_tot_unlocated}")


  all_closest_inv = list()
  if strategy == 'Closest' or (strategy == 'LocationAndClosest' and n_tot_unlocated > 0):

    # > Setup source for closest point (with shift to manage multidomain)
    _, src_clouds = PCU.get_shifted_point_clouds(src_parts_per_dom, src_loc, comm)
    src_clouds = py_utils.to_flat_list(src_clouds)

    # > Setup target for closest point (with shift to manage multidomain)
    _, tgt_clouds = PCU.get_shifted_point_clouds(tgt_parts_per_dom, tgt_loc, comm)
    tgt_clouds = py_utils.to_flat_list(tgt_clouds)

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
      gnum_to_transform = [results["tgt_in_src"] for results in all_closest_inv]
      PDM.transform_to_parent_gnum(gnum_to_transform, all_sub_lngn, all_extracted_lngn, comm)

  # Combine Location & Closest results if both method were used
  if strategy == 'Location' or (strategy == 'LocationAndClosest' and n_tot_unlocated == 0):
    if src_loc=="CellCenter":
      src_to_tgt = [{'target_idx' : data['elt_pts_inside_idx'],
                     'target'     : data['points_gnum_shifted']} for data in all_located_inv]
    elif src_loc=="Vertex":
      src_to_tgt = list()
      for data, cell_vtx, n_vtx in zip(all_located_inv, all_located_cell_vtx, all_n_vtx):
        
        cell_tgt_idx    = data['elt_pts_inside_idx']
        cell_tgt        = data['points_gnum_shifted']
        cell_vtx_weight = data['points_weights']

        cell_vtx_idx = cell_vtx['cell_vtx_idx']
        cell_vtx     = cell_vtx['cell_vtx']
        
        vtx_to_tgt_idx, vtx_to_tgt, vtx_to_weight = _cell_tgt_to_vtx_tgt(cell_vtx_idx, cell_vtx, cell_tgt_idx, cell_tgt, cell_vtx_weight, n_vtx)
        src_to_tgt.append({'target_idx'   : vtx_to_tgt_idx,
                           'target'       : vtx_to_tgt,
                           'target_weight': vtx_to_weight})
  elif strategy == 'Closest':
    src_to_tgt = [{'target_idx'   :    data['tgt_in_src_idx'],
                   'target'       :    data['tgt_in_src'],
                   'target_weight': 1./data['tgt_in_src_dist2']} for data in all_closest_inv]
  else:
    src_to_tgt = []

    if src_loc=="CellCenter":
      for res_loc, res_clo in zip(all_located_inv, all_closest_inv):
        loc_src_weight = np.ones(res_loc['points_gnum_shifted'].size, dtype=np.float64)
        clo_src_weight = 1./res_clo['tgt_in_src_dist2']
        tgt_in_src_idx, tgt_in_src = np_utils.jagged_merge(res_loc['elt_pts_inside_idx'], res_loc['points_gnum_shifted'], \
                                                           res_clo[    'tgt_in_src_idx'], res_clo['tgt_in_src'])
        tgt_in_src_idx, tgt_weight = np_utils.jagged_merge(res_loc['elt_pts_inside_idx'], loc_src_weight, \
                                                           res_clo[    'tgt_in_src_idx'], clo_src_weight)
        src_to_tgt.append({'target_idx' :tgt_in_src_idx, 'target' :tgt_in_src, 'target_weight':tgt_weight})

    elif src_loc=="Vertex":
      for res_loc, cell_vtx, n_vtx, res_clo in zip(all_located_inv, all_located_cell_vtx, all_n_vtx, all_closest_inv):
        cell_tgt_idx    = res_loc['elt_pts_inside_idx']
        cell_tgt        = res_loc['points_gnum_shifted']
        cell_vtx_weight = res_loc['points_weights']

        cell_vtx_idx = cell_vtx['cell_vtx_idx']
        cell_vtx     = cell_vtx['cell_vtx']
        
        loc_vtx_to_tgt_idx, loc_vtx_to_tgt, loc_vtx_to_weight = _cell_tgt_to_vtx_tgt(cell_vtx_idx, cell_vtx, cell_tgt_idx, cell_tgt, cell_vtx_weight, n_vtx)
        
        clo_src_weight = 1./res_clo['tgt_in_src_dist2']

        tgt_in_src_idx, tgt_in_src = np_utils.jagged_merge(loc_vtx_to_tgt_idx, loc_vtx_to_tgt, \
                                                           res_clo['tgt_in_src_idx'], res_clo['tgt_in_src'])
        tgt_in_src_idx, tgt_weight = np_utils.jagged_merge(loc_vtx_to_tgt_idx, loc_vtx_to_weight, \
                                                           res_clo['tgt_in_src_idx'], clo_src_weight)
        src_to_tgt.append({'target_idx' :tgt_in_src_idx, 'target' :tgt_in_src, 'target_weight':tgt_weight})
  
  return src_to_tgt



def interpolate_from_parts_per_dom(src_parts_per_dom, tgt_parts_per_dom, comm, containers_name, location, **options):
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
    input_loc = PT.Subset.GridLocation(PT.get_child_from_name(first_part, containers_name[0]))
  except StopIteration:
    input_loc = ''
  input_loc = comm.allreduce(input_loc, op=MPI.MAX)

  src_to_tgt = create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, comm, input_loc, location, **options)

  interpolator = Interpolator(src_parts_per_dom, tgt_parts_per_dom, src_to_tgt, input_loc, location, comm)
  for container_name in containers_name:
    interpolator.exchange_fields(container_name)

def interpolate(src_tree, tgt_tree, comm, containers_name, location, **options):
  """Interpolate fields between two partitionned trees.

  For now, interpolation is limited to lowest order: target points take the value of the
  closest point (or their englobing cell, depending of choosed options) in the source mesh.
  Interpolation strategy can be controled thought the options kwargs:

  - ``strategy`` (default = 'Closest') -- control interpolation method

    - 'Closest' : Target points take the value of the closest source cell center.
    - 'Location' : Target points take the value of the cell in which they are located.
      Unlocated points have take a ``NaN`` value.
    - 'LocationAndClosest' : Use 'Location' method and then 'ClosestPoint' method
      for the unlocated points.

  - ``loc_tolerance`` (default = 1E-6) -- Geometric tolerance for Location method.

  Important:
    If ``strategy`` is not 'Closest', source tree must have an unstructured-NGON
    connectivity and CellCenter located fields.

  See also:
    :func:`create_interpolator` takes the same parameters (excepted ``containers_name``,
    which must be replaced by ``src_location``), and returns an Interpolator object which can be used
    to exchange containers more than once through its ``Interpolator.exchange_fields(container_name)`` method.

  Args:
    src_tree (CGNSTree): Source tree, partitionned. Only 3D U-NGon connectivities are managed.
    tgt_tree (CGNSTree): Target tree, partitionned. Structured or unstructured connectivities are managed.
    comm       (MPIComm): MPI communicator
    containers_name (list of str) : List of the names of the source FlowSolution_t nodes to transfer.
    location ({'CellCenter', 'Vertex'}) : Expected target location of the fields.
    **options: Options related to interpolation strategy

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #interpolate@start
        :end-before: #interpolate@end
        :dedent: 2
  """
  src_parts_per_dom = list(get_parts_per_blocks(src_tree, comm).values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  interpolate_from_parts_per_dom(src_parts_per_dom, tgt_parts_per_dom, comm, containers_name, location, **options)

def interpolate_from_part_trees(src_tree, tgt_tree, comm, containers_name, location, **options):
  warnings.warn("This function is deprecated in favor of interpolate, and will be removed in next release",
    DeprecationWarning, stacklevel=2)
  return interpolate(src_tree, tgt_tree, comm, containers_name, location, **options)

def create_interpolator(src_tree, tgt_tree, comm, src_location, location, **options):
  """Same as interpolate, but return the interpolator object instead
  of doing interpolations. Interpolator can be called multiple time to exchange
  fields without recomputing the src_to_tgt indirection (geometry must remain the same).
  """
  src_parts_per_dom = list(get_parts_per_blocks(src_tree, comm).values())
  tgt_parts_per_dom = list(get_parts_per_blocks(tgt_tree, comm).values())

  src_to_tgt = create_src_to_tgt(src_parts_per_dom, tgt_parts_per_dom, comm, src_location, location, **options)
  return Interpolator(src_parts_per_dom, tgt_parts_per_dom, src_to_tgt, src_location, location, comm)


def create_interpolator_from_part_trees(src_tree, tgt_tree, comm, src_location, location, **options):
  warnings.warn("This function is deprecated in favor of create_interpolator, and will be removed in next release",
    DeprecationWarning, stacklevel=2)
  return create_interpolator(src_tree, tgt_tree, comm, src_location, location, **options)
