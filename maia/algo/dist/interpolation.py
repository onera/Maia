from mpi4py import MPI
import numpy as np

import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT


from maia.utils                  import np_utils
from maia.utils                  import logging as mlog
from maia.utils                  import vstride as vs
import maia.transfer.protocols as EP

from maia.algo.part import point_cloud_utils as PCU
from .import localize as LOC
from .import closest_points as CLO


class Interpolator:
  """ Low level class to perform interpolations """
  def __init__(self, src_dom, tgt_dom, src_to_tgt, input_loc, output_loc, comm):
    self.src_parts = src_dom
    self.tgt_parts = tgt_dom
    
    self.output_loc = output_loc
    self.input_loc = input_loc
    self.comm = comm

    all_src_lngn = []
    offset = 0
    for zone in src_dom:
      distri_name = 'Cell' if input_loc == 'CellCenter' else 'Vertex'
      distri = MT.get_distribution(zone, distri_name)[1]
      gnum = np.arange(distri[0]+1+offset, distri[1]+1+offset, dtype=distri.dtype)
      if input_loc == 'CellCenter':
        offset += PT.Zone.n_cell(zone)
      else:
        offset += PT.Zone.n_vtx(zone)
      all_src_lngn.append(gnum)

    all_tgt_lngn = []
    offset = 0
    for zone in tgt_dom:
      distri_name = 'Cell' if output_loc == 'CellCenter' else 'Vertex'
      distri = MT.get_distribution(zone, distri_name)[1]
      gnum = np.arange(distri[0]+1+offset, distri[1]+1+offset, dtype=distri.dtype)
      if output_loc == 'CellCenter':
        offset += PT.Zone.n_cell(zone)
      else:
        offset += PT.Zone.n_vtx(zone)
      all_tgt_lngn.append(gnum)


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
    for src_part in self.src_parts:
      container = PT.get_node_from_path(src_part, container_name)
      assert PT.Subset.GridLocation(container) == self.input_loc
      fields_name = sorted([PT.get_name(array) for array in PT.iter_children_from_label(container, 'DataArray_t')])
      fields_per_part.append(fields_name)
    if len(fields_per_part) > 0:
      assert fields_per_part.count(fields_per_part[0]) == len(fields_per_part)

    fields_names = fields_per_part[0] if len(fields_per_part) > 0 else None
    #if self.root is not None: # Some rank have no src partitions, share field names
      #fields_names = self.comm.bcast(fields_names, root=self.root)

    #Cleanup target partitions
    for tgt_part in self.tgt_parts:
      PT.rm_children_from_name(tgt_part, container_name)
      fs = PT.new_FlowSolution(container_name, loc=self.output_loc, parent=tgt_part)

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
      strides, lnp_part_data = self.PTP.wait(request)

      for i_part, tgt_part in enumerate(self.tgt_parts):
        fs = PT.get_node_from_path(tgt_part, container_name)
        distri_name = 'Cell' if self.output_loc == 'CellCenter' else 'Vertex'
        distri = MT.get_distribution(tgt_part, distri_name)[1]
        data_size = distri[1] - distri[0]
        data = np.nan * np.ones(data_size)
        come_from_idx = self.sending_gnums[i_part]['come_from_idx']
        if (np.diff(come_from_idx) == 1).all():
          reduced_data = lnp_part_data[i_part]
        else:
          reduced_data = reduce_func(self, i_part, lnp_part_data[i_part])
        data[self.referenced_nums[i_part]-1] = reduced_data #Use referenced ids to erase default value
        PT.update_child(fs, field_name, 'DataArray_t', data)


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


def create_src_to_tgt(src_dom,
                      tgt_dom,
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

  location_out_inv = [] # Init to avoid unbound error
  closest_out_inv  = []

  #Phase 1 -- localisation
  if strategy != 'Closest':

    # Use midlevel API because we need the number of vertices on the "fake" partitions
    src_parts = LOC._collect_source(src_dom, comm)
    tgt_clouds = LOC._collect_target(tgt_dom, tgt_loc, comm)
    location_out, location_out_inv = LOC._mdom_mesh_location(src_parts, tgt_clouds, \
        comm, True, loc_tolerance)

    # output is nested by domain so we need to flatten it
    n_unlocated = sum([data['unlocated_ids'].size for data in location_out])
    n_tot_unlocated = comm.allreduce(n_unlocated, op=MPI.SUM)
    if comm.Get_rank() == 0:
      mlog.stat(f"[interpolation] Number of unlocated points for Location method is {n_tot_unlocated}")

    """
    # Write fake partitions (debug)
    part = src_parts[0][2]
    tree = PT.new_CGNSTree()
    base = PT.new_CGNSBase(parent=tree)
    zone = PT.new_Zone(f'FakePart.P{comm.rank}.N0', size=[[part[7].size,part[0].size-1,0]], type='Unstructured', parent=base)
    PT.new_GridCoordinates(fields={'CoordinateX' : part[6][0::3], 'CoordinateY' : part[6][1::3], 'CoordinateZ' : part[6][2::3],}, parent=zone)
    PT.new_NGonElements(erange=[1,part[3].size-1], eso=part[3], ec=part[4], parent=zone)
    PT.new_NFaceElements(erange=[part[3].size, part[3].size-1 + part[0].size-1], eso=part[0], ec=part[1], parent=zone)
    import maia
    maia.io.write_trees(tree, 'fake_part.cgns', comm)
    """
    if src_loc=="Vertex":
      # Move results of mesh location from cell to vtx
      for src_zone, src_part, data in zip(src_dom, src_parts, location_out_inv):
        fake_part = src_part[2]
        vtx_gnum = fake_part[-1]
        vtx_to_tgt, vtx_to_weight = _cell_tgt_to_vtx_tgt(data['cell_vtx'],
                                                         data['points_gnum_shifted'],
                                                         data['points_weights'].values,      
                                                         vtx_gnum.size)
        part_data = {'points_gnum_shifted@VTX' : vtx_to_tgt,
                    'points_weights@VTX'       : vtx_to_weight}
        # Careful : vertex case, the "fake partition"  vertices are not equal to the implicit distributed vertices
        # (they are local and reordered). Thus we need to move the vtx output on the distributed vtx view
        # This exchange is done with append mode because we want to merge the located data coming from different
        # partitions, and data is initially computed from cell point of view before beeing moved to vertices
        # We dont need this in Cell mode because "fake partition" cell lngn is equal to the cell distribution
        distri_vtx = MT.getDistribution(src_zone, 'Vertex')[1]
        data.update(EP.part_to_block(part_data, distri_vtx, vtx_gnum-1, comm, append=True))


  #Phase 2 -- closest point
  if strategy == 'Closest' or (strategy == 'LocationAndClosest' and n_tot_unlocated > 0):

    # We hook midlevel API to filter some target points (the one already located)
    src_clouds = [LOC.get_point_cloud(zone, comm, src_loc) for zone in src_dom]
    tgt_clouds = [LOC.get_point_cloud(zone, comm, tgt_loc) for zone in tgt_dom]
    tgt_need_shift = False
    if strategy != 'Closest':
      tgt_need_shift = True
      tgt_clouds = [PCU.extract_sub_cloud(*cloud, location_out[j]['unlocated_ids']) for j,cloud in enumerate(tgt_clouds)]

    _, closest_out_inv = CLO._mdom_closest_points(src_clouds, tgt_clouds, comm, n_pts=n_closest_pt, reverse=True, need_shift=tgt_need_shift)

  dist2weight = lambda V : vs.from_displs(V.displs, 1. / np.maximum(V.values, 1E-20))
  all_located_inv = location_out_inv
  all_closest_inv = closest_out_inv
  #Phase 3 : Combine Location & Closest results if both method were used
  if strategy == 'Location' or (strategy == 'LocationAndClosest' and n_tot_unlocated == 0):
    if src_loc=="CellCenter":
      
      src_to_tgt = [{'target_gnum' : data['points_gnum_shifted']} 
                    for data in all_located_inv]
    elif src_loc=="Vertex":
      src_to_tgt = [{'target_gnum' :data['points_gnum_shifted@VTX'],
                     'target_weight' : data['points_weights@VTX']}
                    for data in all_located_inv]
      
        
  elif strategy == 'Closest':
    src_to_tgt = [{'target_gnum' : data['tgt_in_src_shifted'],
                   'target_weight' : dist2weight(data['tgt_in_src_dist2'])}
                   for data in all_closest_inv]
  else:
    src_to_tgt = []

    for res_loc, res_clo in zip(all_located_inv, all_closest_inv):
      clo_tgt_in_src = res_clo['tgt_in_src_shifted']
      clo_weight     = dist2weight(res_clo['tgt_in_src_dist2'])

      if src_loc=="CellCenter":
        loc_src_to_tgt = res_loc['points_gnum_shifted']
        loc_weight     = vs.from_displs(loc_src_to_tgt.displs, np.ones(loc_src_to_tgt.dsize))
      elif src_loc=="Vertex": # Move results of mesh location from cell to vtx
        loc_src_to_tgt = res_loc['points_gnum_shifted@VTX']
        loc_weight = res_loc['points_weights@VTX']

        
      tgt_in_src_vs = vs.concatenate([loc_src_to_tgt, clo_tgt_in_src], vs.INNER_AXIS)
      tgt_weight_vs = vs.concatenate([loc_weight, clo_weight],         vs.INNER_AXIS)

      src_to_tgt.append({'target_gnum' :tgt_in_src_vs, 'target_weight':tgt_weight_vs})

  return src_to_tgt




def interpolate(src_tree, tgt_tree, comm, containers_name, location, **options):
  """Interpolate fields between two partitioned trees.

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
    src_tree (CGNSTree): Source tree, partitioned. Only 3D unstructured connectivities are managed.
    tgt_tree (CGNSTree): Target tree, partitioned. Structured or unstructured connectivities are managed.
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
  # Early return if containers_name is empty
  assert isinstance(containers_name, list)
  if len(containers_name) == 0:
    return

  # Guess location of input fields using first input zone
  first_part = next(PT.iter_all_Zone_t(src_tree))
  input_loc = PT.Subset.GridLocation(PT.get_child_from_name(first_part, containers_name[0]))

  # Create interpolator
  interpolator = create_interpolator(src_tree, tgt_tree, comm, input_loc, location, **options)

  # Exchange fields
  for container_name in containers_name:
    interpolator.exchange_fields(container_name)



def create_interpolator(src_tree, tgt_tree, comm, src_location, tgt_location, **options):
  """Same as interpolate, but return the interpolator object instead
  of doing interpolations. Interpolator can be called multiple time to exchange
  fields without recomputing the src_to_tgt indirection (geometry must remain the same).
  """
  src_dom = PT.get_children_from_predicates(src_tree, 'CGNSBase_t/Zone_t')
  tgt_dom = PT.get_children_from_predicates(tgt_tree, 'CGNSBase_t/Zone_t')

  src_to_tgt = create_src_to_tgt(src_dom, tgt_dom, comm, src_location, tgt_location, **options)
  return Interpolator(src_dom, tgt_dom, src_to_tgt, src_location, tgt_location, comm)
