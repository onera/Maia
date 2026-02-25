import mpi4py.MPI as MPI
import numpy      as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils    import par_utils
from maia.transfer import utils     as te_utils,\
                          protocols as EP

from maia.factory.dist_from_part import discover_nodes_from_matching
from .                           import index_exchange as IPTB

IS_EMPTY_DA = PT.pred.label_is('DataArray_t') & PT.pred.value_is(None)

def _discover_wrapper(dist_zone, part_zones, pl_path, data_path, comm):
  """
  Wrapper for discover_nodes_from_matching which add the node path in distree,
  but also recreate the distributed pointlist if needed
  """
  if pl_path.split('/')[0] == 'ZoneSubRegion_t':
    is_gc_zsr = PT.pred.label_is('ZoneSubRegion_t') & PT.pred.has_child_of_name('GridConnectivityRegionName')
    ini_zsr_nodes_names = [PT.get_name(n) for n in PT.get_nodes_from_predicate(dist_zone, is_gc_zsr)]

  discover_nodes_from_matching(dist_zone, part_zones, pl_path,   comm, child_list=['GridLocation_t', 'Descriptor_t'])
  discover_nodes_from_matching(dist_zone, part_zones, data_path, comm)
  for nodes in PT.iter_children_from_predicates(dist_zone, pl_path, ancestors=True):
    node_path   = '/'.join([PT.get_name(node) for node in nodes])
    if PT.get_node_from_path(nodes[-1], 'PointRange') is None and \
       par_utils.exists_anywhere(part_zones, node_path+'/PointRange', comm):
       # PointRange must be computed on dist node
       if not par_utils.exists_anywhere(part_zones, node_path+'/:CGNS#GlobalNumbering/Index', comm):
         # > GlobalNumbering is required to do that
         IPTB.create_part_pr_gnum(dist_zone, part_zones, node_path, comm)
       IPTB.part_pr_to_dist_pr(dist_zone, part_zones, node_path, comm)
    if PT.get_node_from_path(nodes[-1], 'PointList') is None and \
       par_utils.exists_anywhere(part_zones, node_path+'/PointList', comm):
      # > Pointlist must be computed on dist node
      if not par_utils.exists_anywhere(part_zones, node_path+'/:CGNS#GlobalNumbering/Index', comm):
        # > GlobalNumbering is required to do that
        IPTB.create_part_pl_gnum(dist_zone, part_zones, node_path, comm)
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, node_path, comm)
  
  if pl_path.split('/')[0] == 'ZoneSubRegion_t':
    # If we have a splitted ZSR (because of GridConnectivityRegionName), we need to merge it
    # Here we assume that GC related ZSR can only refer to original jns, named using maia conventions
    # First remove split node already present in dist_zone
    for name in ini_zsr_nodes_names:
      is_gc_zsr_split = is_gc_zsr & ~PT.pred.name_is(name) & PT.pred.name_matches(name+'*')
      PT.rm_children_from_predicate(dist_zone, is_gc_zsr_split)
    # Now manage split node not present in dist zone (remove all but .0 and rename .0)
    is_gc_zsr_split = is_gc_zsr & ~PT.pred.name_in(ini_zsr_nodes_names)
    PT.rm_children_from_predicate(dist_zone, lambda n: is_gc_zsr_split(n) \
            and int(MT.conv.get_split_suffix(PT.get_name(n))) > 0)
    for zsr in PT.get_children_from_predicate(dist_zone, is_gc_zsr_split):
      PT.update_node(zsr, name=MT.conv.get_split_prefix(PT.get_name(zsr)))
      descri = PT.get_child_from_name(zsr, 'GridConnectivityRegionName')
      PT.update_node(descri, value=MT.conv.get_split_prefix(PT.get_value(descri)))
          
def part_coords_to_dist_coords(dist_zone, part_zones, comm, reduce_op=None):

  distribution = MT.Zone.vtx_distribution(dist_zone)
  lntogn_list = [MT.Zone.vtx_globalnumbering(pzone) for pzone in part_zones]

  d_grid_co = PT.get_child_from_label(dist_zone, "GridCoordinates_t")
  part_data = dict()
  for d_co_name in PT.Zone.coordinates(dist_zone)._fields:
    if PT.get_child_from_name(d_grid_co, d_co_name) is not None:
      part_data[d_co_name] = list()
  
  for part_zone in part_zones:
    for p_co_name, coord in PT.Zone.coordinates(part_zone)._asdict().items():
      if coord is not None:
        flat_data = coord.ravel(order='A') #Reshape structured arrays for PDM exchange
        part_data[p_co_name].append(flat_data)
      else:
        part_data.pop(p_co_name, None) # Remove key from dict

  # Exchange
  dist_data = EP.part_to_block(part_data, distribution, lntogn_list, comm, reduce_op, gnum_offset=1)
  for coord, array in dist_data.items():
    dist_coord = PT.get_child_from_name(d_grid_co, coord)
    PT.set_value(dist_coord, array)

def _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm, reduce_op=None):
  """
  Shared code for FlowSolution_t and DiscreteData_t
  """
  for mask_sol in PT.get_children(mask_tree):
    sol_name = PT.get_name(mask_sol)
    d_sol = PT.get_child_from_name(dist_zone, sol_name) #True container

    # Partial FlowSol may not exist on all zones
    _part_zones = [zone for zone in part_zones if PT.get_child_from_name(zone, sol_name) is not None]
    _part_sols  = [PT.find_child_from_name(zone, sol_name) for zone in _part_zones]

    if comm.allreduce(len(_part_zones)) == 0:
      continue #Skip FS that remains on dist_tree but are not present on part tree

    distribution = MT.Container.distribution(d_sol, dist_zone)
    lntogn_list = [MT.Container.globalnumbering(p_sol, part_zone) for p_sol, part_zone in zip(_part_sols, _part_zones)]

    #Discover data
    fields = [PT.get_name(n) for n in PT.get_children(mask_sol) if \
              par_utils.exists_everywhere(_part_zones, f'{sol_name}/{PT.get_name(n)}', comm)]

    part_data = {field : [] for field in fields}

    for part_sol in _part_sols:
      for field in fields:
        flat_data = PT.get_child_from_name(part_sol, field)[1].ravel(order='A') #Reshape structured arrays for PDM exchange
        part_data[field].append(flat_data)

    # Exchange
    dist_data = EP.part_to_block(part_data, distribution, lntogn_list, comm, reduce_op, gnum_offset=1)
    for field, array in dist_data.items():
      dist_field = PT.get_child_from_name(d_sol, field)
      PT.set_value(dist_field, array)

def part_sol_to_dist_sol(dist_zone, part_zones, comm, include=[], exclude=[], reduce_op=None):
  """
  Transfert all the data included in FlowSolution_t nodes from partitioned
  zones to the distributed zone. Data created on (one or more) partitions and not present in dist_tree
  is also reported to the distributed zone.
  """
  # Complete distree with partitioned fields and exchange PL if needed
  _discover_wrapper(dist_zone, part_zones, 'FlowSolution_t', 'FlowSolution_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['FlowSolution_t', 'DataArray_t'], include, exclude)
  _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm, reduce_op)
  # Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_sol in PT.iter_children_from_label(dist_zone, 'FlowSolution_t'):
    PT.rm_children_from_predicate(dist_sol, IS_EMPTY_DA)
  # Update ZoneIterativeData/FlowSolutionPointers
  discover_nodes_from_matching(dist_zone, part_zones, "ZoneIterativeData_t", comm,
                               child_list=['FlowSolutionPointers'])

def part_discdata_to_dist_discdata(dist_zone, part_zones, comm, include=[], exclude=[], reduce_op=None):
  """
  Transfert all the data included in DiscreteData_t from partitioned
  zones to the distributed zone. Data created on (one or more) partitions and not present in dist_tree
  is also reported to the distributed zone.
  """
  # Complete distree with partitioned fields and exchange PL if needed
  _discover_wrapper(dist_zone, part_zones, 'DiscreteData_t', 'DiscreteData_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['DiscreteData_t', 'DataArray_t'], include, exclude)
  _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm, reduce_op)
  #Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_sol in PT.iter_children_from_label(dist_zone, 'DiscreteData_t'):
    PT.rm_children_from_predicate(dist_sol, IS_EMPTY_DA)

def part_gridmotion_to_dist_gridmotion(dist_zone, part_zones, comm, include=[], exclude=[], reduce_op=None):
  """
  Transfert all the data included in ArbitraryGridMotion_t from partitioned
  zones to the distributed zone. Data created on (one or more) partitions and not present in dist_tree
  is also reported to the distributed zone.
  """
  # Complete distree with partitioned fields and exchange PL if needed
  _discover_wrapper(dist_zone, part_zones, 'ArbitraryGridMotion_t', 'ArbitraryGridMotion_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['ArbitraryGridMotion_t', 'DataArray_t'], include, exclude)
  _part_to_dist_sollike(dist_zone, part_zones, mask_tree, comm, reduce_op)
  #Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_sol in PT.iter_children_from_label(dist_zone, 'ArbitraryGridMotion_t'):
    PT.rm_children_from_predicate(dist_sol, IS_EMPTY_DA)

def part_subregion_to_dist_subregion(dist_zone, part_zones, comm, include=[], exclude=[], reduce_op=None):
  """
  Transfert all the data included in ZoneSubRegion_t nodes from the partitioned
  zones to the distributed zone.
  """
  is_zsr_with_pl = PT.pred.label_is('ZoneSubRegion_t') & PT.pred.has_child_of_name('PointList')
  for zone in part_zones:
    for zsr_n in PT.get_children_from_predicate(zone, is_zsr_with_pl):
      gn_n = MT.get_GlobalNumbering(zsr_n)
      assert gn_n is not None and PT.get_child_from_name(gn_n, 'Index') is not None,\
      f"missing \":CGNS#GlobalNumbering\" node under ZoneSubRegion_t with PointList \"{PT.get_name(zsr_n)}\""
  _discover_wrapper(dist_zone, part_zones, 'ZoneSubRegion_t', 'ZoneSubRegion_t/DataArray_t', comm)
  mask_tree = te_utils.create_mask_tree(dist_zone, ['ZoneSubRegion_t', 'DataArray_t'], include, exclude)
  for mask_zsr in PT.get_children(mask_tree):
    d_zsr = PT.get_child_from_name(dist_zone, PT.get_name(mask_zsr)) #True ZSR
    is_gc_related = PT.get_node_from_name(d_zsr, 'GridConnectivityRegionName') is not None
    tgt_name = PT.get_name(d_zsr) + '.*' if is_gc_related else PT.get_name(d_zsr) 

    #Get distribution
    distribution  = MT.Container.distribution(d_zsr, dist_zone)

    #Get lngn and data
    fields = [PT.get_name(n) for n in PT.get_children(mask_zsr)]
    part_data = {field : [] for field in fields}
    lngn_list = list()
    for part_zone in part_zones:
      for p_zsr in PT.iter_children_from_predicate(part_zone, tgt_name):
        lngn_list.append(MT.Container.globalnumbering(p_zsr, part_zone))
        for field in fields:
          part_data[field].append(PT.get_child_from_name(p_zsr, field)[1])
    
    # Exchange
    dist_data = EP.part_to_block(part_data, distribution, lngn_list, comm, reduce_op, gnum_offset=1)
    for field, array in dist_data.items():
      dist_field = PT.get_child_from_name(d_zsr, field)
      PT.set_value(dist_field, array)

  #Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_zsr in PT.iter_children_from_label(dist_zone, 'ZoneSubRegion_t'):
    PT.rm_children_from_predicate(dist_zsr, IS_EMPTY_DA)

def part_dataset_to_dist_dataset(dist_zone, part_zones, comm, include=[], exclude=[], reduce_op=None):
  """
  Transfert all the data included in BCDataSet_t/BCData_t nodes from partitioned
  zones to the distributed zone.
  """

  # Complete distree with partitioned fields and exchange PL if needed
  bc_ds_path = 'ZoneBC_t/BC_t/BCDataSet_t'
  _discover_wrapper(dist_zone, part_zones, bc_ds_path, bc_ds_path+'/BCData_t/DataArray_t', comm)

  for d_zbc in PT.iter_children_from_label(dist_zone, "ZoneBC_t"):
    labels = ['BC_t', 'BCDataSet_t', 'BCData_t', 'DataArray_t']
    mask_tree = te_utils.create_mask_tree(d_zbc, labels, include, exclude)
    for mask_bc in PT.get_children(mask_tree):
      bc_path   = PT.get_name(d_zbc) + '/' + PT.get_name(mask_bc)
      d_bc = PT.get_node_from_path(dist_zone, bc_path) #True BC
      for mask_dataset in PT.get_children(mask_bc):
        ds_path = bc_path + '/' + PT.get_name(mask_dataset)
        d_dataset = PT.find_node_from_path(dist_zone, ds_path) #True DataSet
        is_subset = PT.Container._is_subset(d_dataset)

        distri_node = MT.get_Distribution(PT.Container.SubsetNode(d_dataset, d_bc))
        distribution = PT.get_child_from_name(distri_node, 'Index')[1]

        path = ds_path if is_subset else bc_path
        _part_zones = [zone for zone in part_zones if PT.get_node_from_path(zone, path) is not None]
        p_subsets   = [PT.get_node_from_path(zone, path) for zone in _part_zones]
        lngn_list   = [MT.Subset.globalnumbering(p_subset) for p_subset in p_subsets]

        #Discover data
        data_paths = PT.predicates_to_paths(mask_dataset, ['*', '*'])
        part_data = {path : [] for path in data_paths}

        for part_zone in _part_zones:
          p_dataset = PT.find_node_from_path(part_zone, ds_path)
          for path in data_paths:
            part_data[path].append(PT.get_node_from_path(p_dataset, path)[1])

        # Filter global (size == 1) data
        size_1_loc = {path : any([t.size == 1 and gn.size != 1 for t,gn in zip(data, lngn_list)]) for path,data in part_data.items()}
        loc_values = np.array([v for v in size_1_loc.values()])
        glo_values = np.empty_like(loc_values)
        comm.Allreduce(loc_values, glo_values, MPI.LOR)
        part_data_loc = {path: data for i,(path,data) in enumerate(part_data.items()) if not glo_values[i]}
        part_data_glo = {path: data for i,(path,data) in enumerate(part_data.items()) if     glo_values[i]}

        #Exchange local data
        dist_data = EP.part_to_block(part_data_loc, distribution, lngn_list, comm, reduce_op, gnum_offset=1)
        for field, array in dist_data.items():
          dist_field = PT.get_node_from_path(d_dataset, field)
          PT.set_value(dist_field, array)
        
        # Exchange global data (take first partition knowing a value)
        master = comm.allreduce(comm.Get_rank() if len(lngn_list) > 0 else comm.Get_size()+1, op=MPI.MIN)
        glob_data_send = None
        if master > comm.Get_size(): # no data any where to exchange -> nothing to do
          continue
        if comm.Get_rank() == master:
          glob_data_send = {path: data[0] for path,data in part_data_glo.items()}
        glob_data_dist = comm.bcast(glob_data_send, root=master)
        for field, array in glob_data_dist.items():
          dist_field = PT.get_node_from_path(d_dataset, field)
          PT.set_value(dist_field, array)
        # Update BCDataGlobal node  
        global_arrays_node = PT.get_child_from_name(distri_node, 'BCDataGlobal')
        old_global_arrays = PT.get_value(global_arrays_node).split('\n') if global_arrays_node is not None else []
        if is_subset:
          prefix = ''
          new_global_arrays = [] # Init to empty 
        else:
          prefix = f"{PT.get_name(d_dataset)}/"
          new_global_arrays = [path for path in old_global_arrays if not path.startswith(prefix)] # Init with other dataset
          old_global_arrays = [PT.utils.path_tail(path, 1) for path in old_global_arrays if path.startswith(prefix)] # Filter prefix 

        new_global_arrays += [f"{prefix}{path}" for path in old_global_arrays if path not in dist_data]         # Old arrays, if not transformed into local 
        new_global_arrays += [f"{prefix}{path}" for path in glob_data_dist    if path not in old_global_arrays] # New arrays, if not already existing 
        if len(new_global_arrays) > 0:
          PT.update_child(distri_node, 'BCDataGlobal', 'Descriptor_t', '\n'.join(new_global_arrays))
        else:
          PT.rm_children_from_name(distri_node, 'BCDataGlobal')

  #Cleanup : if field is None, data has been added by wrapper and must be removed
  for dist_ddata in PT.iter_nodes_from_predicates(dist_zone, bc_ds_path+'/BCData_t'):
    PT.rm_children_from_predicate(dist_ddata, IS_EMPTY_DA)

