from maia.typing import *
import maia.pytree      as PT
import maia.pytree.maia as MT
from maia.utils    import par_utils
from maia.transfer import utils     as te_utils,\
                          protocols as EP

from maia.transfer.part_to_dist import index_exchange as IPTB

import Pypdm.Pypdm as PDM

def dist_coords_to_part_coords(dist_zone: CGNSDistTree, 
                               part_zones: List[CGNSPartTree],
                               comm: MPIComm) -> None:
  """
  Transfert all the data included in GridCoordinates_t nodes from a distributed
  zone to the partitioned zones
  """
  #Get distribution
  distribution_vtx = MT.Zone.vtx_distribution(dist_zone)
  vtx_lntogn_list = [MT.Zone.vtx_globalnumbering(part_zone) for part_zone in part_zones]

  #Get data
  dist_data = dict()
  dist_gc = PT.find_child_from_label(dist_zone, "GridCoordinates_t")
  for grid_co in PT.iter_children_from_predicate(dist_gc, PT.pred.label_is('DataArray_t') & ~PT.pred.name_is('CoordinateTransform')):
    dist_data[PT.get_name(grid_co)] = PT.get_np_value(grid_co)

  part_data = EP.block_to_part(dist_data, distribution_vtx, vtx_lntogn_list, comm, gnum_offset=1)

  for ipart, part_zone in enumerate(part_zones):
    part_gc = PT.new_node('GridCoordinates', 'GridCoordinates_t', parent=part_zone)
    for data_name, data in part_data.items():
      #F is mandatory to keep shared reference. Normally no copy is done
      shaped_data = data[ipart].reshape(PT.Zone.VertexSize(part_zone), order='F')
      PT.new_DataArray(data_name, shaped_data, parent=part_gc)
    PT.add_child(part_gc, PT.get_child_from_name(dist_gc, 'CoordinateTransform'))

def dist_coords_to_part_coords_m(dist_zones: List[CGNSDistTree], 
                                 part_zones_per_dom: List[List[CGNSPartTree]],
                                 comm: MPIComm) -> None:
  """
  Same as dist_coords_to_part_coords, but with the multiblock version (only one collective call)
  """
  block_distris = []
  part_lngn = []
  vtx_offset = 0

  dist_data:Dict[str, List[NDArray]] = dict()
  for dist_zone, part_zones in zip(dist_zones, part_zones_per_dom):
    dist_gc_transform = PT.get_node_from_predicates(dist_zone, "GridCoordinates_t/CoordinateTransform")    

    for dist_gc_name, dist_gc_node in PT.Zone.coordinates(dist_zone)._asdict().items():
      if dist_gc_node is not None:
        try:
          dist_data[dist_gc_name].append(dist_gc_node)
        except KeyError:
          dist_data[dist_gc_name] = [dist_gc_node]

    vtx_distrib = MT.Zone.vtx_distribution(dist_zone)
    block_distris.append(par_utils.partial_to_full_distribution(vtx_distrib, comm))

    # Collect and shift LNToGN
    for part_zone in part_zones:
      part_lngn.append(MT.Zone.vtx_globalnumbering(part_zone) + vtx_offset)
      part_gc = PT.new_GridCoordinates('GridCoordinates', parent=part_zone)
      for dist_gc_name in dist_data.keys():
        PT.new_DataArray(dist_gc_name, None, parent=part_gc) #type:ignore[arg-type] #(will be replaced)
      PT.add_child(part_gc, dist_gc_transform)

    vtx_offset += PT.Zone.n_vtx(dist_zone)

  part_data = dict()
  MBTP = PDM.MultiBlockToPart(block_distris, part_lngn, comm)
  for key, d_data in dist_data.items():
    part_data[key] = MBTP.exchange_field(d_data)[1]
    
  i_part = 0
  for part_zones in part_zones_per_dom: 
    for part_zone in part_zones:
      part_gc = PT.get_child_from_label(part_zone, "GridCoordinates_t")
      for data_name, data in part_data.items():
        part_gc_node = PT.find_child_from_name(part_gc, data_name)
        shaped_data = data[i_part].reshape(PT.Zone.VertexSize(part_zone), order='F')
        PT.update_node(part_gc_node, value=shaped_data)
      i_part += 1


def _dist_to_part_sollike(dist_zone: CGNSDistTree, 
                          part_zones: List[CGNSPartTree], 
                          mask_tree: CGNSTree,
                          comm: MPIComm) -> None:
  """
  Shared code for FlowSolution_t and DiscreteData_t
  """
  #Get distribution
  for mask_sol in PT.get_children(mask_tree):
    d_sol = PT.find_child_from_name(dist_zone, PT.get_name(mask_sol)) #True container
    location = PT.Container.GridLocation(d_sol)
    ctn_name = PT.get_name(d_sol)

    if PT.Container._is_subset(d_sol): # Filter zones where p_sol does not exists
      _part_zones = [zone for zone in part_zones if PT.get_child_from_name(zone, ctn_name) is not None]
      psols = [PT.find_child_from_name(part_zone, ctn_name) for part_zone in _part_zones]
    else: # Filter not needed, but container should be created on part_zones (if needed)
      _part_zones = part_zones
      psols = [PT.update_child(part_zone, ctn_name, PT.get_label(d_sol), PT.get_value(d_sol)) for part_zone in _part_zones]
      [PT.update_child(p_sol, 'GridLocation', 'GridLocation_t', location) for p_sol in psols]

    # Get distribution & gnum
    distribution = MT.Container.distribution(d_sol, dist_zone)
    lntogn_list  = [MT.Container.globalnumbering(psol, part_zone) for psol,part_zone in zip(psols, _part_zones)]

    #Get data
    fields = [PT.get_name(n) for n in PT.get_children(mask_sol)]
    dist_data = {field : PT.get_np_value(PT.find_child_from_name(d_sol, field)) for field in fields}

    #Exchange
    part_data = EP.block_to_part(dist_data, distribution, lntogn_list, comm, gnum_offset=1)

    for ipart, (p_sol, part_zone) in enumerate(zip(psols,_part_zones)):
      if PT.Container._is_subset(p_sol):
        shape = PT.Subset.SizePerIndex(p_sol)
      else:
        shape = PT.Zone.VertexSize(part_zone) if location == 'Vertex' else PT.Zone.CellSize(part_zone)
      for data_name, data in part_data.items():
        #F is mandatory to keep shared reference. Normally no copy is done
        shaped_data = data[ipart].reshape(shape, order='F')
        PT.new_DataArray(data_name, shaped_data, parent=p_sol)

def dist_sol_to_part_sol(dist_zone: CGNSDistTree, 
                         part_zones: List[CGNSPartTree], 
                         comm: MPIComm, 
                         include: List[str] = [],
                         exclude: List[str] = []) -> None:
  """
  Transfert all the data included in FlowSolution_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['FlowSolution_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def dist_discdata_to_part_discdata(dist_zone: CGNSDistTree,
                                   part_zones: List[CGNSPartTree],
                                   comm: MPIComm, 
                                   include: List[str] = [],
                                   exclude: List[str] = []) -> None:
  """
  Transfert all the data included in DiscreteData_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['DiscreteData_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def dist_gridmotion_to_part_gridmotion(dist_zone: CGNSDistTree,
                                       part_zones: List[CGNSPartTree],
                                       comm: MPIComm, 
                                       include: List[str] = [], 
                                       exclude: List[str] = []) -> None:
  """
  Transfert all the data included in ArbitraryGridMotion_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['ArbitraryGridMotion_t', 'DataArray_t'], include, exclude)
  _dist_to_part_sollike(dist_zone, part_zones, mask_tree, comm)

def dist_dataset_to_part_dataset(dist_zone: CGNSDistTree, 
                                 part_zones: List[CGNSPartTree], 
                                 comm: MPIComm,
                                 include: List[str] = [], 
                                 exclude: List[str] = [])-> None:
  """
  Transfert all the data included in BCDataSet_t/BCData_t nodes from a distributed
  zone to the partitioned zones
  """
  for d_zbc in PT.iter_children_from_label(dist_zone, "ZoneBC_t"):
    labels = ['BC_t', 'BCDataSet_t', 'BCData_t', 'DataArray_t']
    mask_tree = te_utils.create_mask_tree(d_zbc, labels, include, exclude)
    for mask_bc in PT.get_children(mask_tree):
      bc_path = PT.get_name(d_zbc) + '/' + PT.get_name(mask_bc)
      d_bc = PT.find_node_from_path(dist_zone, bc_path) #True BC
      for mask_dataset in PT.get_children(mask_bc):
        ds_path = bc_path + '/' + PT.get_name(mask_dataset)
        d_dataset = PT.find_node_from_path(dist_zone, ds_path) #True DataSet
        is_subset = PT.Container._is_subset(d_dataset)

        if not is_subset:
          if not par_utils.exists_anywhere(part_zones, bc_path+'/:CGNS#GlobalNumbering/Index', comm):
            # For structured zones, gnum are not created during partitioning so add it now
            assert PT.Zone.Type(dist_zone) == "Structured"
            IPTB.create_part_pr_gnum(dist_zone, part_zones, bc_path, comm)

        distri_node = MT.get_Distribution(PT.Container.SubsetNode(d_dataset, d_bc))

        #Get data
        data_paths = PT.predicates_to_paths(mask_dataset, ['*', '*'])
        dist_data = {data_path : PT.get_np_value(PT.find_node_from_path(d_dataset, data_path)) \
                     for data_path in data_paths}
        # Filter global / local data
        global_arrays_node = PT.get_child_from_name(distri_node, 'BCDataGlobal')
        global_arrays_list = PT.get_str_value(global_arrays_node).split('\n') if global_arrays_node is not None else []
        as_path = lambda path : path if is_subset else f"{PT.get_name(d_dataset)}/{path}" #Add DS name if required, to search in global_arrays_list  
        dist_data_loc  = {path: data for path, data in dist_data.items() if as_path(path) not in global_arrays_list}
        dist_data_glob = {path: data for path, data in dist_data.items() if as_path(path)     in global_arrays_list}

        # Get gnum from DS or BC depending on has_own_distri
        path = ds_path if is_subset else bc_path
        _part_zones = [zone for zone in part_zones if PT.get_node_from_path(zone, path) is not None]
        p_subsets   = [PT.get_node_from_path(zone, path) for zone in _part_zones]
        lngn_list   = [MT.Subset.globalnumbering(p_subset) for p_subset in p_subsets]

        #Exchange (local data)
        distribution = PT.get_np_value(PT.find_child_from_name(distri_node, 'Index'))
        part_data = EP.block_to_part(dist_data_loc, distribution, lngn_list, comm, gnum_offset=1)

        #Put part data in tree
        for ipart, part_zone in enumerate(_part_zones):
          part_bc = PT.get_node_from_path(part_zone, bc_path)
          # Create dataset if no existing
          assert part_bc is not None
          part_ds = PT.update_child(part_bc, PT.get_name(d_dataset), PT.get_label(d_dataset), PT.get_value(d_dataset))
          # Add data
          for data_name, p_data in part_data.items():
            container_name, field_name = data_name.split('/')
            p_container = PT.update_child(part_ds, container_name, 'BCData_t')
            PT.new_DataArray(field_name, p_data[ipart], parent=p_container)
          for data_name, d_data in dist_data_glob.items(): # Copy global data
            container_name, field_name = data_name.split('/')
            p_container = PT.update_child(part_ds, container_name, 'BCData_t')
            PT.new_DataArray(field_name, d_data.copy(), parent=p_container)

def dist_subregion_to_part_subregion(dist_zone: CGNSDistTree,
                                     part_zones: List[CGNSPartTree],
                                     comm: MPIComm,
                                     include: List[str] = [], 
                                     exclude: List[str] = []) -> None:
  """
  Transfert all the data included in ZoneSubRegion_t nodes from a distributed
  zone to the partitioned zones
  """
  mask_tree = te_utils.create_mask_tree(dist_zone, ['ZoneSubRegion_t', 'DataArray_t'], include, exclude)
  for mask_zsr in PT.get_children(mask_tree):
    d_zsr = PT.find_child_from_name(dist_zone, PT.get_name(mask_zsr)) #True ZSR
    is_gc_related = PT.get_node_from_name(d_zsr, 'GridConnectivityRegionName') is not None

    if not is_gc_related:
      matching_region_path = PT.Container.SubsetNodePath(d_zsr, dist_zone)
      if not par_utils.exists_anywhere(part_zones, matching_region_path+'/:CGNS#GlobalNumbering/Index', comm):
        # For structured zones, gnum are not created during partitioning so add it now
        assert PT.Zone.Type(dist_zone) == "Structured"
        IPTB.create_part_pr_gnum(dist_zone, part_zones, matching_region_path, comm)
    
    #Get distribution and dist data
    distribution = MT.Container.distribution(d_zsr, dist_zone)
    fields = [PT.get_name(n) for n in PT.get_children(mask_zsr)]
    dist_data = {field : PT.get_np_value(PT.find_child_from_name(d_zsr, field)) \
                  for field in fields}

    # GC related ZSR are splitted in several nodes (as GC)
    tgt_name = PT.get_name(d_zsr) + '.*' if is_gc_related else PT.get_name(d_zsr) 
    lngn_list = list()
    for part_zone in part_zones:
      for p_zsr in PT.iter_children_from_predicate(part_zone, tgt_name):
        lngn_list.append(MT.Container.globalnumbering(p_zsr, part_zone))

    #Exchange
    part_data = EP.block_to_part(dist_data, distribution, lngn_list, comm, gnum_offset=1)

    #Put part data in tree
    i_pseudo_part = 0
    # Use same loop order than data lngn collecting
    for part_zone in part_zones:
      for p_zsr in PT.iter_children_from_predicate(part_zone, tgt_name):
        for field_name, data in part_data.items():
          PT.new_DataArray(field_name, data[i_pseudo_part], parent=p_zsr)
        i_pseudo_part += 1

    assert i_pseudo_part == len(lngn_list)
