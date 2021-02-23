import Converter.Internal as I
import maia.sids.sids as SIDS
import fnmatch

from .hdf_dataspace    import create_data_array_filter
from .cgns_elements    import create_zone_elements_filter
from .cgns_subsets     import create_zone_subregion_filter,\
                              create_zone_bc_filter,\
                              create_zone_grid_connectivity_filter
from .                 import utils

def create_zone_filter(zone, zone_path, hdf_filter, mode):
  """
  Fill up the hdf filter for the following elements of the zone:
  Coordinates, Elements (NGon / NFace, Standards), FlowSolution
  (vertex & cells only), ZoneSubRegion, ZoneBC (including BCDataSet)
  and ZoneGridConnectivity.

  The bounds of the filter are determined by the :CGNS#Distribution
  node and, for the structured zones, by the size of the blocks.
  """
  n_vtx  = SIDS.zone_n_vtx (zone)
  n_cell = SIDS.zone_n_cell(zone)

  distrib_ud   = I.getNodeFromName1(zone , ':CGNS#Distribution')
  distrib_vtx  = I.getNodeFromName1(distrib_ud, 'Vertex')[1]
  distrib_cell = I.getNodeFromName1(distrib_ud, 'Cell'  )[1]

  all_vtx_dataspace   = create_data_array_filter(distrib_vtx, zone[1][:,0])
  all_cells_dataspace = create_data_array_filter(distrib_cell, zone[1][:,1])

  # Coords
  for grid_c in I.getNodesFromType1(zone, 'GridCoordinates_t'):
    grid_coord_path = zone_path+"/"+grid_c[0]
    utils.apply_dataspace_to_arrays(grid_c, grid_coord_path, all_vtx_dataspace, hdf_filter)

  create_zone_elements_filter(zone, zone_path, hdf_filter, mode)

  create_zone_bc_filter(zone, zone_path, hdf_filter)
  create_zone_grid_connectivity_filter(zone, zone_path, hdf_filter)
  create_zone_subregion_filter(zone, zone_path, hdf_filter)

  for flow_solution in I.getNodesFromType1(zone, 'FlowSolution_t'):
    flow_solution_path = zone_path+"/"+flow_solution[0]
    grid_location = SIDS.GridLocation(flow_solution)
    if(grid_location == 'CellCenter'):
      data_space = all_cells_dataspace
    elif(grid_location == 'Vertex'):
      data_space = all_vtx_dataspace
    else:
      raise NotImplementedError(f"GridLocation {grid_location} not implemented")
    utils.apply_dataspace_to_arrays(flow_solution, flow_solution_path, data_space, hdf_filter)

def create_tree_hdf_filter(dist_tree, hdf_filter, mode='read'):
  """
  On a besoin du write pour gérer le ElementStartIndex
  It can be replace by a if None in tree to see if read/write ?
  """
  for base in I.getNodesFromType1(dist_tree, 'CGNSBase_t'):
    for zone in I.getNodesFromType1(base, 'Zone_t'):
      zone_path = "/"+I.getName(base)+"/"+I.getName(zone)
      create_zone_filter(zone, zone_path, hdf_filter, mode)


def filtering_filter(dist_tree, hdf_filter, name_or_type_list, skip=True):
  """
  """
  cur_hdf_filter = dict()

  # print("name_or_type_list::", name_or_type_list)
  n_ancestor = 0
  for skip_type in name_or_type_list:
    ancestors_type = skip_type # .split("/")
    n_ancestor     = len(ancestors_type)
    # print("ancestors_type::", ancestors_type)

  # > We filter the hdf_filter to keep only the unskip data
  for path, data in hdf_filter.items():
    split_path = path.split("/")
    # print(split_path)
    # > We should remove from dictionnary all entry
    first_path = ''
    first_n = 0
    for idx in range(len(split_path)-n_ancestor):
      first_path += "/"+split_path[idx]
      first_n    += 1
    prev_node = I.getNodeFromPath(dist_tree, first_path[1:]) # Remove //

    # Now we need to skip if matching with ancestors type
    ancestors_name = []
    ancestors_type = []
    for idx in range(first_n, len(split_path)):
      next_name = split_path[idx]
      next_node = I.getNodeFromName1(prev_node, next_name)
      ancestors_name.append(next_node[0])
      ancestors_type.append(next_node[3])
      prev_node = next_node

    # print("ancestors_type::", ancestors_type)
    # print("ancestors_name::", ancestors_name)

    keep_path = skip
    for skip_type in name_or_type_list:
      n_ancestor = len(skip_type)
      n_match_name_or_type = 0
      for idx in reversed(range(n_ancestor)):
        # print(f"idx::{idx} with skip_type={skip_type[idx]} compare to {ancestors_name[idx]} and {ancestors_type[idx]} ")
        # print("fnmatch::", fnmatch.fnmatch(ancestors_name[idx], skip_type[idx]))
        if( fnmatch.fnmatch(ancestors_name[idx], skip_type[idx])):
          n_match_name_or_type += 1
        elif( (skip_type[idx] == ancestors_name[idx] ) or
              (skip_type[idx] == ancestors_type[idx] )):
          n_match_name_or_type += 1
      if(n_match_name_or_type == len(skip_type)):
        keep_path = not skip
      # print("n_match_name_or_type::", n_match_name_or_type, "/", n_ancestor)
      # print("******************************", path, keep_path)

    if(keep_path):
      cur_hdf_filter[path] = data

  return cur_hdf_filter

