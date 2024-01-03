import fnmatch

import maia.pytree        as PT
import maia.pytree.maia   as MT

from .hdf_dataspace    import create_data_array_filter
from .cgns_elements    import create_zone_elements_filter
from .cgns_subsets     import create_zone_subregion_filter,\
                              create_flow_solution_filter,\
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
  # Coords
  distrib_vtx  = PT.get_value(MT.getDistribution(zone, 'Vertex'))
  all_vtx_dataspace   = create_data_array_filter(distrib_vtx, zone[1][:,0])
  for grid_c in PT.iter_children_from_label(zone, 'GridCoordinates_t'):
    grid_coord_path = zone_path + "/" + PT.get_name(grid_c)
    utils.apply_dataspace_to_arrays(grid_c, grid_coord_path, all_vtx_dataspace, hdf_filter)

  create_zone_elements_filter(zone, zone_path, hdf_filter, mode)

  create_zone_bc_filter(zone, zone_path, hdf_filter)
  create_zone_grid_connectivity_filter(zone, zone_path, hdf_filter)
  create_flow_solution_filter(zone, zone_path, hdf_filter)
  create_zone_subregion_filter(zone, zone_path, hdf_filter)


def create_tree_hdf_filter(dist_tree, mode='read'):
  """
  On a besoin du write pour gérer le ElementStartIndex
  It can be replace by a if None in tree to see if read/write ?
  """
  hdf_filter = dict()
  for base, zone in PT.iter_nodes_from_predicates(dist_tree, 'CGNSBase_t/Zone_t', ancestors=True):
    zone_path = PT.get_name(base)+"/"+PT.get_name(zone)
    create_zone_filter(zone, zone_path, hdf_filter, mode)
  return hdf_filter
