import Converter.Internal as I
import maia.sids.sids as SIDS

from maia.cgns_io.hdf_filter import elements        as HEF

from .hdf_dataspace          import create_data_array_filter
from .zone_sub_region        import create_zone_subregion_filter
from .zone_bc                import create_zone_bc_filter
from .zone_grid_connectivity import create_zone_grid_connectivity_filter
from .                       import utils

def create_zone_filter(zone, zone_path, hdf_filter, mode):
  """
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

  HEF.create_zone_elements_filter(zone, zone_path, hdf_filter, mode)

  create_zone_bc_filter(zone, zone_path, hdf_filter)
  create_zone_grid_connectivity_filter(zone, zone_path, hdf_filter)
  create_zone_subregion_filter(zone, zone_path, hdf_filter)

  for flow_solution in I.getNodesFromType1(zone, 'FlowSolution_t'):
    flow_solution_path = zone_path+"/"+flow_solution[0]
    grid_location_n = I.getNodeFromType1(flow_solution, 'GridLocation_t')
    if(grid_location_n is None):
      raise RuntimeError("You need specify GridLocation in FlowSolution to load the cgns ")
    grid_location = grid_location_n[1].tostring()
    if(grid_location == b'CellCenter'):
      data_space = all_cells_dataspace
    elif(grid_location == b'Vertex'):
      data_space = all_vtx_dataspace
    else:
      raise NotImplementedError(f"GridLocation {grid_location} not implemented")
    utils.apply_dataspace_to_arrays(flow_solution, flow_solution_path, data_space, hdf_filter)
