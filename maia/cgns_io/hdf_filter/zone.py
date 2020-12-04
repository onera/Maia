import Converter.Internal as I
import maia.sids.sids as SIDS

from maia.cgns_io.hdf_filter import elements        as HEF
from maia.cgns_io.hdf_filter import zone_sub_region as SRF

from .data_array             import create_data_array_filter
from .zone_sub_region        import create_zone_subregion_filter
from .zone_bc                import create_zone_bc_filter
from .zone_grid_connectivity import create_zone_grid_connectivity_filter


def create_grid_coord_filter(zone, zone_path, hdf_filter):
  """
  """
  data_shape = zone[1][:,0] #Usefull to distinguish U and S
  distrib_ud   = I.getNodeFromName1(zone , ':CGNS#Distribution')
  distrib_vtx  = I.getNodeFromName1(distrib_ud, 'Vertex')[1]

  for grid_c in I.getNodesFromType1(zone, 'GridCoordinates_t'):
    grid_coord_path = zone_path+"/"+grid_c[0]
    create_data_array_filter(grid_c, grid_coord_path, distrib_vtx, hdf_filter, data_shape)

def create_zone_filter(zone, zone_path, hdf_filter, mode):
  """
  """
  n_vtx  = SIDS.zone_n_vtx (zone)
  n_cell = SIDS.zone_n_cell(zone)

  distrib_ud   = I.getNodeFromName1(zone , ':CGNS#Distribution')
  distrib_vtx  = I.getNodeFromName1(distrib_ud, 'Vertex')[1]
  distrib_cell = I.getNodeFromName1(distrib_ud, 'Cell'  )[1]

  create_zone_bc_filter(zone, zone_path, hdf_filter)
  create_zone_grid_connectivity_filter(zone, zone_path, hdf_filter)
  create_grid_coord_filter(zone, zone_path, hdf_filter)

  HEF.create_zone_elements_filter(zone, zone_path, hdf_filter, mode)

  for zone_subregion in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
    zone_sub_region_path = zone_path+"/"+zone_subregion[0]
    create_zone_subregion_filter(zone_subregion, zone_sub_region_path, hdf_filter)

  for flow_solution in I.getNodesFromType1(zone, 'FlowSolution_t'):
    flow_solution_path = zone_path+"/"+flow_solution[0]
    grid_location_n = I.getNodeFromType1(flow_solution, 'GridLocation_t')
    if(grid_location_n is None):
      raise RuntimeError("You need specify GridLocation in FlowSolution to load the cgns ")
    grid_location = grid_location_n[1].tostring()
    if(grid_location == b'CellCenter'):
      create_data_array_filter(flow_solution, flow_solution_path, distrib_cell, hdf_filter)
    elif(grid_location == b'Vertex'):
      create_data_array_filter(flow_solution, flow_solution_path, distrib_vtx, hdf_filter)
    else:
      raise NotImplementedError(f"GridLocation {grid_location} not implemented")
