import Converter.Internal as I
import maia.sids.sids as SIDS

from maia.cgns_io.hdf_filter import elements        as HEF
from maia.cgns_io.hdf_filter import zone_sub_region as SRF

from .data_array import create_data_array_filter

def create_zone_filter(zone_tree, zone_path, hdf_filter):
  """
  """
  n_vtx  = SIDS.zone_n_vtx (zone_tree)
  n_cell = SIDS.zone_n_cell(zone_tree)


  distrib_ud   = I.getNodeFromName1(zone_tree , ':CGNS#Distribution')
  distrib_vtx  = I.getNodeFromName1(distrib_ud, 'Vertex')[1]
  distrib_cell = I.getNodeFromName1(distrib_ud, 'Cell'  )[1]

  zone_type = I.getNodeFromType1(zone_tree, "ZoneType_t")[1].tostring()
  if(zone_type == b'Structured'):
    raise RuntimeError("create_zone_filter not implemented for structured grid ")

  HEF.create_zone_elements_filter(zone_tree, zone_path, hdf_filter)

  for zone_subregion in I.getNodesFromType1(zone_tree, 'ZoneSubRegion_t'):
    zone_sub_region_path = zone_path+"/"+zone_subregion[0]
    create_zone_subregion_filter(zone_subregion, zone_sub_region_path, hdf_filter)

  for flow_solution in I.getNodesFromType1(zone_tree, 'FlowSolution_t'):
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
      raise NotImplemented(f"GridLocation {grid_location} not implemented")
