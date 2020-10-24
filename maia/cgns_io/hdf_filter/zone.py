import Converter.Internal as I

from maia.cgns_io.hdf_filter import elements        as HEF
from maia.cgns_io.hdf_filter import zone_sub_region as SRF

def create_zone_filter(zone_tree, zone_path, hdf_filter):
  """
  """
  HEF.create_zone_elements_filter(zone_tree, zone_path, hdf_filter)

  for zone_subregion in I.getNodesFromType1(zone_tree, 'ZoneSubRegion_t'):
    zone_sub_region_path = zone_path+"/"+zone_subregion[0]
    create_zone_subregion_filter(zone_subregion, zone_sub_region_path, hdf_filter)
