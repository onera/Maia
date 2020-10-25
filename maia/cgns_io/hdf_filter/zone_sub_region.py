import Converter.Internal as I

from .data_array import create_data_array_filter

def create_zone_subregion_filter(zone_subregion, zone_sub_region_path, hdf_filter):
  """
  """
  distrib_ud = I.getNodeFromName1(zone_subregion, ':CGNS#Distribution')
  distrib_da = I.getNodeFromName1(distrib_ud    , 'Distribution')[1]

  create_data_array_filter(zone_subregion, zone_sub_region_path, distrib_da, hdf_filter)

