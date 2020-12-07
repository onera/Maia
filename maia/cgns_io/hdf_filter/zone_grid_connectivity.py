import Converter.Internal as I
# from .index_array         import create_index_array_filter
from .data_array          import create_point_list_filter

def create_zone_grid_connectivity_filter(zone, zone_path, hdf_filter):
  """
  """
  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+zone_gc[0]
    for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity_t'):
      gc_path = zone_gc_path+"/"+gc[0]
      distrib_ud = I.getNodeFromName1(gc        , ':CGNS#Distribution')
      distrib_ia = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
      data_space = create_point_list_filter(distrib_ia)
      hdf_filter[gc_path + "/PointList"] = data_space

      pld_n = I.getNodeFromName1(gc, "PointListDonor")
      if(pld_n is not None):
        hdf_filter[gc_path + "/PointListDonor"] = data_space

