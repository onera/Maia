import Converter.Internal as I
# from .index_array         import create_index_array_filter
from .point_list          import create_point_list_filter

def create_zone_grid_connectivity_filter(zone, zone_path, hdf_filter):
  """
  """
  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+zone_gc[0]
    for gc in I.getNodesFromType1(zone_gc, 'GridConnectivity_t'):
      gc_path = zone_gc_path+"/"+gc[0]
      distrib_ud = I.getNodeFromName1(gc        , ':CGNS#Distribution')
      distrib_ia = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
      create_point_list_filter(gc, gc_path, "PointList", distrib_ia, hdf_filter)

      pld_n = I.getNodeFromName1(gc, "PointListDonor")
      if(pld_n is not None):
        create_point_list_filter(gc, gc_path, "PointListDonor", distrib_ia, hdf_filter)

