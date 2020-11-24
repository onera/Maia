import Converter.Internal as I
# from .index_array         import create_index_array_filter
from .point_list          import create_point_list_filter

def create_zone_bc_filter(zone, zone_path, hdf_filter):
  """
  """
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]

      pl_n = I.getNodeFromName1(bc, 'PointList')
      pr_n = I.getNodeFromName1(bc, 'PointRange')
      if(pl_n):
        distrib_ud = I.getNodeFromName1(bc        , ':CGNS#Distribution')
        distrib_ia = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
        create_point_list_filter(bc, bc_path, "PointList", distrib_ia, hdf_filter)

      if(pl_n):
        distrib_ud = I.getNodeFromName1(bc        , ':CGNS#Distribution')
        distrib_ia = I.getNodeFromName1(distrib_ud, 'Distribution')[1]
        create_point_list_filter(bc, bc_path, "PointRange", distrib_ia, hdf_filter)
        # I.newIndexArray()
