import Converter.Internal as I

from maia.cgns_io.hdf_filter import zone as ZEF

def create_tree_hdf_filter(dist_tree, hdf_filter, mode='read'):
  """
  On a besoin du write pour gérer le ElementStartIndex
  It can be replace by a if None in tree to see if read/write ?
  """
  for base in I.getNodesFromType1(dist_tree, 'CGNSBase_t'):
    for zone in I.getNodesFromType1(base, 'Zone_t'):
      zone_path = "/"+I.getName(base)+"/"+I.getName(zone)
      ZEF.create_zone_filter(zone, zone_path, hdf_filter, mode)
