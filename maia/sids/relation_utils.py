import Converter.Internal as I

from maia.utils          import py_utils

def get_subregion_extent(sub_region_n, zone):
  """
  Return the path of the node (starting from zone node) related to sub_region_n
  node (BC, GC or itself)
  """
  assert I.getType(sub_region_n) == "ZoneSubRegion_t"
  if I.getNodeFromName1(sub_region_n, "BCRegionName") is not None:
    for zbc, bc in py_utils.getNodesWithParentsFromTypePath(zone, "ZoneBC_t/BC_t"):
      if I.getName(bc) == I.getValue(I.getNodeFromName1(sub_region_n, "BCRegionName")):
        return I.getName(zbc) + '/' + I.getName(bc)
  elif I.getNodeFromName1(sub_region_n, "GridConnectivityRegionName") is not None:
    gc_pathes = ["ZoneGridConnectivity_t/GridConnectivity_t", "ZoneGridConnectivity_t/GridConnectivity1to1_t"]
    for gc_path in gc_pathes:
      for zgc, gc in py_utils.getNodesWithParentsFromTypePath(zone, gc_path):
        if I.getName(gc) == I.getValue(I.getNodeFromName1(sub_region_n, "GridConnectivityRegionName")):
          return I.getName(zgc) + '/' + I.getName(gc)
  else:
    return I.getName(sub_region_n)

  raise ValueError("ZoneSubRegion {0} has no valid extent".format(I.getName(sub_region_n)))

