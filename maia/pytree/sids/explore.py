import Converter.Internal as I

from maia.utils import py_utils

from maia.pytree.compare import check_is_label
import maia.pytree as PT

def getZoneDonorPath(current_base, gc):
  """
  Returns the Base/Zone path of the opposite zone of a gc node (add the Base/
  part if not present, using current_base name
  """
  opp_zone = I.getValue(gc)
  return opp_zone if '/' in opp_zone else current_base + '/' + opp_zone


@check_is_label('ZoneSubRegion_t', 0)
@check_is_label('Zone_t', 1)
def getSubregionExtent(sub_region_node, zone):
  """
  Return the path of the node (starting from zone node) related to sub_region_node
  node (BC, GC or itself)
  """
  if I.getNodeFromName1(sub_region_node, "BCRegionName") is not None:
    for zbc, bc in PT.iter_children_from_predicates(zone, "ZoneBC_t/BC_t", ancestors=True):
      if I.getName(bc) == I.getValue(I.getNodeFromName1(sub_region_node, "BCRegionName")):
        return I.getName(zbc) + '/' + I.getName(bc)
  elif I.getNodeFromName1(sub_region_node, "GridConnectivityRegionName") is not None:
    gc_pathes = ["ZoneGridConnectivity_t/GridConnectivity_t", "ZoneGridConnectivity_t/GridConnectivity1to1_t"]
    for gc_path in gc_pathes:
      for zgc, gc in PT.iter_children_from_predicates(zone, gc_path, ancestors=True):
        if I.getName(gc) == I.getValue(I.getNodeFromName1(sub_region_node, "GridConnectivityRegionName")):
          return I.getName(zgc) + '/' + I.getName(gc)
  else:
    return I.getName(sub_region_node)

  raise ValueError("ZoneSubRegion {0} has no valid extent".format(I.getName(sub_region_node)))


def find_connected_zones(tree):
  """
  Return a list of groups of zones (ie their path from root tree) connected through
  non periodic match grid connectivities (GridConnectivity_t & GridConnectivity1to1_t
  without Periodic_t node).
  """
  connected_zones = []
  matching_gcs_u = lambda n : I.getType(n) == 'GridConnectivity_t' \
                          and I.getValue(I.getNodeFromType1(n, 'GridConnectivityType_t')) == 'Abutting1to1'
  matching_gcs_s = lambda n : I.getType(n) == 'GridConnectivity1to1_t'
  matching_gcs = lambda n : (matching_gcs_u(n) or matching_gcs_s(n)) \
                          and I.getNodeFromType1(n, 'GridConnectivityProperty_t') is None
  
  for base, zone in PT.iter_children_from_predicates(tree, 'CGNSBase_t/Zone_t', ancestors=True):
    zone_path = I.getName(base) + '/' + I.getName(zone)
    group     = [zone_path]
    for gc in PT.iter_children_from_predicates(zone, ['ZoneGridConnectivity_t', matching_gcs]):
      opp_zone_path = getZoneDonorPath(I.getName(base), gc)
      py_utils.append_unique(group, opp_zone_path)
    connected_zones.append(group)

  for base, zone in PT.iter_children_from_predicates(tree, 'CGNSBase_t/Zone_t', ancestors=True):
    zone_path     = I.getName(base) + '/' + I.getName(zone)
    groups_to_merge = []
    for i, group in enumerate(connected_zones):
      if zone_path in group:
        groups_to_merge.append(i)
    if groups_to_merge != []:
      new_group = []
      for i in groups_to_merge[::-1]: #Reverse loop to pop without changing idx
        zones_paths = connected_zones.pop(i)
        for z_p in zones_paths:
          py_utils.append_unique(new_group, z_p)
      connected_zones.append(new_group)
  return [sorted(zones) for zones in connected_zones]

