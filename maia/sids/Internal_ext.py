import numpy as np
import Converter.Internal as I

from maia.sids.pytree import iter_children_from_predicates, get_children_from_predicates
from maia.sids.pytree import check_is_label

# Some old aliases to replace with iterNodesFromPredicates

def iterNodesByMatching(root, predicates):
  return iter_children_from_predicates(root, predicates)

def getNodesByMatching(root, predicates):
  return get_children_from_predicates(root, predicates)

def iterNodesWithParentsByMatching(root, predicates):
  return iter_children_from_predicates(root, predicates, ancestors=True)

def getNodesWithParentsByMatching(root, predicates):
  return get_children_from_predicates(root, predicates, ancestors=True)

# --------------------------------------------------------------------------
def getParentFromPredicate(start, node, predicate, prev=None):
    """Return thee first parent node matching type."""
    if id(start) == id(node):
      return prev
    if predicate(start):
      prev = start
    for n in start[2]:
        ret = getParentFromPredicate(n, node, parentType, prev)
        if ret is not None: return ret
    return None

def getParentsFromPredicate(start, node, predicate, l=[]):
    """Return all parent nodes matching type."""
    if id(start) == id(node):
      return l
    if predicate(start):
      l.append(start)
    for n in start[2]:
        ret = getParentsFromPredicate(n, node, predicate, l)
        if ret != []: return ret
    return []

# --------------------------------------------------------------------------
@check_is_label('ZoneSubRegion_t', 0)
@check_is_label('Zone_t', 1)
def getSubregionExtent(sub_region_node, zone):
  """
  Return the path of the node (starting from zone node) related to sub_region_node
  node (BC, GC or itself)
  """
  if I.getNodeFromName1(sub_region_node, "BCRegionName") is not None:
    for zbc, bc in iterNodesWithParentsByMatching(zone, "ZoneBC_t/BC_t"):
      if I.getName(bc) == I.getValue(I.getNodeFromName1(sub_region_node, "BCRegionName")):
        return I.getName(zbc) + '/' + I.getName(bc)
  elif I.getNodeFromName1(sub_region_node, "GridConnectivityRegionName") is not None:
    gc_pathes = ["ZoneGridConnectivity_t/GridConnectivity_t", "ZoneGridConnectivity_t/GridConnectivity1to1_t"]
    for gc_path in gc_pathes:
      for zgc, gc in iterNodesWithParentsByMatching(zone, gc_path):
        if I.getName(gc) == I.getValue(I.getNodeFromName1(sub_region_node, "GridConnectivityRegionName")):
          return I.getName(zgc) + '/' + I.getName(gc)
  else:
    return I.getName(sub_region_node)

  raise ValueError("ZoneSubRegion {0} has no valid extent".format(I.getName(sub_region_node)))

def getZoneDonorPath(current_base, gc):
  """
  Returns the Base/Zone path of the opposite zone of a gc node (add the Base/
  part if not present, using current_base name
  """
  opp_zone = I.getValue(gc)
  return opp_zone if '/' in opp_zone else current_base + '/' + opp_zone

def enforceDonorAsPath(tree):
  """ Force the GCs to indicate their opposite zone under the form BaseName/ZoneName """
  predicates = ['Zone_t', 'ZoneGridConnectivity_t', lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']]
  for base in I.getBases(tree):
    base_n = I.getName(base)
    for gc in iterNodesByMatching(base, predicates):
      I.setValue(gc, getZoneDonorPath(base_n, gc))

def getDistribution(node, distri_name=None):
  """
  Starting from node, return the CGNS#Distribution node if distri_name is None
  or the value of the requested distribution if distri_name is not None
  """
  return I.getNodeFromPath(node, '/'.join([':CGNS#Distribution', distri_name])) if distri_name \
      else I.getNodeFromName1(node, ':CGNS#Distribution')

def getGlobalNumbering(node, lngn_name=None):
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  return I.getNodeFromPath(node, '/'.join([':CGNS#GlobalNumbering', lngn_name])) if lngn_name \
      else I.getNodeFromName1(node, ':CGNS#GlobalNumbering')

# --------------------------------------------------------------------------
def newDistribution(distributions = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add distribution arrays specified in distributions dictionnary.
  distributions must be a dictionnary {DistriName : distri_array}
  """
  distri_node = I.newUserDefinedData(':CGNS#Distribution', None, parent)
  for name, value in distributions.items():
    I.newDataArray(name, value, parent=distri_node)
  return distri_node

def newGlobalNumbering(glob_numberings = dict(), parent=None):
  """
  Create and return a CGNSNode to be used to store distribution data
  Attach it to parent node if not None
  In addition, add global numbering arrays specified in glob_numberings dictionnary.
  glob_numberings must be a dictionnary {NumberingName : lngn_array}
  """
  lngn_node = I.newUserDefinedData(':CGNS#GlobalNumbering', None, parent)
  for name, value in glob_numberings.items():
    I.newDataArray(name, value, parent=lngn_node)
  return lngn_node

# --------------------------------------------------------------------------

get_subregion_extent                  = getSubregionExtent
get_distribution                      = getDistribution
get_global_numbering                  = getGlobalNumbering
new_distribution                      = newDistribution
new_global_numbering                  = newGlobalNumbering
