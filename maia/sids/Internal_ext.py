import numpy as np
import Converter.Internal as I
from maia.sids.cgns_keywords import Label as CGL
import maia.sids.cgns_keywords as CGK

from maia.sids.pytree import *

#Here is the legacy Internal_ext
# --------------------------------------------------------------------------
#
#   iterNodesByMatching
#
# --------------------------------------------------------------------------
def getNodesDispatch1(node, predicate):
  """ Interface to adapted getNodesFromXXX1 function depending of predicate type"""
  if isinstance(predicate, str):
    return getNodesFromLabel1(node, predicate) if is_valid_label(predicate) else getNodesFromName1(node, predicate)
  elif isinstance(predicate, CGK.Label):
    return getNodesFromLabel1(node, predicate.name)
  elif isinstance(predicate, np.ndarray):
    return getNodesFromValue1(node, predicate)
  elif callable(predicate):
    return getNodesFromPredicate1(node, predicate)
  else:
    raise TypeError("predicate must be a string for name, a numpy for value, a CGNS Label or a callable python function.")

def iterNodesByMatching(root, predicates):
  """Generator following predicates, doing 1 level search using
  getNodesFromLabel1 or getNodesFromName1. Equivalent to
  (predicate = 'type1_t/name2/type3_t' or ['type1_t', 'name2', lambda n: I.getType(n) == CGL.type3_t.name] )
  for level1 in I.getNodesFromType1(root, type1_t):
    for level2 in I.getNodesFromName1(level1, name2):
      for level3 in I.getNodesFromType1(level2, type3_t):
        ...
  """
  return iterNodesFromPredicates(root, predicates, search='dfs', depth=1)

def getNodesByMatching(root, predicates):
  return getNodesFromPredicates(root, predicates, search='dfs', depth=1)

iter_children_by_matching = iterNodesByMatching
get_children_by_matching  = getNodesByMatching

# --------------------------------------------------------------------------
#
#   iterNodesWithParentsByMatching
#
# --------------------------------------------------------------------------
def iterNodesWithParentsByMatching(root, predicates):
  """Same than iterNodesByMatching, but return
  a tuple of size len(predicates) containing the node and its parents
  """
  return iterNodesFromPredicates(root, predicates, search='dfs', depth=1, ancestors=True)

def getNodesWithParentsByMatching(root, predicates):
  return getNodesFromPredicates(root, predicates, search='dfs', depth=1, ancestors=True)


iter_children_with_parents_by_matching = iterNodesWithParentsByMatching
get_children_with_parents_by_matching  = getNodesWithParentsByMatching




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
  if requestNodeFromName1(sub_region_node, "BCRegionName") is not None:
    for zbc, bc in iterNodesWithParentsByMatching(zone, "ZoneBC_t/BC_t"):
      if I.getName(bc) == I.getValue(requestNodeFromName1(sub_region_node, "BCRegionName")):
        return I.getName(zbc) + '/' + I.getName(bc)
  elif requestNodeFromName1(sub_region_node, "GridConnectivityRegionName") is not None:
    gc_pathes = ["ZoneGridConnectivity_t/GridConnectivity_t", "ZoneGridConnectivity_t/GridConnectivity1to1_t"]
    for gc_path in gc_pathes:
      for zgc, gc in iterNodesWithParentsByMatching(zone, gc_path):
        if I.getName(gc) == I.getValue(requestNodeFromName1(sub_region_node, "GridConnectivityRegionName")):
          return I.getName(zgc) + '/' + I.getName(gc)
  else:
    return I.getName(sub_region_node)

  raise ValueError("ZoneSubRegion {0} has no valid extent".format(I.getName(sub_region_node)))

def getDistribution(node, distri_name=None):
  """
  Starting from node, return the CGNS#Distribution node if distri_name is None
  or the value of the requested distribution if distri_name is not None
  """
  return I.getNodeFromPath(node, '/'.join([':CGNS#Distribution', distri_name])) if distri_name \
      else requestNodeFromName1(node, ':CGNS#Distribution')

def getGlobalNumbering(node, lngn_name=None):
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  return I.getNodeFromPath(node, '/'.join([':CGNS#GlobalNumbering', lngn_name])) if lngn_name \
      else requestNodeFromName1(node, ':CGNS#GlobalNumbering')

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


get_node_dispatch1                    = getNodesDispatch1
iter_nodes_by_matching                = iterNodesByMatching
iter_nodes_with_parents_matching      = iterNodesWithParentsByMatching
get_subregion_extent                  = getSubregionExtent
get_distribution                      = getDistribution
get_global_numbering                  = getGlobalNumbering
new_distribution                      = newDistribution
new_global_numbering                  = newGlobalNumbering
