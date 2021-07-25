import numpy as np
import Converter.Internal as I

from maia.sids.pytree import iter_children_from_predicates, get_children_from_predicates
from maia.sids.pytree import check_is_label

def create_value(label, value):
  # print(f"label = {label}, value = {value}")
  result = None
  if value is None:
    return result
  if label == 'CGNSLibraryVersion_t':
    if isinstance(value, int) or isinstance(value, float):
      result = np.array([value], 'f')
    elif isinstance(value, np.ndarray):
      result = np.array(value, 'f')
    else:
      raise TypeError("setValue: CGNSLibraryVersion node value should be a float.")
  else:
    # value as a numpy: convert to fortran order
    if isinstance(value, np.ndarray):
      # print(f"-> value as a numpy")
      if value.flags.f_contiguous:
        result = value
      else:
        result = np.asfortranarray(value)
    # value as a single value
    elif isinstance(value, float):   # "R8"
      # print(f"-> value as float")
      result = np.array([value],'d')
    elif isinstance(value, int):     # "I4"
      # print(f"-> value as int")
      result = np.array([value],'i')
    elif isinstance(value, str):     # "C1"
      # print(f"-> value as str with {CGK.cgns_to_dtype[CGK.C1]}")
      result = np.array([c for c in value], CGK.cgns_to_dtype[CGK.C1])
    elif isinstance(value, CGK.dtypes):
      # print(f"-> value as CGK.dtypes with {np.dtype(value)}")
      result = np.array([value], np.dtype(value))
    # value as an iterable (list, tuple, set, ...)
    elif isinstance(value, PYU.Iterable):
      # print(f"-> value as PYU.Iterable : {PYU.flatten(value)}")
      try:
        first_value = next(PYU.flatten(value))
        if isinstance(first_value, float):                       # "R8"
          # print(f"-> first_value as float")
          result = np.array(value, dtype=np.float64, order='F')
        elif isinstance(first_value, int):                       # "I4"
          # print(f"-> first_value as int")
          result = np.array(value, dtype=np.int32, order='F')
        elif isinstance(first_value, str):                       # "C1"
          # print(f"-> first_value as with {CGK.cgns_to_dtype[CGK.C1]}")
          # WARNING: string numpy is limited to rank=2
          size = max([len(v) for v in PYU.flatten(value)])
          if isinstance(value[0], str):
            v = np.empty( (size,len(value) ), dtype='c', order='F')
            for c, i in enumerate(value):
              s = min(len(i),size)
              v[:,c] = ' '
              v[0:s,c] = i[0:s]
            result = v
          else:
            v = np.empty( (size,len(value[0]),len(value) ), dtype='c', order='F')
            for c in range(len(value)):
              for d in range(len(value[c])):
                s = min(len(value[c][d]),size)
                v[:,d,c] = ' '
                v[0:s,d,c] = value[c][d][0:s]
            result = v
        elif isinstance(first_value, CGK.dtypes):
          result = np.array(value, dtype=np.dtype(first_value), order='F')
      except StopIteration:
        # empty iterable
        result = np.array(value, order='F')
    else:
      # print(f"-> value as unknown type")
      result = np.array([value], order='F')
  return result

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
