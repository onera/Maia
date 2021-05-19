import fnmatch
import numpy as np
import Converter.Internal as I
import maia.sids.cgns_keywords as CGK

def getValue(t):
  return t[1]

def getNodesFromQuery1(node, query):
    result = []
    isStd = I.isStdNode(node)
    if isStd >= 0:
        for c in node[isStd:]:
          getNodesFromQuery1__(c, query, result)
    else:
      getNodesFromQuery1__(node, query, result)
    return result

def getNodesFromQuery1__(node, query, result):
    if query(node):
      result.append(node)
    for c in node[2]:
      if query(c):
        result.append(c)

def getNodesFromName1(node, name):
  return getNodesFromQuery1(node, lambda n : fnmatch.fnmatch(n[0], name))

def getNodesFromType1(node, label):
  return getNodesFromQuery1(node, lambda n : n[3] == label)

def getNodesFromValue1(node, value):
  return getNodesFromQuery1(node, lambda n : np.array_equal(n[1], value))


def getNodesDispatch1(node, query):
  if isinstance(query, str):
    if query.endswith('_t') and query in CGK.Label.__members__:
      return getNodesFromType1(node, query)
    else:
      return getNodesFromName1(node, query)
  elif isinstance(query, CGK.Label):
    return getNodesFromType1(node, query.name)
  elif isinstance(query, np.ndarray):
    return getNodesFromValue1(node, query)
  elif callable(query):
    return getNodesFromQuery1(node, query)
  else:
    raise TypeError("query must be a string for name, a numpy for value, a CGNS Label or a callable python function.")


def getSubregionExtent(sub_region_n, zone):
  """
  Return the path of the node (starting from zone node) related to sub_region_n
  node (BC, GC or itself)
  """
  assert I.getType(sub_region_n) == "ZoneSubRegion_t"
  if I.getNodeFromName1(sub_region_n, "BCRegionName") is not None:
    for zbc, bc in getNodesWithParentsFromTypePath(zone, "ZoneBC_t/BC_t"):
      if I.getName(bc) == I.getValue(I.getNodeFromName1(sub_region_n, "BCRegionName")):
        return I.getName(zbc) + '/' + I.getName(bc)
  elif I.getNodeFromName1(sub_region_n, "GridConnectivityRegionName") is not None:
    gc_pathes = ["ZoneGridConnectivity_t/GridConnectivity_t", "ZoneGridConnectivity_t/GridConnectivity1to1_t"]
    for gc_path in gc_pathes:
      for zgc, gc in getNodesWithParentsFromTypePath(zone, gc_path):
        if I.getName(gc) == I.getValue(I.getNodeFromName1(sub_region_n, "GridConnectivityRegionName")):
          return I.getName(zgc) + '/' + I.getName(gc)
  else:
    return I.getName(sub_region_n)

  raise ValueError("ZoneSubRegion {0} has no valid extent".format(I.getName(sub_region_n)))


def getNodesByMatching(root, queries):
  """Generator following queries, doing 1 level search using
  getNodesFromType1 or getNodesFromName1. Equivalent to
  (query = 'type1_t/name2/type3_t' or ['type1_t', 'name2', lambda n: I.getType(n) == CGL.type3_t.name] )
  for level1 in I.getNodesFromType1(root, type1_t):
    for level2 in I.getNodesFromName1(level1, name2):
      for level3 in I.getNodesFromType1(level2, type3_t):
        ...
  """
  query_list = []
  if isinstance(queries, str):
    for query in queries.split('/'):
      query_list.append(eval(query) if query.startswith('lambda') else query)
  elif isinstance(queries, (list, tuple)):
    query_list = queries
  else:
    raise TypeError("getNodesByMatching: queries must be a sequence or a path as with strings separated by '/'.")

  # query_list = query_path.split('/')
  # getNodes1 = I.getNodesFromType1 if query_list[0][-2:] == '_t' else I.getNodesFromName1
  if len(query_list) > 1:
    next_root = getNodesDispatch1(root, query_list[0])
    for node in next_root:
      yield from getNodesByMatching(node, query_list[1:])
  elif len(query_list) == 1:
    nodes =  getNodesDispatch1(root, query_list[0])
    yield from nodes

def getNodesWithParentsByMatching(root, queries):
  """Same than getNodesByMatching, but return
  a tuple of size len(queries) containing the node and its parents
  """
  query_list = []
  if isinstance(queries, str):
    for query in queries.split('/'):
      query_list.append(eval(query) if query.startswith('lambda') else query)
  elif isinstance(queries, (list, tuple)):
    query_list = queries
  else:
    raise TypeError("getNodesWithParentsByMatching: queries must be a sequence or a path with strings separated by '/'.")

  # query_list = query_path.split('/')
  # getNodes1 = I.getNodesFromType1 if query_list[0][-2:] == '_t' else I.getNodesFromName1
  if len(query_list) > 1:
    next_root = getNodesDispatch1(root, query_list[0])
    for node in next_root:
      for subnode in getNodesWithParentsByMatching(node, query_list[1:]):
        yield (node, *subnode)
  elif len(query_list) == 1:
    nodes =  getNodesDispatch1(root, query_list[0])
    for node in nodes:
      yield (node,)

def getNodesFromTypeMatching(root, queries):
  """Generator following type queries, equivalent to
  for level1 in I.getNodesFromType1(root, type1):
    for level2 in I.getNodesFromType1(level1, type2):
      for level3 in I.getNodesFromType1(level2, type3):
        ...
  """
  types_list = []
  if isinstance(queries, str):
    types_list = queries.split('/')
  elif isinstance(queries, (list, tuple)):
    types_list = queries
  else:
    raise TypeError("getNodesFromTypeMatching: queries must be a sequence of CGNS label or a path with CGNS labels separated by '/'.")

  if len(types_list) > 1:
    next_root = I.getNodesFromType1(root, types_list[0])
    for node in next_root:
      yield from getNodesFromTypeMatching(node, types_list[1:])
  elif len(types_list) == 1:
    nodes =  I.getNodesFromType1(root, types_list[0])
    yield from nodes

def getNodesWithParentsFromTypeMatching(root, queries):
  """Same than getNodesWithParentsFromTypeMatching, but return
  a tuple of size nb_types containing the node and its parents
  """
  types_list = []
  if isinstance(queries, str):
    types_list = queries.split('/')
  elif isinstance(queries, (list, tuple)):
    types_list = queries
  else:
    raise TypeError("getNodesFromTypeMatching: queries must be a sequence of CGNS label or a path with CGNS labels separated by '/'.")

  if len(types_list) > 1:
    next_root = I.getNodesFromType1(root, types_list[0])
    for node in next_root:
      for subnode in getNodesWithParentsFromTypeMatching(node, types_list[1:]):
        yield (node, *subnode)
  elif len(types_list) == 1:
    nodes =  I.getNodesFromType1(root, types_list[0])
    for node in nodes:
      yield (node,)

getNodesFromTypePath = getNodesFromTypeMatching
getNodesWithParentsFromTypePath = getNodesWithParentsFromTypeMatching

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

def getDistribution(node, distri_name=None):
  """
  Starting from node, return the CGNS#Distribution node if distri_name is None
  or the value of the requested distribution if distri_name is not None
  """
  return getValue(I.getNodeFromPath(node, ':CGNS#Distribution/' + distri_name)) if distri_name \
      else I.getNodeFromName1(node, ':CGNS#Distribution')

def getGlobalNumbering(node, lngn_name=None):
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  return getValue(I.getNodeFromPath(node, ':CGNS#GlobalNumbering/' + lngn_name)) if lngn_name \
      else I.getNodeFromName1(node, ':CGNS#GlobalNumbering')
