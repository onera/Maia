import Converter.Internal as I

def getVal(t):
  return t[1]

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


def getNodesFromTypePath(root, types_path):
  """Generator following type path, equivalent to
  for level1 in I.getNodesFromType1(root, type1):
    for level2 in I.getNodesFromType1(level1, type2):
      for level3 in I.getNodesFromType1(level2, type3):
        ...
  """
  types_list = types_path.split('/')
  if len(types_list) > 1:
    next_root = I.getNodesFromType1(root, types_list[0])
    for node in next_root:
      yield from getNodesFromTypePath(node, '/'.join(types_list[1:]))
  elif len(types_list) == 1:
    nodes =  I.getNodesFromType1(root, types_list[0])
    yield from nodes

def getNodesWithParentsFromTypePath(root, types_path):
  """Same than getNodesFromTypePath, but return
  a tuple of size nb_types containing the node and its parents
  """
  types_list = types_path.split('/')
  if len(types_list) > 1:
    next_root = I.getNodesFromType1(root, types_list[0])
    for node in next_root:
      for subnode in getNodesWithParentsFromTypePath(node, '/'.join(types_list[1:])):
        yield (node, *subnode)
  elif len(types_list) == 1:
    nodes =  I.getNodesFromType1(root, types_list[0])
    for node in nodes:
      yield (node,)

def getNodesByMatching(root, query_path):
  """Generator following query_path, doing 1 level search using
  getNodesFromType1 or getNodesFromName1. Equivalent to
  (query = 'type1_t/name2/type3_t' )
  for level1 in I.getNodesFromType1(root, type1_t):
    for level2 in I.getNodesFromName1(level1, name2):
      for level3 in I.getNodesFromType1(level2, type3_t):
        ...
  """
  query_list = query_path.split('/')
  getNodes1 = I.getNodesFromType1 if query_list[0][-2:] == '_t' else I.getNodesFromName1
  if len(query_list) > 1:
    next_root = getNodes1(root, query_list[0])
    for node in next_root:
      yield from getNodesByMatching(node, '/'.join(query_list[1:]))
  elif len(query_list) == 1:
    nodes =  getNodes1(root, query_list[0])
    yield from nodes

def getNodesWithParentsByMatching(root, query_path):
  """Same than getNodesByMatching, but return
  a tuple of size len(query_path) containing the node and its parents
  """
  query_list = query_path.split('/')
  getNodes1 = I.getNodesFromType1 if query_list[0][-2:] == '_t' else I.getNodesFromName1
  if len(query_list) > 1:
    next_root = getNodes1(root, query_list[0])
    for node in next_root:
      for subnode in getNodesWithParentsByMatching(node, '/'.join(query_list[1:])):
        yield (node, *subnode)
  elif len(query_list) == 1:
    nodes =  getNodes1(root, query_list[0])
    for node in nodes:
      yield (node,)

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
  return getVal(I.getNodeFromPath(node, ':CGNS#Distribution/' + distri_name)) if distri_name \
      else I.getNodeFromName1(node, ':CGNS#Distribution')

def getGlobalNumbering(node, lngn_name=None):
  """
  Starting from node, return the CGNS#GlobalNumbering node if lngn_name is None
  or the value of the requested globalnumbering if lngn_name is not None
  """
  return getVal(I.getNodeFromPath(node, ':CGNS#GlobalNumbering/' + lngn_name)) if lngn_name \
      else I.getNodeFromName1(node, ':CGNS#GlobalNumbering')
