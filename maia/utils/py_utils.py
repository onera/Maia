import Converter.Internal as I
import numpy as np

def list_or_only_elt(l):
  return l[0] if len(l) == 1 else l

def interweave_arrays(array_list):
  #https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
  first  = array_list[0]
  number = len(array_list)
  output = np.empty(number*first.size, first.dtype)
  for i,array in enumerate(array_list):
    output[i::number] = array
  return output

def concatenate_point_list(point_lists, dtype=None):
  """
  Merge all the PointList arrays in point_lists list
  into a flat 1d array and an index array
  """
  sizes = [pl_n.size for pl_n in point_lists]

  merged_pl_idx = np.empty(len(sizes)+1, dtype='int32')
  merged_pl_idx[0] = 0
  np.cumsum(sizes, out=merged_pl_idx[1:])

  if dtype is None:
    dtype = point_lists[0].dtype if point_lists != [] else np.int

  merged_pl = np.empty(sum(sizes), dtype=dtype)
  for ipl, pl in enumerate(point_lists):
    merged_pl[merged_pl_idx[ipl]:merged_pl_idx[ipl+1]] = pl[0,:]

  return merged_pl_idx, merged_pl

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
