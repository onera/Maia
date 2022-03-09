from . import predicate
from . import walkers_api as WAPI

# Keys to access TreeNode values
__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3

def predicates_to_pathes(root, predicates):
  """
  An utility function searching node descendance from root matching given predicates,
  and returning the path of these nodes (instead of the nodes itselves)
  """
  pathes = []
  for nodes in WAPI.iter_nodes_from_predicates(root, predicates, depth=[1,1], ancestors=True):
    pathes.append('/'.join([n[__NAME__] for n in nodes]))
  return pathes

def concretise_pathes(root, wanted_path_list, labels):
  """
  """
  all_pathes = []
  for path in wanted_path_list:
    names = path.split('/')
    assert len(names) == len(labels)
    predicates = [lambda n, _name=name, _label=label: predicate.match_name_label(n, _name, _label) \
        for (name, label) in zip(names,labels)] 
    pathes = predicates_to_pathes(root, predicates)
    all_pathes.extend(pathes)

  return sorted(list(set(all_pathes))) #Unique + sort

def pathes_to_tree(pathes, root_name='CGNSTree'):
  """
  Convert a list of pathes to a CGNSTreeLike
  """
  def unroll(root):
    """ Internal recursive function """
    if root[__VALUE__] is None:
      return
    for path in root[__VALUE__]:
      first = path.split('/')[0]
      others = '/'.join(path.split('/')[1:])
      node = WAPI.request_node_from_predicate(root, lambda n : predicate.match_name(n, first), depth=[1,1])
      if node is None:
        if others:
          root[__CHILDREN__].append([first, [others], [], None])
        else:
          root[__CHILDREN__].append([first, None, [], None])
      else:
        node[__VALUE__].append(others)
    root[__VALUE__] = None
    for child in root[__CHILDREN__]:
      unroll(child)

  path_tree = [root_name, pathes, [], None]
  unroll(path_tree)
  return path_tree

