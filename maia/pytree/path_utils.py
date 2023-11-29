from maia.pytree.typing import *

from . import predicate
from . import walkers_api as WAPI

# Keys to access TreeNode values
__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3


def path_head(path:str, i:int=-1) -> str:
  """
  Return the start of a path until elt i (excluded)
  """
  splited = path.split('/')
  return '/'.join(splited[0:i])

def path_tail(path:str, i:int=-1) -> str:
  """
  Return the end of a path from elt i (included)
  """
  splited = path.split('/')
  return '/'.join(splited[i:])

def update_path_elt(path:str, i:int, func:Callable[[str],str]) -> str:
  """
  Replace the ith element of the input path using the function func
  func take one argument, which is the original value of the ith element
  """
  splited = path.split('/')
  splited[i] = func(splited[i])
  return '/'.join(splited)

def predicates_to_paths(root:CGNSTree, predicates) -> List[str]:
  """
  An utility function searching descendants matching predicates,
  and returning the path of these nodes (instead of the nodes themselves)
  """
  paths = []
  for nodes in WAPI.iter_nodes_from_predicates(root, predicates, depth=[1,1], ancestors=True):
    paths.append('/'.join([n[__NAME__] for n in nodes]))
  return paths

def concretize_paths(root:CGNSTree, wanted_path_list:List[str], labels:List[str]) -> List[str]:
  """
  """
  all_paths = []
  for path in wanted_path_list:
    names = path.split('/')
    assert len(names) == len(labels)
    predicates = [lambda n, _name=name, _label=label: predicate.match_name_label(n, _name, _label) \
        for (name, label) in zip(names,labels)] 
    paths = predicates_to_paths(root, predicates)
    all_paths.extend(paths)

  return sorted(list(set(all_paths))) #Unique + sort

def paths_to_tree(paths:List[str], root_name='CGNSTree') -> CGNSTree:
  """
  Convert a list of paths to a CGNSTreeLike
  """
  def unroll(root):
    """ Internal recursive function """
    if root[__VALUE__] is None:
      return
    for path in root[__VALUE__]:
      first = path.split('/')[0]
      others = '/'.join(path.split('/')[1:])
      node = WAPI.get_node_from_predicate(root, lambda n : predicate.match_name(n, first), depth=[1,1])
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

  path_tree = [root_name, paths, [], None]
  unroll(path_tree)
  return path_tree

