from maia.pytree.typing import *

from maia.pytree                  import pred
from maia.pytree.walk.walkers_api import predicates_to_paths

#begin_api_export()


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


def concretize_paths(root:CGNSTree, wanted_path_list:List[str], labels:List[str]) -> List[str]:
  """
  """
  all_paths = []
  for path in wanted_path_list:
    names = path.split('/')
    assert len(names) == len(labels)
    predicates:Predicates = [pred.name_matches(name) & pred.label_matches(label) for (name, label) in zip(names,labels)]
    paths = predicates_to_paths(root, predicates)
    all_paths.extend(paths)

  return sorted(list(set(all_paths))) #Unique + sort

def paths_to_tree(paths:List[str], root_name='CGNSTree') -> CGNSTree:
  """
  Convert a list of paths to a CGNSTreeLike
  """
  path_tree:CGNSTree = [root_name, None, [], None] #type:ignore[assignment]
  for path in paths:
    node = path_tree
    for name in [n for n in path.split('/') if n]:
      try:
        next_node = next(n for n in node[2] if n[0] == name)
      except StopIteration:
        next_node:CGNSTree = [name, None, [], None] #type:ignore[assignment,no-redef]
        node[2].append(next_node)
      node = next_node
  return path_tree

#end_api_export()