from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
import numpy as np

# Declare a type alias for type hint checkers
# For Python>=3.9 it is possible to set the MaxLen
# from typing import Annotated
# TreeNode = Annotated[List[Union[str, Optional[numpy.ndarray], List['TreeNode']]], MaxLen(4)]
TreeNode = List[Union[str, Optional[np.ndarray], List["TreeNode"]]]

# Keys to access TreeNode values
__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3

def rm_children_from_predicate(root: TreeNode, predicate: Callable[[TreeNode], bool]) -> NoReturn:
  """
  Remove the children of root node satisfying Predicate function
  """
  results = []
  for ichild, child in enumerate(root[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
  for ichild in reversed(results):
    del root[__CHILDREN__][ichild]

def keep_children_from_predicate(root: TreeNode, predicate: Callable[[TreeNode], bool]) -> NoReturn:
  """
  Remove all the children of root node expect the ones satisfying Predicate function
  """
  rm_children_from_predicate(root, lambda n: not predicate(n))


def rm_nodes_from_predicate(root, predicate, **kwargs):
  """
  Starting from root node, remove all the nodes matching Predicate function
  Removal can be limited to a given depth
  """
  depth = kwargs.get('depth')
  if depth and not isinstance(depth, int):
    raise TypeError(f"depth must be an integer.")
  if depth and depth >= 1:
    rm_nodes_from_predicate_with_level__(root, predicate, depth)
  else:
    rm_nodes_from_predicate__(root, predicate)

def rm_nodes_from_predicate_with_level__(parent, predicate, depth, level=1):
  results = []
  for ichild, child in enumerate(parent[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
    else:
      if level < depth:
        rm_nodes_from_predicate_with_level__(child, predicate, depth, level=level+1)
  for ichild in reversed(results):
    del parent[__CHILDREN__][ichild]


def rm_nodes_from_predicate__(parent, predicate):
  results = []
  for ichild, child in enumerate(parent[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
    else:
      rm_nodes_from_predicate__(child, predicate)
  for ichild in reversed(results):
    del parent[__CHILDREN__][ichild]

