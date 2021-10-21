from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
import numpy as np

from ._node_parsers import NodeParser, LevelNodeParser
from .compare import is_valid_node
TreeNode = List[Union[str, Optional[np.ndarray], List["TreeNode"]]]


# --------------------------------------------------------------------------
class NodeWalker:
  """ Return the first node found in the Python/CGNS tree """

  FORWARD  = lambda children:children
  BACKWARD = lambda children:reverse(children)

  def __init__(self, root: TreeNode,
                     predicate: Callable[[TreeNode], bool],
                     search: str=NodeParser.DEFAULT,
                     depth=None,
                     sort=FORWARD):
    """
    Hold all the manner to explore and parse the CGNS Tree

    Args:
        root (TreeNode): CGNS node root searching
        predicate (Callable[[TreeNode], bool]): condition to select node
        search (str, optional): 'dfs' for Depth-First-Search or 'bfs' for Breath-First-Search
        depth (int, optional): stop exploring after the limited depth
        sort (Callable[TreeNode], optional): parsing children sort
    """
    self.root      = root
    self.predicate = predicate
    # Register default value
    self.search = search
    self.depth  = depth
    self.sort   = sort

  @property
  def root(self):
    return self._root

  @root.setter
  def root(self, node: TreeNode):
    if is_valid_node(node):
      self._root = node

  @property
  def predicate(self):
    return self._predicate

  @predicate.setter
  def predicate(self, predicate: Callable[[TreeNode], bool]):
    if callable(predicate):
      self._predicate = predicate
    else:
      raise TypeError("predicate must be a callable function.")

  @property
  def search(self):
    return self._search

  @search.setter
  def search(self, value: str):
    if value in ['bfs', 'dfs']:
      self._search = value
    else:
      raise ValueError("search must 'bfs' or 'dfs'.")

  @property
  def depth(self):
    return self._depth

  @depth.setter
  def depth(self, value):
    if value is None:
      self._depth = value
    elif isinstance(value, int) and value >= 0:
      self._depth = value
    else:
      raise ValueError("depth must None or an integer >= 0.")

  @property
  def sort(self):
    return self._sort

  @sort.setter
  def sort(self, value):
    if callable(value):
      self._sort = value
    else:
      raise TypeError("sort must be a callable function.")

  @property
  def parser(self):
    return self._parser

  def __call__(self):
    # Create parser
    if self.depth is not None and self.depth >= 0:
      self._parser = LevelNodeParser(depth=self.depth, sort=self.sort)
    elif self.depth is None:
      self._parser = NodeParser(sort=self.sort)
    else:
      raise Exception(f"Wrong definition of depth '{self.depth}'.")
    func = getattr(self._parser, self.search)
    return func(self._root, self._predicate)
