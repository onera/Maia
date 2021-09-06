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

  def __init__(self, root: TreeNode, predicate: Callable[[TreeNode], bool],
                     search=NodeParser.DEFAULT, depth=0, sort=FORWARD):
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
  def depth(self, value: str):
    if isinstance(value, int) and value >= 0:
      self._depth = value
    else:
      raise ValueError("depth must a integer >= 0.")

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
    self._parser = LevelNodeParser(depth=self.depth, sort=self.sort) if self.depth > 0 else NodeParser(sort=self.sort)
    func = getattr(self._parser, self.search)
    return func(self._root, self._predicate)
