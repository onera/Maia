from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
import numpy as np

from ._node_parsers import NodeParser, RangeLevelNodeParser
from .compare import is_valid_node
from .predicate import auto_predicate

TreeNode = List[Union[str, Optional[np.ndarray], List["TreeNode"]]]

# --------------------------------------------------------------------------
class NodeWalker:
  """ Return the first node found in the Python/CGNS tree """

  FORWARD  = lambda children:children
  BACKWARD = lambda children:reverse(children)

  def __init__(self, root: TreeNode,
                     predicate,
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
  def predicate(self, predicate):
    self._predicate = auto_predicate(predicate)

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
      self._depth = [0, None]
    elif isinstance(value, int) and value >= 0:
      self._depth = [0, value]
    elif isinstance(value, (tuple, list)):
      check1 = isinstance(value[1], int) and value[0] <= value[1] and all([i >= 0 for i in value])
      check2 = value[1] is None and value[0] >= 0
      if len(value) != 2 and (check1 or check2):
        raise Exception(f"depth must be define with only two positive integers, '{value}' given here.")
      self._depth = value
    else:
      raise ValueError("depth must None or an integer >= 0 (ex:3) or a range (ex:[1,3] or [1,None]).")

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
    if self.depth[0] == 0 and self.depth[1] is None:
      self._parser = NodeParser(sort=self.sort)
    else:
      self._parser = RangeLevelNodeParser(depth=self.depth, sort=self.sort)
    func = getattr(self._parser, self.search)
    return func(self._root, self._predicate)
