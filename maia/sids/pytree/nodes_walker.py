from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
# from functools import partial
import numpy as np

from ._node_parsers import NodesIterator
from ._node_parsers import ShallowNodesIterator
from ._node_parsers import RangeLevelNodesIterator
from ._node_parsers import ShallowRangeLevelNodesIterator
from .compare import is_valid_node

TreeNode = List[Union[str, Optional[np.ndarray], List["TreeNode"]]]

# --------------------------------------------------------------------------
class NodesWalker:
  """ Walker of pyTree """

  FORWARD  = lambda children:children
  BACKWARD = lambda children:reversed(children)

  def __init__(self, root: TreeNode,
                     predicate,
                     search: str=NodesIterator.DEFAULT,
                     explore: str='shallow',
                     depth=None,
                     sort=FORWARD,
                     caching: bool=False):
    """
    Hold all the manner to explore and parse the CGNS Tree

    Args:
        root (TreeNode): CGNS node root searching
        predicate (Callable[[TreeNode], bool]): condition to select node
        search (str, optional): 'dfs' for Depth-First-Search or 'bfs' for Breath-First-Search
        explore (str, optional): 'deep' explore the whole tree or 'shallow' stop exploring node child when the node is found
        depth (int, optional): stop exploring after the limited depth
        sort (Callable[TreeNode], optional): parsing children sort
        caching (bool, optional): Results is store into a list. Avoid parsing next call(s).
    """
    self.root      = root
    self.predicate = predicate
    # Register default value
    self.search  = search
    self.explore = explore
    self.depth   = depth
    self.sort    = sort
    self.caching = caching
    # Internal
    self._parser = None
    self._cache  = []

  @property
  def root(self):
    return self._root

  @root.setter
  def root(self, node: TreeNode):
    if is_valid_node(node):
      self._root = node
      self.clean()

  @property
  def predicate(self):
    return self._predicate

  @predicate.setter
  def predicate(self, predicate: Callable[[TreeNode], bool]):
    if callable(predicate):
      self._predicate = predicate
      self.clean()
    else:
      raise TypeError("predicate must be a callable function.")

  @property
  def search(self):
    return self._search

  @search.setter
  def search(self, value: str):
    if value in ['bfs', 'dfs']:
      self._search = value
      self.clean()
    else:
      raise ValueError("search must 'bfs' or 'dfs'.")

  @property
  def explore(self):
    return self._explore

  @explore.setter
  def explore(self, value: str):
    if value in ['deep', 'shallow']:
      self._explore = value
      self.clean()
    else:
      raise ValueError("search must 'deep' or 'shallow'.")

  @property
  def depth(self):
    return self._depth

  @depth.setter
  def depth(self, value):
    if value is None:
      self._depth = [0, None]
      self.clean()
    elif isinstance(value, int) and value >= 0:
      self._depth = [0,value]
      self.clean()
    elif isinstance(value, (tuple, list)):
      check1 = isinstance(value[1], int) and value[0] <= value[1] and all([i >= 0 for i in value])
      check2 = value[1] is None and value[0] >= 0
      if len(value) != 2 and (check1 or check2):
        raise Exception(f"depth must be define with only two positive integers, '{value}' given here.")
      self._depth = value
      self.clean()
    else:
      raise ValueError("depth must None or an integer >= 0 (ex:3) or a range (ex:[1,3] or [1,None]).")

  @property
  def sort(self):
    return self._sort

  @sort.setter
  def sort(self, value):
    if callable(value):
      self._sort = value
      self.clean()
    else:
      raise TypeError("sort must be a callable function.")

  @property
  def caching(self):
    return self._caching

  @caching.setter
  def caching(self, value):
    if isinstance(value, bool):
      self._caching = value
    else:
      raise TypeError("caching must be a boolean.")

  @property
  def cache(self):
    return self._cache

  @property
  def parser(self):
    return self._parser

  def _get_parser(self):
    if self.explore == "shallow":
      if self.depth[0] == 0 and self.depth[1] is None:
        parser = ShallowNodesIterator(sort=self.sort)
      else:
        parser = ShallowRangeLevelNodesIterator(depth=self.depth, sort=self.sort)
    else:
      if self.depth[0] == 0 and self.depth[1] is None:
        parser = NodesIterator(sort=self.sort)
      else:
        parser = RangeLevelNodesIterator(depth=self.depth, sort=self.sort)
    return parser

  def __call__(self):
    # Generate iterator
    self._parser = self._get_parser()
    walker   = getattr(self._parser, self.search)
    iterator = walker(self._root, self._predicate)
    if self.caching:
      if not bool(self._cache):
        self._cache = list(iterator)
      return self._cache
    else:
      return iterator

  def apply(self, f, *args, **kwargs):
    for n in self.__call__():
      f(n, *args, **kwargs)

  def clean(self):
    """ Reset the cache """
    self._cache = []

  def __del__(self):
    self.clean()
