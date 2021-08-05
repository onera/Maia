from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
import numpy as np
import copy

from .compare      import is_valid_node
from .nodes_walker import NodesWalker

TreeNode = List[Union[str, Optional[np.ndarray], List["TreeNode"]]]

#We put here subfunctions used by NodesWalkers (based on NodesWalker)
#since they don't do directly tree parsing

def iter_nodes_from_predicates_for_each__(parent, predicates, for_each):
  # print("iter_nodes_from_predicates_for_each__")
  if len(predicates) > 1:
    for node in NodesWalker(parent, predicates[0], **for_each[0])():
      yield from iter_nodes_from_predicates_for_each__(node, predicates[1:], for_each[1:])
  elif len(predicates) == 1:
    yield from NodesWalker(parent, predicates[0], **for_each[0])()

def iter_nodes_from_predicates__(parent, predicates, **kwargs):
  # print("iter_nodes_from_predicates__")
  if len(predicates) > 1:
    for node in NodesWalker(parent, predicates[0], **kwargs)():
      yield from iter_nodes_from_predicates__(node, predicates[1:], **kwargs)
  elif len(predicates) == 1:
    yield from NodesWalker(parent, predicates[0], **kwargs)()

def iter_nodes_from_predicates_with_parents_for_each__(parent, predicates, for_each):
  # print("iter_nodes_from_predicates_with_parents_for_each__")
  if len(predicates) > 1:
    for node in NodesWalker(parent, predicates[0], **for_each[0])():
      for subnode in iter_nodes_from_predicates_with_parents_for_each__(node, predicates[1:], for_each[1:]):
        yield (node, *subnode)
  elif len(predicates) == 1:
    for node in NodesWalker(parent, predicates[0], **for_each[0])():
      yield (node,)

def iter_nodes_from_predicates_with_parents__(parent, predicates, **kwargs):
  # print("iter_nodes_from_predicates_with_parents__")
  if len(predicates) > 1:
    for node in NodesWalker(parent, predicates[0], **kwargs)():
      for subnode in iter_nodes_from_predicates_with_parents__(node, predicates[1:], **kwargs):
        yield (node, *subnode)
  elif len(predicates) == 1:
    for node in NodesWalker(parent, predicates[0], **kwargs)():
      yield (node,)


# --------------------------------------------------------------------------
#
#   NodesWalkers
#
# --------------------------------------------------------------------------
class NodesWalkers:

  def __init__(self, root, predicates, **kwargs):
    self.root       = root
    self.predicates = predicates
    self.kwargs     = kwargs
    self.ancestors  = kwargs.get('ancestors', False)
    if kwargs.get('ancestors'):
      kwargs.pop('ancestors')
    self._cache = []

  @property
  def root(self):
    return self._root

  @root.setter
  def root(self, node: TreeNode):
    if is_valid_node(node):
      self._root = node
      self.clean()

  @property
  def predicates(self):
    return self._predicates

  @predicates.setter
  def predicates(self, predicates):
    self._predicates = []
    if isinstance(predicates, (list, tuple)):
      for p in predicates:
        if not (callable(p) or (isinstance(p, dict) and callable(p['predicate']))):
          raise TypeError("Non callable function found in predicates list")
    else:
      raise TypeError("predicates must be a sequence of callable functions")
    self._predicates = predicates
    self.clean()

  @property
  def ancestors(self):
    return self._ancestor

  @ancestors.setter
  def ancestors(self, value):
    if isinstance(value, bool):
      self._ancestor = value
      self.clean()
    else:
      raise TypeError("ancestors must be a boolean.")

  @property
  def caching(self):
    return self.kwargs.get("caching", False)

  @caching.setter
  def caching(self, value):
    if isinstance(value, bool):
      self.kwargs['caching'] = value
      self.clean()
    else:
      raise TypeError("caching must be a boolean.")

  @property
  def cache(self):
    return self._cache

  @property
  def parser(self):
    return self._parser

  def _deconv_kwargs(self):
    predicates = []; for_each = []
    for kwargs in self.predicates:
      lkwargs = {}
      for k,v in kwargs.items():
        if k == 'predicate':
          predicates.append(v)
        else:
          lkwargs[k] = v
      for_each.append(lkwargs)
    if len(predicates) != len(self.predicates):
      raise ValueError(f"Missing predicate.")
    return predicates, for_each

  def __call__(self):
    if self.ancestors:
      return self._parse_with_parents()
    else:
      return self._parse()

  def _parse_with_parents(self):
    if any([isinstance(kwargs, dict) for kwargs in self.predicates]):
      predicates, for_each = self._deconv_kwargs()
      for index, kwargs in enumerate(for_each):
        if kwargs.get('caching'):
          print(f"Warning: unable to activate caching for predicate at index {index}.")
          kwargs['caching'] = False
      if self.caching:
        if not bool(self._cache):
          self._cache = list(iter_nodes_from_predicates_with_parents_for_each__(self.root, predicates, for_each))
        return self._cache
      else:
        return iter_nodes_from_predicates_with_parents_for_each__(self.root, predicates, for_each)
    else:
      if self.caching:
        if not bool(self._cache):
          kwargs = copy.deepcopy(self.kwargs)
          kwargs['caching'] = False
          self._cache = list(iter_nodes_from_predicates_with_parents__(self.root, self.predicates, **kwargs))
        return self._cache
      else:
        return iter_nodes_from_predicates_with_parents__(self.root, self.predicates, **self.kwargs)

  def _parse(self):
    if any([isinstance(kwargs, dict) for kwargs in self.predicates]):
      predicates, for_each = self._deconv_kwargs()
      for index, kwargs in enumerate(for_each):
        if kwargs.get('caching'):
          print(f"Warning: unable to activate caching for predicate at index {index}.")
          kwargs['caching'] = False
      if self.caching:
        if not bool(self._cache):
          self._cache = list(iter_nodes_from_predicates_for_each__(self.root, predicates, for_each))
        return self._cache
      else:
        return iter_nodes_from_predicates_for_each__(self.root, predicates, for_each)
    else:
      if self.caching:
        if not bool(self._cache):
          kwargs = copy.deepcopy(self.kwargs)
          kwargs['caching'] = False
          self._cache = list(iter_nodes_from_predicates__(self.root, self.predicates, **kwargs))
        return self._cache
      else:
        return iter_nodes_from_predicates__(self.root, self.predicates, **self.kwargs)

  def apply(self, f, *args, **kwargs):
    for n in self.__call__():
      f(n, *args, **kwargs)

  def clean(self):
    """ Reset the cache """
    self._cache = []

  def __del__(self):
    self.clean()

