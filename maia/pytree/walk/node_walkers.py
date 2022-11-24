from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
import numpy as np

from .node_walker import NodeWalker

TreeNode = List[Union[str, Optional[np.ndarray], List["TreeNode"]]]


# --------------------------------------------------------------------------
def get_node_from_predicates_for_each__(parent, predicates, for_each):
  # Different kwargs
  if len(predicates) > 1:
    node = NodeWalker(parent, predicates[0], **for_each[0])()
    if node is not None:
      return get_node_from_predicates_for_each__(node, predicates[1:], for_each[1:])
    else:
      return None
  elif len(predicates) == 1:
    return NodeWalker(parent, predicates[0], **for_each[0])()

def get_node_from_predicates__(parent, predicates, **kwargs):
  # Same kwargs
  if len(predicates) > 1:
    node = NodeWalker(parent, predicates[0], **kwargs)()
    if node is not None:
      return get_node_from_predicates__(node, predicates[1:], **kwargs)
  elif len(predicates) == 1:
    return NodeWalker(parent, predicates[0], **kwargs)()


# --------------------------------------------------------------------------
#
#   NodeWalkers
#
# --------------------------------------------------------------------------
class NodeWalkers:

  def __init__(self, root, predicates, **kwargs):
    self.root       = root
    self.predicates = predicates
    self.kwargs     = kwargs

  @property
  def root(self):
    return self._root

  @root.setter
  def root(self, node: TreeNode):
    self._root = node

  @property
  def predicates(self):
    return self._predicates

  @predicates.setter
  def predicates(self, predicates):
    self._predicates = []
    if isinstance(predicates, (list, tuple)):
      for p in predicates:
        self._predicates.append(p)
    else:
      self._predicates.append(predicates)

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
    if any([isinstance(kwargs, dict) for kwargs in self.predicates]):
      predicates, for_each = self._deconv_kwargs()
      return get_node_from_predicates_for_each__(self.root, predicates, for_each)
    else:
      return get_node_from_predicates__(self.root, self.predicates, **self.kwargs)
