from .node_walker import NodeWalker


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

def get_node_from_predicates_for_each_with_parents__(parent, predicates, for_each):
  # Different kwargs + ancestors
  res = ()
  for predicate, kwargs in zip(predicates, for_each):
    next = NodeWalker(parent, predicate, **kwargs)()
    if next is None:
      return None
    res = (*res, next)
    parent = next
  return res

def get_node_from_predicates_with_parents__(parent, predicates, **kwargs):
  # Same kwargs + ancestors
  for_each_kw = [kwargs for _ in predicates]
  return get_node_from_predicates_for_each_with_parents__(parent, predicates, for_each_kw)

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
    self.ancestors  = kwargs.pop('ancestors', False)

  @property
  def root(self):
    return self._root

  @root.setter
  def root(self, node):
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
  def ancestors(self):
    return self._ancestors
  @ancestors.setter
  def ancestors(self, value):
    if isinstance(value, bool):
      self._ancestors = value
    else:
      raise TypeError("ancestors must be a boolean.")

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
      search = get_node_from_predicates_for_each_with_parents__ if self.ancestors else get_node_from_predicates_for_each__
      return search(self.root, predicates, for_each)
    else:
      search = get_node_from_predicates_with_parents__ if self.ancestors else get_node_from_predicates__
      return search(self.root, self.predicates, **self.kwargs)
