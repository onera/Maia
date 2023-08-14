import weakref
from maia.pytree.graph.utils import list_iterator_type

INDENT_SIZE = 4

# Sometimes, we want the root to just gather all branches together.
# That is the purpose of SYMBOLIC_ROOT, whose only job is to symbolize a root with no value.
# Of course, we could just use `None`, or '/', but these could be valid node values.
# Hence, we create a custom type that is meant to be used for this sole purpose.
class _SymbolicRoot:
  # type with no state: all instances are equivalent
  def __eq__(self, other):
    # each _SymbolicRoot is equal to any other _SymbolicRoot but to no other type
    if isinstance(other, _SymbolicRoot):
      return True
    else:
      return False
  def __hash__(self):
    return 0
SYMBOLIC_ROOT = _SymbolicRoot()


# Mixins {
## Note: a 'mixin' is an implementation class that factorizes implementation of a partial interface,
##   and that can be inherited be several classes that want to provide this interface.
## Here we have the `_TreeToStrMixin` and `_TreeDepthFirstSearchInterfaceMixing` mixins
##   that are inherited by `Tree` and `ForwardBackwardTree`.
class _TreeToStrMixin:
  @staticmethod
  def _to_string_impl(tree, indent_sz, rec):
    s = ' '*indent_sz + str(tree.node_value) + '\n'
    for sub in tree.sub_nodes:
      if rec < 20:
        s += Tree._to_string_impl(sub, indent_sz+INDENT_SIZE, rec+1)
    return s

  @staticmethod
  def _to_string(tree):
    if tree.node_value == SYMBOLIC_ROOT:
      return ''.join([Tree._to_string_impl(sub, 0, 0) for sub in tree.sub_nodes])
    else:
      return Tree._to_string_impl(tree, 0, 0)

  def __str__(self):
    return Tree._to_string(self)

class _TreeDepthFirstSearchInterfaceMixing: # depth_first_search interface
  def child_iterator(self, tree) -> list_iterator_type:
    return iter(tree.sub_nodes)
  def root_iterator(self) -> list_iterator_type:
    return iter([self])
# Mixins }


class Tree(_TreeToStrMixin,_TreeDepthFirstSearchInterfaceMixing):
  """ Nested tree structure
      A nested tree is a recursive structure: it has
        - a `node_value` attribute,
        - a `sub_nodes` attribute that is a sequence of `Tree`s.
  """
  def __init__(self, node_value, sub_nodes = None):
    # Can't write `sub_nodes = []` directly because of mutable default arguments
    if sub_nodes is None: 
      sub_nodes = []

    # Precondition: all sub_nodes should be `Tree`s
    for c in sub_nodes:
      assert isinstance(c, Tree)

    self.node_value = node_value
    self.sub_nodes  = sub_nodes

class _ForwardBackwardTreeList:
  """
  List-like class to be used by `ForwardBackwardTree`
  Use case:
    Suppose that we have a `t` object of type `ForwardBackwardTree`,
    and we ask for its children: `cs = t.sub_nodes`.
    Now, if we change the first child: `cs[0] = ForwardBackwardTree(my_new_leaf)`
    Then we would like `cs[0].parent is t`

    For that to append, we need `cs.__setitem__(0, sub_tree)` to set the parent.
    And for that to work, `cs` can't be a regular Python `list`:
      we need to override `__setitem__` (and other methods),
      hence we make `t.sub_nodes` return a `_ForwardBackwardTreeList`
      and adapt the methods to set the parent
  """
  def __init__(self, l, parent):
    self._list = l
    self._parent = parent

  # Same methods as list {
  ## Note: these could be delegated through inheritance,
  ##       but on the other hand, inheritance is dangerous,
  ##       since we might inherit methods that should be replaced
  ##       without noticing they are present
  def __len__(self):
    return len(self._list)
  def __getitem__(self, i):
    return self._list[i]
  def __contains__(self, x):
    return x in self._list
  def clear(self):
    self._list.clear()
  def pop(self, i):
    return self._list.pop(i)
  # Same methods as list }

  # Methods to mutate elements to the list {
  ## We make the `.parent` of the element be the one of `self`
  def append(self, x):
    assert isinstance(x, ForwardBackwardTree)
    x.parent = self._parent
    self._list.append(x)
  def __setitem__(self, i, x):
    assert isinstance(x, ForwardBackwardTree)
    x.parent = self._parent
    self._list[i].parent = None # the previous child is orphaned
    self._list[i] = x
  def __iadd__(self, other):
    if isinstance(other, _ForwardBackwardTreeList):
      self._list += other._list
    else:
      self._list += other
    for x in other:
      x.parent = self._parent
    return self
  def insert(self, i, x):
    assert isinstance(x, ForwardBackwardTree)
    x.parent = self._parent
    self._list.insert(i, x)
  # Methods to mutate elements to the list }


class ForwardBackwardTree(_TreeToStrMixin,_TreeDepthFirstSearchInterfaceMixing):
  """
  `ForwardBackwardTree` means that we can go both directions within the tree:
    - either get the `sub_nodes` trees
    - or the `parent` tree
  """
  def __init__(self, node_value, sub_nodes = None, parent=None):
    # Can't write `sub_nodes = []` directly because of mutable default arguments
    if sub_nodes is None: 
      sub_nodes = []

    # Precondition: all `sub_nodes` and `parent` should be `ForwardBackwardTree`s

    for c in sub_nodes:
      assert isinstance(c, ForwardBackwardTree)
    if parent is not None:
      assert isinstance(parent, ForwardBackwardTree)

    self.node_value = node_value

    self._sub_nodes = sub_nodes
    for sub_node in self._sub_nodes:
      sub_node.parent = self

    self.parent = parent

  @property
  def parent(self):
    if self._parent_weakref is None:
      return None
    else:
      return self._parent_weakref()

  @parent.setter
  def parent(self, p):
    if p is None:
      self._parent_weakref = None
    else:
      self._parent_weakref = weakref.ref(p)

  @property
  def sub_nodes(self):
    return _ForwardBackwardTreeList(self._sub_nodes, self)

  @sub_nodes.setter
  def sub_nodes(self, cs):
    if isinstance(cs, _ForwardBackwardTreeList):
      self._sub_nodes = cs._list
    else:
      self._sub_nodes = cs
    for sub_node in self._sub_nodes:
      sub_node.parent = self
