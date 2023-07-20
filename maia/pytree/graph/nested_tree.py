INDENT_SIZE = 4

# Sometimes, we want the root to just gather all branches together,
#   but treat it differently in some contexts
# Hence SYMBOLIC_ROOT, whose only job is to symbolize such a root
class _SymbolicRoot:
  # type with no state: all instances are equivalent
  def __eq__(self, other):
    # each _SymbolicRoot is equal to other _SymbolicRoot but no other type
    if isinstance(other, _SymbolicRoot):
      return True
    else:
      return False
  def __hash__(self):
    return 0
SYMBOLIC_ROOT = _SymbolicRoot()


class Tree:
  def __init__(self, node_value, sub_nodes = []):
    self.node_value = node_value
    self.sub_nodes  = sub_nodes

  @staticmethod
  def _to_string_impl(tree, indent_sz):
    s = ' '*indent_sz + str(tree.node_value) + '\n'
    for sub in tree.sub_nodes:
      s += Tree._to_string_impl(sub, indent_sz+INDENT_SIZE)
    return s

  @staticmethod
  def _to_string(tree):
    if tree.node_value == SYMBOLIC_ROOT:
      return ''.join([Tree._to_string_impl(sub, 0) for sub in tree.sub_nodes])
    else:
      return Tree._to_string_impl(tree, 0)

  def __str__(self):
    return Tree._to_string(self)

  # depth_first_search interface {
  def children(self, tree):
    return iter(tree.sub_nodes)
  def roots(self):
    return iter([self])
  # depth_first_search interface }


class FBTree(Tree): # TODO Test
  def __init__(self, node_value, sub_nodes = [], parent=None):
    for sub_node in sub_nodes:
      sub_node.parent = self
    super().__init__(node_value, sub_nodes)
    self.parent = parent
