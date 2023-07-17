

INDENT_SIZE = 4


class Tree:
  def __init__(self, node_value, sub_nodes = []):
    self.node_value = node_value
    self.sub_nodes  = sub_nodes

  @staticmethod
  def _to_string(spec, indent_sz):
    s = ' '*indent_sz + str(spec.node_value) + '\n'
    for sub in spec.sub_nodes:
      s += Tree._to_string(sub, indent_sz+INDENT_SIZE)
    return s

  def __str__(self):
    return Tree._to_string(self, 0)

  # depth_first_search interface {
  def children(self, spec):
    return iter(spec.sub_nodes)
  def roots(self):
    return iter([self])
  # depth_first_search interface }
