from maia.pytree.graph.algo import dfs_interface_report, depth_first_search
from maia.pytree.graph.fe_graph import rooted_fe_graph_example

def test_fe_graph_tree_adaptor_is_depth_first_searchable():
  t = rooted_fe_graph_example()
  assert dfs_interface_report(t) == (True,'')


class visitor_to_test_depth_first_scan:
  def __init__(self):
    self.s = ''

  def pre(self, node_with_down_edge):
    self.s += '[pre ] ' + str(node_with_down_edge.node) + '\n'
  def post(self, node_with_down_edge):
    self.s += '[post] ' + str(node_with_down_edge.node) + '\n'
  def down(self, parent_node_with_down_edge, current_node_with_down_edge):
    self.s += '[down] ' + str(current_node_with_down_edge.edge) + '\n'
  def up(self, current_node_with_down_edge, parent_node_with_down_edge):
    self.s += '[up  ] ' + str(current_node_with_down_edge.edge) + '\n'

  def accumulation_string(self):
    return self.s

def test_depth_first_scan():
  #   Reminder:
  #                1
  #               /|\
  #              / | \
  #             /  |  \
  #           A/  E|  F\
  #           /    |    \
  #          /     |     \
  #         2      |      3
  #        /|\     |     /|\
  #       / | \    |   G/ | \
  #      /  |  \   |   /  |  \
  #    B/  C|  D\  |  8   |   \I
  #    /    |    \ | /    |H   \
  #   /     |     \|/J    |     \
  #  /      |      |      |      \
  # 4       7      9      10     11

  g = rooted_fe_graph_example()
  expected_s = \
    '[pre ] 1\n' \
    '[down] A\n' \
    '[pre ] 2\n' \
    '[down] B\n' \
    '[pre ] 4\n' \
    '[post] 4\n' \
    '[up  ] B\n' \
    '[down] C\n' \
    '[pre ] 7\n' \
    '[post] 7\n' \
    '[up  ] C\n' \
    '[down] D\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] D\n' \
    '[post] 2\n' \
    '[up  ] A\n' \
    '[down] E\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] E\n' \
    '[down] F\n' \
    '[pre ] 3\n' \
    '[down] G\n' \
    '[pre ] 8\n' \
    '[down] J\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] J\n' \
    '[post] 8\n' \
    '[up  ] G\n' \
    '[down] H\n' \
    '[pre ] 10\n' \
    '[post] 10\n' \
    '[up  ] H\n' \
    '[down] I\n' \
    '[pre ] 11\n' \
    '[post] 11\n' \
    '[up  ] I\n' \
    '[post] 3\n' \
    '[up  ] F\n' \
    '[post] 1\n'

  # 1. test with regular depth first search
  v = visitor_to_test_depth_first_scan()
  depth_first_search(g,v)
  assert v.accumulation_string() == expected_s


  # 2. Test with more a natural visitor for graph with node and edge values
  # Note: While we could put this directly in the library instead of in a test,
  #       we lack a real world use case to see if this is actually the interface we want

  # 2.1. Client code: the visitor that the user creates uses familiar `node` and `edge` concepts
  class visitor_to_test_depth_first_scan_alt:
    def __init__(self):
      self.s = ''

    def pre(self, node):
      self.s += '[pre ] ' + str(node) + '\n'
    def post(self, node):
      self.s += '[post] ' + str(node) + '\n'
    def down(self, edge):
      self.s += '[down] ' + str(edge) + '\n'
    def up(self, edge):
      self.s += '[up  ] ' + str(edge) + '\n'

    def accumulation_string(self):
      return self.s

  # 2.2. Library code: adapt the visitor of the client to one that matches `depth_first_search` requirements
  class visitor_with_edge_values:
    def __init__(self, v):
      self.v = v

    # Here we make the design choice that the iterator returned by `.children`
    # generates an object with `.node` and `.edge` attributes
    def pre(self, nwe):
      return v.pre(nwe.node)
    def post(self, nwe):
      return v.post(nwe.node)
    def down(self, parent, current):
      return v.down(current.edge)
    def up(self, current, parent):
      return v.up(current.edge)

  # 2.3. Library code: propose an alternative function that can be used by this new kind of visitor
  def depth_first_search_alt(g, v):
    return depth_first_search(g, visitor_with_edge_values(v))

  # 2.4. Test with the new visitor and the new function
  v = visitor_to_test_depth_first_scan_alt()
  depth_first_search_alt(g,v)

  # 2.5. Check we have the same result
  assert v.accumulation_string() == expected_s
