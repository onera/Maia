from maia.pytree.graph.algo import dfs_interface_report, depth_first_search
from maia.pytree.graph.fe_graph import rooted_fe_graph_example

def test_fe_graph_tree_adaptor_is_depth_first_searchable():
  t = rooted_fe_graph_example()
  assert dfs_interface_report(t) == (True,'')


class visitor_for_testing_depth_first_scan:
  def __init__(self):
    self.s = ''

  def pre(self, nwe):
    self.s += '[pre ] ' + str(nwe.node) + '\n'
  def post(self, nwe):
    self.s += '[post] ' + str(nwe.node) + '\n'
  def down(self, parent, current):
    self.s += '[down] ' + str(current.edge) + '\n'
  def up(self, current, parent):
    self.s += '[up  ] ' + str(current.edge) + '\n'

  def accumulation_string(self):
    return self.s

def test_depth_first_scan():
  #   Reminder:
  #         1               lvl 3
  #    A /  |  \F
  #     |   |    3          lvl 2
  #     |  E| /G |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |J |H   \I
  #  B| C| D\ |  |     \
  #   4  7   \9  10    11   lvl 0

  g = rooted_fe_graph_example()
  v = visitor_for_testing_depth_first_scan()
  depth_first_search(g,v)

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

  assert v.accumulation_string() == expected_s
