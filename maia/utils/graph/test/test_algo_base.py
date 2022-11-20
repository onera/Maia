from maia.utils.graph.algo_base import step, depth_first_search, depth_first_prune

from maia.utils.graph.io_graph import node_value, set_node_value, rooted_tree_example


class visitor_for_testing_depth_first_scan:
  def __init__(self):
    self.s = ''

  def pre(self, x):
    self.s += '[pre ] ' + str(node_value(x)) + '\n'
  def post(self, x):
    self.s += '[post] ' + str(node_value(x)) + '\n'
  def up(self, below, above):
    self.s += '[up  ] ' + str(node_value(below)) + ' -> ' + str(node_value(above)) + '\n'
  def down(self, above, below):
    self.s += '[down] ' + str(node_value(above)) + ' -> ' + str(node_value(below)) + '\n'

  def accumulation_string(self):
    return self.s

#TEST_CASE('depth_first_scan') {
#  /* Reminder:
#         1               lvl 3
#      /  |  \
#     |   |    3          lvl 2
#     |   | /  |  \
#     2\_ | 8  |   \      lvl 1
#   /  \ \| |  |    \
#   |  |  \ |  |    \
#  4    7  \9  10   11    lvl 0
#  */
#  auto g = create_rooted_graph_for_tests()
#  auto S = graph_traversal_stack(first_root(g),last_root(g))
#
#  visitor_for_testing_depth_first_scan v
#  depth_first_scan(S,v)
#
#  std::string expected_s =
#    '[pre ] 1\n'
#    '[down] 1 -> 2\n'
#    '[pre ] 2\n'
#    '[down] 2 -> 4\n'
#    '[pre ] 4\n'
#    '[post] 4\n'
#    '[up  ] 4 -> 2\n'
#    '[down] 2 -> 7\n'
#    '[pre ] 7\n'
#    '[post] 7\n'
#    '[up  ] 7 -> 2\n'
#    '[down] 2 -> 9\n'
#    '[pre ] 9\n'
#    '[post] 9\n'
#    '[up  ] 9 -> 2\n'
#    '[post] 2\n'
#    '[up  ] 2 -> 1\n'
#    '[down] 1 -> 3\n'
#    '[pre ] 3\n'
#    '[down] 3 -> 8\n'
#    '[pre ] 8\n'
#    '[down] 8 -> 9\n'
#    '[pre ] 9\n'
#    '[post] 9\n'
#    '[up  ] 9 -> 8\n'
#    '[post] 8\n'
#    '[up  ] 8 -> 3\n'
#    '[down] 3 -> 10\n'
#    '[pre ] 10\n'
#    '[post] 10\n'
#    '[up  ] 10 -> 3\n'
#    '[down] 3 -> 11\n'
#    '[pre ] 11\n'
#    '[post] 11\n'
#    '[up  ] 11 -> 3\n'
#    '[post] 3\n'
#    '[up  ] 3 -> 1\n'
#    '[down] 1 -> 9\n'
#    '[pre ] 9\n'
#    '[post] 9\n'
#    '[up  ] 9 -> 1\n'
#    '[post] 1\n'
#
#  CHECK( v.s == expected_s )
#}
#
#
#struct visitor_for_testing_depth_first_find : visitor_for_testing_depth_first_scan {
#  auto
#  pre(auto&& x) -> bool {
#    visitor_for_testing_depth_first_scan::pre(x)
#    return node_value(x) == 3
#  }
#}
#TEST_CASE('depth_first_find') {
#  /* Reminder:
#         1               lvl 3
#      /  |  \
#     |   |    3          lvl 2
#     |   | /  |  \
#     2\_ | 8  |   \      lvl 1
#   /  \ \| |  |    \
#   |  |  \ |  |    \
#  4    7  \9  10   11    lvl 0
#  */
#  auto g = create_rooted_graph_for_tests()
#  auto S = graph_traversal_stack(first_root(g),last_root(g))
#
#  visitor_for_testing_depth_first_find v
#  auto found = depth_first_find(S,v)
#
#  std::string expected_s =
#    '[pre ] 1\n'
#    '[down] 1 -> 2\n'
#    '[pre ] 2\n'
#    '[down] 2 -> 4\n'
#    '[pre ] 4\n'
#    '[post] 4\n'
#    '[up  ] 4 -> 2\n'
#    '[down] 2 -> 7\n'
#    '[pre ] 7\n'
#    '[post] 7\n'
#    '[up  ] 7 -> 2\n'
#    '[down] 2 -> 9\n'
#    '[pre ] 9\n'
#    '[post] 9\n'
#    '[up  ] 9 -> 2\n'
#    '[post] 2\n'
#    '[up  ] 2 -> 1\n'
#    '[down] 1 -> 3\n'
#    '[pre ] 3\n'
#    '[post] 3\n'
#    '[up  ] 3 -> 1\n'
#    '[post] 1\n'
#
#  CHECK( v.s == expected_s )
#  CHECK( node_value(*found) == 3 )
#}
#
#
class visitor_for_testing_depth_first_prune(visitor_for_testing_depth_first_scan):
  def pre(self, x) -> bool:
    visitor_for_testing_depth_first_scan.pre(self,x)
    return node_value(x) == 2

def test_depth_first_prune():
  #  Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0

  g = rooted_tree_example()
  v = visitor_for_testing_depth_first_prune()
  depth_first_prune(g,v)

  expected_s = \
    '[pre ] 1\n' \
    '[down] 1 -> 2\n' \
    '[pre ] 2\n' \
    '[post] 2\n' \
    '[up  ] 2 -> 1\n' \
    '[down] 1 -> 3\n' \
    '[pre ] 3\n' \
    '[down] 3 -> 8\n' \
    '[pre ] 8\n' \
    '[down] 8 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 8\n' \
    '[post] 8\n' \
    '[up  ] 8 -> 3\n' \
    '[down] 3 -> 10\n' \
    '[pre ] 10\n' \
    '[post] 10\n' \
    '[up  ] 10 -> 3\n' \
    '[down] 3 -> 11\n' \
    '[pre ] 11\n' \
    '[post] 11\n' \
    '[up  ] 11 -> 3\n' \
    '[post] 3\n' \
    '[up  ] 3 -> 1\n' \
    '[down] 1 -> 9\n' \
    '[pre ] 9\n' \
    '[post] 9\n' \
    '[up  ] 9 -> 1\n' \
    '[post] 1\n'

  assert v.accumulation_string() == expected_s


class visitor_for_testing_dfs(visitor_for_testing_depth_first_scan):
  def pre(self,x):
    visitor_for_testing_depth_first_scan.pre(self,x)
    if node_value(x) == 8: return step.out
    if node_value(x) == 2: return step.over
    else: return step.into

def test_depth_first_search():
  #  Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0
  #
  g = rooted_tree_example()
  v = visitor_for_testing_dfs()
  depth_first_search(g,v)

  expected_s = \
    '[pre ] 1\n' \
    '[down] 1 -> 2\n' \
    '[pre ] 2\n' \
    '[post] 2\n' \
    '[up  ] 2 -> 1\n' \
    '[down] 1 -> 3\n' \
    '[pre ] 3\n' \
    '[down] 3 -> 8\n' \
    '[pre ] 8\n' \
    '[post] 8\n' \
    '[up  ] 8 -> 3\n' \
    '[post] 3\n' \
    '[up  ] 3 -> 1\n' \
    '[post] 1\n'

  assert v.accumulation_string() == expected_s


class modifying_visitor_for_testing_dfs(visitor_for_testing_depth_first_scan):
  def pre(self, x):
    if node_value(x) == 8: s = step.out
    elif node_value(x) == 2: s = step.over
    else: s = step.into

    set_node_value(x, node_value(x)+100)
    return s

  def post(self, x):
    set_node_value(x, node_value(x)+10000)

def test_depth_first_search_inplace_modif():
  #   Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0

  g = rooted_tree_example()
  v = modifying_visitor_for_testing_dfs()
  depth_first_search(g,v)

  assert g.nodes() == [4,7,10102,9,10108,10,11,10103,10101]
