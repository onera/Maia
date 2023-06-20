from maia.pytree.graph.algo import dfs_interface_report, depth_first_search
from maia.pytree.graph.fb_graph import rooted_fb_graph_example, VALUE, depth_first_build_fb_tree


def test_fb_graph():
  #   Reminder:
  #         1
  #      /  |  \
  #     |   |    3
  #     |   | /  |  \
  #     2\_ | 8  |   \
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11
  g = rooted_fb_graph_example()

  roots = list(g.roots())
  assert len(roots) == 1
  assert roots[0][VALUE] == 1

  cs = list(g.children(roots[0]))
  assert len(cs) == 3
  assert cs[0][VALUE] == 2
  assert cs[1][VALUE] == 9
  assert cs[2][VALUE] == 3


def test_fb_graph_tree_adaptor_is_depth_first_searchable():
  t = rooted_fb_graph_example()
  assert dfs_interface_report(t) == (True,'')


class visitor_for_testing_depth_first_scan:
  def __init__(self):
    self.s = ''
  def pre(self, x):
    self.s += '[pre] ' + str(x[VALUE]) + '\n'
  def accumulation_string(self):
    return self.s

def test_depth_first_scan():
  #   Reminder:
  #         1
  #      /  |  \
  #     |   |    3
  #     |   | /  |  \
  #     2\_ | 8  |   \
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11

  g = rooted_fb_graph_example()
  v = visitor_for_testing_depth_first_scan()
  depth_first_search(g,v)

  expected_s = \
    '[pre] 1\n' \
    '[pre] 2\n' \
    '[pre] 4\n' \
    '[pre] 7\n' \
    '[pre] 9\n' \
    '[pre] 9\n' \
    '[pre] 3\n' \
    '[pre] 8\n' \
    '[pre] 9\n' \
    '[pre] 10\n' \
    '[pre] 11\n'

  assert v.accumulation_string() == expected_s


def test_depth_first_build_fb_tree():
  #   Reminder:
  #         1
  #      /  |  \
  #     |   |    3
  #     |   | /  |  \
  #     2\_ | 8  |   \
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11

  g = rooted_fb_graph_example()

  fb_tree = depth_first_build_fb_tree(g)

  expected_fb_tree = [
    [4 , []       , [3 ]], # 0
    [7 , []       , [3 ]], # 1
    [9 , []       , [3 ]], # 2
    [2 , [0, 1, 2], [10]], # 3
    [9 , []       , [10]], # 4
    [9 , []       , [6 ]], # 5
    [8 , [5]      , [9 ]], # 6
    [10, []       , [9 ]], # 7
    [11, []       , [9 ]], # 8
    [3 , [6, 7, 8], [10]], # 9
    [1 , [3, 4, 9], []  ], # 10
  ]
  # expected_fb_tree is:
  #             1
  #          /  |  \
  #         /   |   \
  #        /    9    \
  #       /          3
  #      /         / | \
  #     2         8  |  \
  #   / | \       |  |   \
  # 4   7   9     9  10   11

  assert fb_tree == expected_fb_tree
