from maia.pytree.graph.algo import dfs_interface_report
from maia.pytree.graph.fb_graph import rooted_fb_graph_example, VALUE


def test_f_graph():
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

  roots = list(g.root_iterator())
  assert len(roots) == 1
  assert roots[0][VALUE] == 1

  cs = list(g.child_iterator(roots[0]))
  assert len(cs) == 3
  assert cs[0][VALUE] == 2
  assert cs[1][VALUE] == 9
  assert cs[2][VALUE] == 3


def test_fb_graph_tree_adaptor_is_depth_first_searchable():
  t = rooted_fb_graph_example()
  assert dfs_interface_report(t) == (True,'')
