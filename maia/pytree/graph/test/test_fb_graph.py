from maia.pytree.graph.algo import dfs_interface_report, depth_first_search
from maia.pytree.graph.fb_graph import fb_graph_example, rooted_fb_graph_example, VALUE, BACKWARD, backward_tree_adaptor, depth_first_build_fb_tree
from maia.pytree.algo_utils import find


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


class visitor_to_test_depth_first_scan:
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
  v = visitor_to_test_depth_first_scan()
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




def get_path(g, node_idx):
  path = ''
  current_node_idx = node_idx
  while current_node_idx is not None:
    path = str(g[current_node_idx][VALUE]) + '/' + path
    parents = g[current_node_idx][BACKWARD]
    if len(parents) >= 1:
      current_node_idx = parents[0]
    else:
      current_node_idx = None
  return '/'+path


def test_get_path():
  g = fb_graph_example()
  node_4_idx = find(g, lambda node: node[VALUE] == 4)
  assert get_path(g, node_4_idx) == '/1/2/4/'

  node_1_idx = find(g, lambda node: node[VALUE] == 1)
  assert get_path(g, node_1_idx) == '/1/'

  node_9_idx = find(g, lambda node: node[VALUE] == 9)
  assert get_path(g, node_9_idx) == '/1/3/8/9/'




class _get_path_visitor:
  def __init__(self):
    self.paths = []

  def pre(self, ancestors):
    current_node = ancestors[-1]
    parents_of_current = current_node[BACKWARD]
    is_leaf = len(parents_of_current) == 0
    if is_leaf:
      self.paths.append('/'+'/'.join([str(a[VALUE]) for a in ancestors[::-1]])+'/')

def get_paths(g, node_idx):
  inversed_g = backward_tree_adaptor(g,node_idx)
  v = _get_path_visitor()
  depth_first_search(inversed_g, v, depth='all')
  return v.paths

def test_get_paths():
  g = fb_graph_example()
  node_4_idx = find(g, lambda node: node[VALUE] == 4)
  assert get_paths(g, node_4_idx) == ['/1/2/4/']

  node_1_idx = find(g, lambda node: node[VALUE] == 1)
  assert get_paths(g, node_1_idx) == ['/1/']

  node_9_idx = find(g, lambda node: node[VALUE] == 9)
  assert get_paths(g, node_9_idx) == ['/1/3/8/9/', '/1/9/', '/1/2/9/']

