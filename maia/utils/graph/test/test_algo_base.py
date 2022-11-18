from maia.utils.graph.algo_base import step, depth_first_search


def create_io_graph_for_tests():
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0
  return [
    [ 4, [2]    , []     ], #0
    [ 7, [2]    , []     ], #1
    [ 2, [8]    , [0,1,3]], #2
    [ 9, [4,8,2], []     ], #3
    [ 8, [7]    , [3]    ], #4
    [10, [7]    , []     ], #5
    [11, [7]    , []     ], #6
    [ 3, [8]    , [4,5,6]], #7
    [ 1, []     , [2,7,3]], #8
  ]

def node_value(x):
  return x[0]
def set_node_value(x, val):
  x[0] = val
def inward_nodes(x):
  return x[1]
def outward_nodes(x):
  return x[2]

def create_rooted_graph_for_tests():
  return create_io_graph_for_tests(), 8 # just give the info that 8 is the root

class my_test_graph_adaptor:
  @staticmethod
  def root_nodes(g):
    return [ g[1] ]

  @staticmethod
  def children(n):
    return outward_nodes(n)

  @staticmethod
  def first_child(g,n):
    return g[0] [ outward_nodes(n)[0] ]

  @staticmethod
  def nodes(g):
    return [n for n,_,_ in g[0]]


#def create_rooted_graph_for_tests():
#  return
#    (1,[
#       (2,[
#          (4,[]),
#          (7,[]),
#        ]),
#       (3,[
#          (8,[
#            (9,[]),
#           ]),
#          (10,[]),
#          (11,[]),
#        ]),
#     ])
#}

class visitor_for_testing_depth_first_scan:
  def __init__(self):
    self.s = ''

  def pre(self, x):
    self.s += '[pre ] ' + str(node_value(x)) + '\n';
  def post(self, x):
    self.s += '[post] ' + str(node_value(x)) + '\n';
  def up(self, below, above):
    self.s += '[up  ] ' + str(node_value(below)) + ' -> ' + str(node_value(above)) + '\n';
  def down(self, above, below):
    self.s += '[down] ' + str(node_value(above)) + ' -> ' + str(node_value(below)) + '\n';

  def accumulation_string(self):
    return self.s

#TEST_CASE('depth_first_scan_adjacency_stack') {
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
#  auto g = create_rooted_graph_for_tests();
#  auto S = graph_traversal_stack(first_root(g),last_root(g));
#
#  visitor_for_testing_depth_first_scan v;
#  depth_first_scan_adjacency_stack(S,v);
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
#    '[post] 1\n';
#
#  CHECK( v.s == expected_s );
#}
#
#
#struct visitor_for_testing_depth_first_find : visitor_for_testing_depth_first_scan {
#  auto
#  pre(auto&& x) -> bool {
#    visitor_for_testing_depth_first_scan::pre(x);
#    return node(x) == 3;
#  }
#};
#TEST_CASE('depth_first_find_adjacency_stack') {
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
#  auto g = create_rooted_graph_for_tests();
#  auto S = graph_traversal_stack(first_root(g),last_root(g));
#
#  visitor_for_testing_depth_first_find v;
#  auto found = depth_first_find_adjacency_stack(S,v);
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
#    '[post] 1\n';
#
#  CHECK( v.s == expected_s );
#  CHECK( node(*found) == 3 );
#}
#
#
#struct visitor_for_testing_depth_first_prune : visitor_for_testing_depth_first_scan {
#  auto
#  pre(auto&& x) -> bool {
#    visitor_for_testing_depth_first_scan::pre(x);
#    return node(x) == 2;
#  }
#};
#TEST_CASE('depth_first_prune_adjacency_stack') {
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
#  auto g = create_rooted_graph_for_tests();
#  auto S = graph_traversal_stack(first_root(g),last_root(g));
#
#  visitor_for_testing_depth_first_prune v;
#  depth_first_prune_adjacency_stack(S,v);
#
#  std::string expected_s =
#    '[pre ] 1\n'
#    '[down] 1 -> 2\n'
#    '[pre ] 2\n'
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
#    '[post] 1\n';
#
#  CHECK( v.s == expected_s );
#}
#
#
class visitor_for_testing_dfs(visitor_for_testing_depth_first_scan):
  def pre(self,x):
    visitor_for_testing_depth_first_scan.pre(self,x)
    if node_value(x) == 8: return step.out
    if node_value(x) == 2: return step.over
    else: return step.into

def test_depth_first_search_adjacency_stack():
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
  g = create_rooted_graph_for_tests()

  v = visitor_for_testing_dfs()
  depth_first_search(g,v,my_test_graph_adaptor);

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

def test_depth_first_search_adjacency_stack_inplace_modif():
  #   Reminder:
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0

  g = create_rooted_graph_for_tests()

  v = modifying_visitor_for_testing_dfs()
  depth_first_search(g,v,my_test_graph_adaptor)

  assert my_test_graph_adaptor.nodes(g) == [4,7,10102,9,10108,10,11,10103,10101]
