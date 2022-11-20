"""
  'io' means 'inward/outward'
  An 'io_graph' is a directed graph stored by an adjacency list
  Each adjacency is a 3-tuple of:
    - a node value
    - a sequence of indices of nodes which point to the adjacency (i.e. inward nodes)
    - a sequence of indices of nodes which pointed to by the adjacency (i.e. outward nodes)
"""

def io_graph_data_example():
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

def node_value(n):
  return n[0]
def set_node_value(n, val):
  n[0] = val
def inward_nodes(n):
  return n[1]
def outward_nodes(n):
  return n[2]


class io_graph_tree_adaptor:
  """
    An io_graph is adapted to a tree by going through its outward nodes
  """
  def __init__(self, g, root_idx):
    self.g = g
    self.root_idx = root_idx
  def nodes(self):
    return [n for n,_,_ in self.g]

# Interface to satisfy dfs_interface_report {
  def first_child(self, n):
    return self.g[ outward_nodes(n)[0] ]

# Below: range methods
# The idea is that the DFS algorithm needs to save `roots()` and `children()` ranges
#     and iterate through them at various points.
# The algorithm does not have to know anything about the objects,
#     it just saves them and then gives them back to `current_node`, `range_is_done`...
#     so that these methods can decode them

# range-returning methods {
#    for this test, we choose to represent a range of nodes by a pair of:
#        1. the underlying sequence
#        2. the current position on the sequence
  @staticmethod
  def children(n):
    return [outward_nodes(n), 0]
  def roots(self):
    return [[self.root_idx], 0]
# }

# range-input methods {
#    here, we get ranges that we produced with `roots` and `children`
#    and extract the relevant information from them
  def current_node(self, ns):
    nodes,pos = ns
    return self.g[nodes[pos]]

  @staticmethod
  def range_is_done(ns) -> bool:
    nodes,pos = ns
    return len(nodes) == pos

  @staticmethod
  def advance_node_range(ns):
    ns[1] += 1

  @staticmethod
  def advance_node_range_to_last(ns):
    ns[1] = len(ns[0])
# }
# Interface to satisfy dfs_interface_report }


def rooted_tree_example():
  g = io_graph_data_example()
  return io_graph_tree_adaptor(g,8) # note: 8 is the index of node '1'
