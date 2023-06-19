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
    [ 8, [7]    , [3]    ], #4  The node at position '4' has value '8'.
                            #   Its parents are at positions [7], that is, it has only one parent, its value is '3'
                            #   Its children are at positions [3], that is, it has only one child, its value is '9'
    [10, [7]    , []     ], #5
    [11, [7]    , []     ], #6
    [ 3, [8]    , [4,5,6]], #7
    [ 1, []     , [2,7,3]], #8
  ]


VALUE = 0
INWARD = 1
OUTWARD = 2

class adjacency_iterator:
  def __init__(self, g, adj_idcs):
    self.g = g
    self.adj_idcs = adj_idcs

  def __iter__(self):
    self.idx = 0
    return self

  def __next__(self):
    if self.idx < len(self.adj_idcs):
      x = self.adj_idcs[self.idx]
      self.idx += 1
      return self.g[x]
    else:
      raise StopIteration()

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
  def children(self, n):
    return adjacency_iterator(self.g, n[OUTWARD])
  def roots(self):
    return adjacency_iterator(self.g, [self.root_idx])
# Interface to satisfy dfs_interface_report }


def rooted_tree_example():
  g = io_graph_data_example()
  return io_graph_tree_adaptor(g,8) # note: 8 is the index of node '1'


# dfs build {
## building a io_tree (which is just an io_graph that is a tree) from depth-first search
class io_tree_ctor:
  def __init__(self):
    self.i = 0

  def __call__(self, from_io_node, to_sub_io_graphs):
    node_outwards = []
    cat_sub_io_graphs = []
    for to_sub_io_graph in to_sub_io_graphs:
      # now we know the index of the parent
      direct_child = to_sub_io_graph[-1]
      direct_child[INWARD] = [self.i]

      # we need to offset outward indices due to previous sub-graphs already in the adjacency list
      offset = len(cat_sub_io_graphs)
      for adj in to_sub_io_graph:
        for i in range(len(adj[OUTWARD])):
          adj[OUTWARD][i] += offset

      # append the sub-graph to list
      cat_sub_io_graphs += to_sub_io_graph

      # keep track of the index of this direct child for the current node
      direct_child_index = len(cat_sub_io_graphs)-1
      node_outwards.append(direct_child_index)

    to_node = [from_io_node[VALUE], [], node_outwards]
    self.i += 1
    return cat_sub_io_graphs + [to_node]


def depth_first_build_io_tree(g):
  from maia.pytree.graph.build import depth_first_build
  return depth_first_build(g, io_tree_ctor())
# dfs build }
