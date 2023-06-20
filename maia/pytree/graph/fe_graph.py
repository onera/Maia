"""
  'fe' means 'forward + edge'
  An 'fe_graph' is a graph stored by an adjacency list
  Each adjacency is a 3-list of:
    - a node value
    - a list of forward nodes, i.e. indices of nodes pointed to by the adjacency
    - a list of edge values

  Note:
    - For each adjacency, the list of forward nodes and the list of edge values must have the same size
        Beware that while this assumption must hold, nothing is actually enforced
    - The graph can be used to represent directed or undirected graphs
        - If the graph is undirected,
            if `X` points to `Y` (i.e. the index of Y is in the list of forward nodes of X)
            the `Y` must point to `X`
          Beware that this is by convention of what an undirected 'fe_graph' is supposed to mean, nothing is actually enforced
        - If the graph is directed, then forward nodes are used to represent oriented edges
            note that if `X` points to `Y`, there is no quick way to determine that `Y` is pointed to by `X`
            if you need this information, you have two options:
              - consider either using a `fb_graph`
              - make your `fe_graph` adjacencies symmetric has with an unstructured graph, an encode the edge direction in the edge value
"""

def fe_graph_example():
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
  #    /    |    \ | /    |    \
  #   /     |     \|/H    |     \
  #  /      |      |      |      \
  # 4       7      9      10     11
  return [
    [ 4, []     , []            ], #0
    [ 7, []     , []            ], #1
    [ 2, [0,1,3], ['B','C','D'] ], #2
    [ 9, []     , []            ], #3
    [ 8, [3]    , ['J']         ], #4  The node at index '4' has value '8'.
                                   #   Its children are at indices [3], that is, it has only one child, of value '9'
                                   #   The edge from index 4 to index 3 has value 'J'
    [10, []     , []            ], #5
    [11, []     , []            ], #6
    [ 3, [4,5,6], ['G','H','I'] ], #7
    [ 1, [2,3,7], ['A','E','F'] ], #8
  ]

VALUE = 0
FORWARD = 1
EDGE_VALUES = 2


class node_with_edge:
  def __init__(self, adj, edge):
    self.adj = adj
    self.edge = edge

  @property
  def node(self):
    return self.adj[VALUE]


class _adjacency_iterator:
  def __init__(self, g, adj_idcs, edge_values):
    self.g = g
    self.adj_idcs = adj_idcs
    self.edge_values = edge_values

  def __iter__(self):
    self.idx = 0
    return self

  def __next__(self):
    if self.idx < len(self.adj_idcs):
      node_idx = self.adj_idcs   [self.idx]
      edge     = self.edge_values[self.idx]
      self.idx += 1
      return node_with_edge(self.g[node_idx], edge)
    else:
      raise StopIteration()


class tree_adaptor:
  """
    An f_graph is adapted to a tree by going through its forward nodes
  """
  def __init__(self, g, root_idx):
    self.g = g
    self.root_idx = root_idx
  def nodes(self):
    return [adj[VALUE] for adj in self.g]

# Interface to satisfy dfs_interface_report {
  def children(self, nwe):
    n = nwe.adj
    return _adjacency_iterator(self.g, n[FORWARD], n[EDGE_VALUES])
  def roots(self):
    return _adjacency_iterator(self.g, [self.root_idx], [None])
# Interface to satisfy dfs_interface_report }


def rooted_fe_graph_example():
  g = fe_graph_example()
  return tree_adaptor(g,8) # note: 8 is the index of node '1'
