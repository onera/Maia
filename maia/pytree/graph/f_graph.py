"""
  'f' means 'forward'
  An 'f_graph' is a graph stored by an adjacency list.
  Each adjacency is a pair of:
    - a node value
    - a list of forward nodes, i.e. indices of nodes pointed to by the adjacency

  Note:
    - The graph can be used to represent directed or undirected graphs
        - If the graph is undirected,
            if `X` points to `Y` (i.e. the index of Y is in the list of forward nodes of X)
            the `Y` must point to `X`
          Beware that this is by convention of what an undirected 'f_graph' is supposed to mean, nothing is actually enforced
        - If the graph is directed, then forward nodes are used to represent oriented edges
            note that if `X` points to `Y`, there is no quick way to determine that `Y` is pointed to by `X`
            if you need this information, consider either using a `fb_graph` or a `fe_graph`
"""

def f_graph_example():
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0
  return [
    [ 4 , []      ], #0
    [ 7 , []      ], #1
    [ 2 , [0,1,3] ], #2
    [ 9 , []      ], #3
    [ 8 , [3]     ], #4  The node at position '4' has value '8'.
                     #   Its parents are at positions [7], that is, it has only one parent, its value is '3'
                     #   Its children are at positions [3], that is, it has only one child, its value is '9'
    [10 , []      ], #5
    [11 , []      ], #6
    [ 3 , [4,5,6] ], #7
    [ 1 , [2,3,7] ], #8
  ]

VALUE = 0
FORWARD = 1

class _adjacency_iterator:
  """
    Used to iterate over the children of a node of a `f_graph`
  """
  def __init__(self, g, adj_idcs):
    """ 
      created by storing the `f_graph` `g` and a list of children of a node within `f`
    """
    self.g = g
    self.adj_idcs = adj_idcs

  def __iter__(self):
    self.idx = 0 # position of the current child in `self.adj_idcs`
    return self

  def __next__(self):
    if self.idx < len(self.adj_idcs):
      node_idx = self.adj_idcs[self.idx] # the index of the current child
      self.idx += 1 # the next position of the child in `self.adj_idcs` should be incremented
      return self.g[node_idx] # we return the value of the current child
    else:
      raise StopIteration()

class tree_adaptor:
  """
    An f_graph is adapted to a tree by going through its forward nodes.
  """
  def __init__(self, g, root_idx):
    self.g = g
    self.root_idx = root_idx
  def nodes(self):
    return [adj[VALUE] for adj in self.g]

# Interface to satisfy dfs_interface_report {
  def children(self, n):
    return _adjacency_iterator(self.g, n[FORWARD])
  def roots(self):
    return _adjacency_iterator(self.g, [self.root_idx])
# Interface to satisfy dfs_interface_report }


def rooted_f_graph_example():
  g = f_graph_example()
  return tree_adaptor(g,8) # note: 8 is the index of node '1'
