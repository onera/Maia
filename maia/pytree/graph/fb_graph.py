"""
  'fb' means 'forward/backward'
  An 'fb_graph' is a graph stored by an adjacency list
  Each adjacency is a 3-list of:
    - a node value
    - a list of forward nodes, i.e. indices of nodes pointed to by the adjacency
    - a list of backward nodes, i.e. indices of nodes which point to the adjacency

  Notes:
    - Backward nodes do not bring new information, the complete structure of the graph is already encoded in forward nodes
        They are supposed to be used in algorithm that need to go backward.
        If you don't need to go backward, just use a f_graph (forward(-only) graph)
        Forward and backward nodes are supposed to be coherent,
          but beware that this is by convention of what 'fb_graph' is supposed to mean, nothing is actually enforced
    - The graph can be used to represent directed or undirected graphs
        but it has limited interest if the graph is undirected,
          In this case, for each node, forward nodes == backward nodes
"""

def fb_graph_example():
  #         1               lvl 3
  #      /  |  \
  #     |   |    3          lvl 2
  #     |   | /  |  \
  #     2\_ | 8  |   \      lvl 1
  #   /  \ \| |  |    \
  #   |  |  \ |  |    \
  #  4    7  \9  10   11    lvl 0
  return [
    [ 4 , []      , [2]     ], #0
    [ 7 , []      , [2]     ], #1
    [ 2 , [0,1,3] , [8]     ], #2
    [ 9 , []      , [4,8,2] ], #3
    [ 8 , [3]     , [7]     ], #4  The node at position '4' has value '8'.
                               #   Its parents are at positions [7], that is, it has only one parent, its value is '3'
                               #   Its children are at positions [3], that is, it has only one child, its value is '9'
    [10 , []      , [7]     ], #5
    [11 , []      , [7]     ], #6
    [ 3 , [4,5,6] , [8]     ], #7
    [ 1 , [2,3,7] , []      ], #8
  ]


# We can import from `f_graph` because the position of VALUE and FORWARD are the same on a `f_graph` and an `fb_graph`
# From a depth-first search point-of-view, `f_graph` and `fb_graph` are the same
from .f_graph import VALUE, FORWARD, tree_adaptor, _adjacency_iterator
BACKWARD = 2

class backward_tree_adaptor(tree_adaptor):
  """
    An fb_graph is adapted to a tree by defining its children through its backward nodes.
  """
  def child_iterator(self, n) -> _adjacency_iterator:
    return _adjacency_iterator(self.g, n[BACKWARD])


def rooted_fb_graph_example():
  g = fb_graph_example()
  return tree_adaptor(g,[8]) # note: 8 is the index of node '1'


# dfs build {
## building an fb_tree (which is just an fb_graph that has the property of being a tree) from depth-first search
class fb_tree_ctor:
  def __init__(self):
    self.i = 0

  def __call__(self, from_fb_node, to_sub_fb_graphs):
    node_forwards = []
    cat_sub_fb_graphs = []
    for to_sub_fb_graph in to_sub_fb_graphs:
      # now we know the index of the parent
      direct_child = to_sub_fb_graph[-1]
      direct_child[BACKWARD] = [self.i]

      # we need to offset forward indices due to previous sub-graphs already in the adjacency list
      offset = len(cat_sub_fb_graphs)
      for adj in to_sub_fb_graph:
        for i in range(len(adj[FORWARD])):
          adj[FORWARD][i] += offset

      # append the sub-graph to list
      cat_sub_fb_graphs += to_sub_fb_graph

      # keep track of the index of this direct child for the current node
      direct_child_index = len(cat_sub_fb_graphs)-1
      node_forwards.append(direct_child_index)

    to_node = [from_fb_node[VALUE], node_forwards, []]
    self.i += 1
    return cat_sub_fb_graphs + [to_node]


def depth_first_build_fb_tree(g):
  from maia.pytree.graph.build import depth_first_build
  return depth_first_build(g, fb_tree_ctor())
# dfs build }
