from maia.pytree.graph import algo


class _build_tree_visitor:
  def __init__(self, node_constructor, pre):
    self.node_constructor = node_constructor
    self.sub_tree_stack = [[]]
    self.pre = pre

  def pre(self, node):
    self.sub_tree_stack.append([]) # add a level
    if self.pre is not None:
      return self.pre(node)

  def post(self, from_node):
    to_children = self.sub_tree_stack[-1] # get children from stack
    to_node = self.node_constructor(from_node, to_children)

    self.sub_tree_stack.pop(-1) # we are done with the children
    self.sub_tree_stack[-1].append(to_node) # one level up, we can now deposit the sub_tree that we just built

  def retrieve_composition_term(self): # supposed to be called at the end
    assert len(self.sub_tree_stack)==1 # there should be only one level left since we finished the search
    assert len(self.sub_tree_stack[0])==1 # and there should be only one node: the tree that was built during the search
    return self.sub_tree_stack[0][0]


def depth_first_build(g, node_constructor, pre):
  """
    Depth first traversal of graph `g` in order to create another graph `g_out`

  Args:
    g: Graph object that should conform to the depth-first search interface. See :func:`dfs_interface_report` for full documentation.

    node_constructor: tell the algorithm how to construct a node from its children and the current traversed value in the original graph `g`
        More precisely, `node_constructor` is a function with two arguments `from_node` and `to_children` and returning `to_node`
        `from_node` is the current node that the depth-first algorithm is traversing
        `to_children` is a list of the previous sub-graphs of `g_out`
        `to_node` is the new graph node that is to be created. It should have the same type as the elements of `to_children`

    pre: tells the algorithm what to do on each node traversed (step.in, step,
        More precisely, `pre` is a function of argument `node` and returning a `graph.algo.step` value
  """
  v = _build_tree_visitor(node_constructor, pre=None)
  algo.depth_first_search(g, v)
  return v.retrieve_composition_term()
