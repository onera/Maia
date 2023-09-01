from maia.pytree.graph import algo


class _build_tree_visitor:
  """ Visitor used by `depth_first_build`.

  In order to build a tree `t1` from another `t0`, 
    when constructing a node of `t1`, we need to have its children already constructed,
    which means that the construction is done in a post-order fashion.

  The idea it to keep a stack of partial sub-trees of `t1`:
    - when the node is leaf, we can construct the sub-tree with no child,
    - when not a leaf, when traversing the node post-order, we now have the children in the stack:
        we can pop them from the stack and give them to the node's constructor.
  """
  def __init__(self, node_constructor, pre):
    self._node_constructor = node_constructor
    self._sub_tree_stack = [[]]
    self._pre = pre

  def pre(self, node):
    self._sub_tree_stack.append([]) # add a level
    if self._pre is not None:
      return self._pre(node)

  def post(self, from_node):
    to_children = self._sub_tree_stack[-1] # get children from stack
    to_node = self._node_constructor(from_node, to_children)

    self._sub_tree_stack.pop(-1) # we are done with the children
    self._sub_tree_stack[-1].append(to_node) # one level up, we can now deposit the sub_tree that we just built

  def retrieve_composition_term(self): # supposed to be called at the end
    assert len(self._sub_tree_stack)==1 # there should be only one level left since we finished the search
    return self._sub_tree_stack[0]


def depth_first_build_trees(g, node_constructor, pre=None):
  v = _build_tree_visitor(node_constructor, pre)
  algo.depth_first_search(g, v)
  return v.retrieve_composition_term()

def depth_first_build(g, node_constructor, pre=None):
  """
    Depth first traversal of graph `g` in order to create another graph `g_out`.

  Args:
    g: Graph object that should conform to the depth-first search interface. See :func:`dfs_interface_report` for full documentation.

    node_constructor: tell the algorithm how to construct a node from its children and the current traversed value in the original graph `g`
        More precisely, `node_constructor` is a function with two arguments `from_node` and `to_children` and returning `to_node`
        `from_node` is the current node that the depth-first algorithm is traversing
        `to_children` is a list of the previous sub-graphs of `g_out`
        `to_node` is the new graph node that is to be created.

    pre: tells the algorithm what to do on each node traversed (step.into, step.over, step.out)
        More precisely, `pre` is a function of argument `node` and returning a `graph.algo.step` value.
  """
  ts = depth_first_build_trees(g, node_constructor, pre)
  assert len(ts) == 1 # precondition: len(g.root_iterator()) == 1
  return ts[0]
