from enum import Enum
from maia.pytree.graph.algo_interface import dfs_interface_report

class step(Enum):
  """ Information on what to do when a node is visited by a tree traversal algorithm:
    - `step.into`: go down and visit children
    - `step.over`: do not visit children and continue with the next sibling
    - `step.out`: stop the traversal
  """
  into = 0
  over = 1
  out = 2


class graph_traversal_stack:
  """ Main data structure that is used to capture and update the position we are at during a graph traversal.
  """
  def __init__(self, g):
    """ Initialize the ancestor stacks with the roots """
    is_ok, msg  = dfs_interface_report(g)
    assert is_ok, msg

    self._g = g
    self._iterators = []
    self._nodes     = []

    self._push_level(self._g.root_iterator())

  def push_level(self):
    """ Add children of current node to the stack """
    n = self._nodes[-1]
    self._push_level(self._g.child_iterator(n))

  def _push_level(self, siblings):
    sibling_iter = iter(siblings)
    try:
      first_sibling = next(sibling_iter)
    except StopIteration:
      first_sibling = None
    self._iterators.append(sibling_iter)
    self._nodes    .append(first_sibling)

  def pop_level(self):
    self._iterators.pop(-1)
    self._nodes    .pop(-1)

  def advance_node_range(self):
    """ Increment the current node of the stack to be the next sibling """
    try:
      self._nodes[-1] = next(self._iterators[-1])
    except StopIteration:
      self._nodes[-1] = None
  def advance_node_range_to_last(self):
    """ Increment the current node of the stack to be the one-past-the-last sibling """
    self._nodes[-1] = None

  def push_done_level(self):
    """ Add a new level of children and place the current node to one-past-the-last child """
    # Logically equivalent to `self.push_level(); advance_node_range_to_last`,
    # but no need to ask the graph for a potentially non-existent child iterator that will not be used anyways
    self._iterators.append(None)
    self._nodes.append(None)

  def level_is_done(self) -> bool:
    return self._nodes[-1] is None
  def is_at_root_level(self) -> bool:
    return len(self._nodes) == 1
  def is_done(self) -> bool:
    return self.is_at_root_level() and self.level_is_done()

  def nodes(self):
    return self._nodes
  def current_node(self):
    return self._nodes[-1]


def unwind(S, f):
  """ End a graph traversal by going up to the root level
  """
  while not S.is_at_root_level():
    f.post(S.nodes())
    f.up(S.nodes())
    S.pop_level()

  f.post(S.nodes())

def advance_stack(S, f):
  """ Go to the next node in depth-first order
  """
  if not S.level_is_done():
    S.push_level()
    if not S.level_is_done():
      f.down(S.nodes())

def _depth_first_search_stack(S, f):
  """ Depth-first graph traversal

  This is the low-level algorithm
  """
  while not S.is_done():
    if not S.level_is_done():
      next_step = f.pre(S.nodes())
      if next_step == step.out: # stop
        return False
      if next_step == step.over: # prune
        S.push_done_level()
      if next_step is None or next_step == step.into: # go down
        S.push_level()
        if not S.level_is_done():
          f.down(S.nodes())
    else:
      S.pop_level()
      f.post(S.nodes())
      if not S.is_at_root_level():
        f.up(S.nodes())
        S.advance_node_range()
        if not S.level_is_done():
          f.down(S.nodes())
      else:
        S.advance_node_range()

  return True

class complete_visitor:
  """ The `_depth_first_search_stack` algorithm expects a visitor with `pre`, `post`, `down` and `up`.

  If the user does not provide `post`, `down` or `up`, then add them on-the-fly to do nothing.
  """

  def __init__(self, v):
    def _do_nothing(*args): pass

    # take v.pre, v.post, v.down and v.up if they exist, otherwise, create them to do nothing
    for f_name in ['pre','post','down','up']:
      setattr(self, f_name, getattr(v, f_name, _do_nothing))


class close_ancestor_visitor:
  """ The `_depth_first_search_stack` algorithm calls its visitor by passing it
  the complete list of ancestors of the current node.

  However, most ot the times, the visitor only cares about
  the current node (or just the first few ancestors) as input.

  This adaptor class turns such a visitor into a visitor acceptable by `_depth_first_search_stack`
  and delegates all calls with the list of ancestors to calls with only the first `depth` ancestors:
    depth = 1: only the current node
    depth = 2: the current node + its parent
    ...

  """
  def __init__(self, visitor, depth):
    self.f = visitor
    assert depth >= 1
    self.depth = depth

  def _ancestors_list(self, ancestors):
    res = ancestors[-self.depth:]
    sz = len(res)
    if sz < self.depth:
      diff = self.depth-sz
      padding = [None] * diff
      res = padding + res
    return res

  def pre(self, ancestors):
    return self.f.pre ( *self._ancestors_list(ancestors) )
  def post(self, ancestors):
    return self.f.post( *self._ancestors_list(ancestors) )

  # Note that if we wanted to be more general,
  # we could have a second `depth` parameter for `down` and `up`
  def down(self, ancestors):
    return self.f.down( ancestors[-2], ancestors[-1])
  def up(self, ancestors):
    return self.f.up  ( ancestors[-1], ancestors[-2])


def adapt_visitor(f, depth='node'):
  f = complete_visitor(f)
  if depth != 'all':
    if depth == 'node'  : depth = 1
    if depth == 'parent': depth = 2
    f = close_ancestor_visitor(f,depth)
  return f


def depth_first_search_stack(S, f, depth='node'):
  return _depth_first_search_stack(S, f)


def depth_first_search(g, f, depth='node'):
  """
  Depth-first graph traversal

  Args:
    g: Graph object that should conform to the depth-first search interface. See :func:`dfs_interface_report` for full documentation.
    f : A visitor object that has a `pre` method, and optionally `post`, `up` and `down` methods.
    depth ('node','parent','all' or integer): Control the arguments that are passed to the visitor methods.
      - If `depth==1` or `depth='node', passes the current node as the only argument to `pre` and `post`
      - If `depth==2` or `depth='parent', passes the parent and the current nodes as the two arguments to `pre` and `post`
      - ...
      - If `depth=='all', passes the list of all the ancestors as the only argument to `pre` and `post`

  - `pre` is called on a node when it is traversed for the first time (going from parents to children).
  - `pre` can return a `step` to tell the algorithm to step over the node or to stop. By default will continue the search. See :class:`step` for more info.
  - `post` works as `pre` but is called once all the children of the node have been visited.
  - `up` and `down` are called once the algorithm is moving from a parent to its child or inversely.
  - `down` takes the parent then the child as its arguments.
  - `up` takes the child then the parent as its arguments.
  """
  f = adapt_visitor(f, depth)

  S = graph_traversal_stack(g)

  done = _depth_first_search_stack(S, f)

  if not done:
    matching_node = S.current_node()
    unwind(S, f)
    return matching_node
