from enum import Enum


def dfs_interface_report(g):
  """ Tells if `g` conforms to the depth-first search interface, and if not, why

  To be conforming, `g` has to have:
    - a `roots(self)` method that returns the roots of the graph.
    - a `children(self, n)` method that returns the children of node `n` in the graph.
  Both methods should return object that are iterators over nodes of the graph
  """
  report = ''
  is_ok = True

  # check has `roots` and `children`
  expected_attrs = ['roots', 'children']
  for attr in expected_attrs:
    if not getattr(g, attr, None):
      is_ok = False
      report += f'Attribute {attr} is missing'

  # check `roots` and `children` returns iterators
  if is_ok:
    roots_iter = g.roots()
    expected_attrs = ['__iter__', '__next__']
    for attr in expected_attrs:
      if not getattr(roots_iter, attr, None):
        is_ok = False
        report += f'Iterator attribute {attr} is missing'

  # prefix report if not empty
  if not is_ok:
    report = f'dfs_interface_report of type {type(g)}:\n'  + report

  return is_ok, report


class step(Enum):
  """ Information on what to do when a node is visited by a tree traversal algorithm:
    - `step.into`: go down and visit children
    - `step.over`: do not visit children and continue with the next sibling
    - `step.out`: stop the traversal
  """
  into = 0
  over = 1
  out = 2


def make_visitor(v):
  """ The `depth_first_search` algorithm expects a visitor with `pre`, `post`, `down` and `up`

  If the user does not provide `post`, `down` or `up`, then add them on-the-fly to do nothing
  """
  def do_nothing(*args): pass

  if not getattr(v, 'post', None):
    v.post = do_nothing
  if not getattr(v, 'down', None):
    v.down = do_nothing
  if not getattr(v, 'up', None):
    v.up = do_nothing
  return v


class graph_traversal_stack:
  """ Main data structure that is used to capture and update the position we are in during a graph traversal
  """
  def __init__(self, g):
    self._g = g
    self._iterators = []
    self._nodes     = []

    self._push_level(self._g.roots())

  def push_level(self):
    n = self._nodes[-1]
    self._push_level(self._g.children(n))

  def _push_level(self, siblings):
    sibling_iter = iter(siblings)
    try:
      first_sibling = next(sibling_iter)
    except StopIteration:
      first_sibling = None
    self._iterators += [sibling_iter]
    self._nodes     += [first_sibling]

  def pop_level(self):
    self._iterators.pop(-1)
    self._nodes    .pop(-1)

  def advance_node_range(self):
    try:
      self._nodes[-1] = next(self._iterators[-1])
    except StopIteration:
      self._nodes[-1] = None
  def advance_node_range_to_last(self):
    self._nodes[-1] = None

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


def depth_first_search_stack(S, f):
  """ Depth-first graph traversal

  This is the low-level algorithm that is called by `depth_first_search`
  """
  while not S.is_done():
    if not S.level_is_done():
      next_step = f.pre(S.nodes())
      if next_step == step.out: # stop
        matching_node = S.current_node()
        unwind(S, f)
        return matching_node
      if next_step == step.over: # prune
        S.push_level()
        S.advance_node_range_to_last()
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

  return None


class close_ancestor_visitor:
  """ The `depth_first_search_stack` algorithm calls its visitor by passing it 
  the complete list of ancestors of the current node

  However, most ot the times, the visitor only cares about 
  the current node (or just the first few ancestors) as input
  
  This adaptor class turns such a visitor into a visitor acceptable by `depth_first_search_stack`
  and delegates all calls with the list of ancestors to calls with only the first `depth` ancestors
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

  # For `down` and `up`, we don't have enough examples to really make the correct decision for the interface
  # Here we take:
  #   the current node (i.e. ancestors[-1]) for the `below` arguement
  #   the parent node (i.e. ancestors[-2]) for the `above` arguement
  # Which at least seems natural for depth==1
  def down(self, ancestors):
    return self.f.down( ancestors[-2], ancestors[-1] )
  def up(self, ancestors):
    return self.f.up  ( ancestors[-1], ancestors[-2] )


def depth_first_search(g, f, depth='node'):
  """
  Depth-first graph traversal
    
  Args:
    g: Graph object that should conform to the depth-first search interface. See :func:`dfs_interface_report` for full documentation.
    f : A visitor object that has a `pre` method, and optionally `post`, `up` and `down` methods
    depth ('node','parent','all' or integer): Control the arguments that are passed to the visitor methods.
      - If `depth==1` or `depth='node', passes the current node as the only argument to `pre` and `post`
      - If `depth==2` or `depth='parent', passes the parent and the current nodes as the two arguments to `pre` and `post`
      - ...
      - If `depth=='all', passes the list of all the ancestors as the only argument to `pre` and `post`

  - if `only_nodes` is `True` then `pre` will be given the current node of the graph as argument
    else it will be given all the ancestors up to the current node as arguments
  - `pre` is called on a node as it is found for the first time
  - `pre` can return a `step` to tell the algorithm to step over the node or to stop. By default will continue the search. See :class:`step` for more info.
  - `post` works as `pre` but is called once all the children of the node have been visited
  - `up` and `down` are called once the algorithm is moving from a parent to it child or inversely.
  - `down` takes the parent then the child as its arguments
  - `up` takes the child then the parent as its arguments
  """
  is_ok, msg  = dfs_interface_report(g)
  assert is_ok, msg

  S = graph_traversal_stack(g)
  v = make_visitor(f)
  if depth != 'all':
    if depth == 'node'  : depth = 1
    if depth == 'parent': depth = 2
    v = close_ancestor_visitor(v,depth)
  return depth_first_search_stack(S, v)
