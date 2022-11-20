from enum import Enum


def dfs_interface_report(g):
  report = f'dfs_interface_report of type {type(g)}:\n' 
  is_ok = True

  expected_attrs = [ \
    'first_child', \
    'children','roots', \
    'current_node', \
    'range_is_done','advance_node_range','advance_node_range_to_last' \
  ]
  for attr in expected_attrs:
    if not getattr(g, attr, None):
      is_ok = False
      report += f'Attribute {attr} is missing'

  return is_ok, report


class step(Enum):
  out = 0
  over = 1
  into = 2


class graph_stack:
    def __init__(self, root):
      self.S = [root]

  # basic query
    def is_valid(self) -> bool:
      return len(self.S)>0
   
    def is_at_root_level(self) -> bool:
      return len(self.S)==1
  

  # stack functions
    def push_level(self, x):
      self.S.append(x)
 
    def pop_level(self):
      self.S.pop(-1)


  # accessors
    def current_level(self):
      assert self.is_valid()
      return self.S[-1]

    def parent_level(self):
      assert self.is_valid() and not self.is_at_root_level()
      return self.S[-2]


class graph_traversal_stack:
    def __init__(self, g):
      self.g = g
      self.S = graph_stack(g.roots())

    def current_node(self):
      return self.g.current_node( self.S.current_level() )
    def parent_node(self):
      # note: unable to use Python iterators because this function needs to read several times without incrementing
      return self.g.current_node( self.S.parent_level() )

    def advance_node_range(self):
      self.g.advance_node_range( self.S.current_level() )
    def advance_node_range_to_last(self):
      self.g.advance_node_range_to_last( self.S.current_level() )

    def push_level(self, n):
      self.S.push_level(self.g.children(n))
    def pop_level(self):
      self.S.pop_level()

    def level_is_done(self) -> bool:
      return self.g.range_is_done( self.S.current_level() )
    def is_at_root_level(self) -> bool:
      return self.S.is_at_root_level()
    def is_done(self) -> bool:
      return self.is_at_root_level() and self.level_is_done()


def unwind(S, f):
  while not S.is_at_root_level():
    n = S.current_node()
    parent = S.parent_node()
    f.post(n)
    f.up(n, parent)
    S.pop_level()

  n = S.current_node()
  f.post(n)


def depth_first_search_stack(S, g, f):
  while not S.is_done():
    if not S.level_is_done():
      n = S.current_node()
      next_step = f.pre(n)
      if next_step == step.out: # stop
        matching_node = S.current_node()
        unwind(S, f)
        return matching_node
      if next_step == step.over:  # prune
        S.push_level(n)
        S.advance_node_range_to_last()
      if next_step == step.into:  # go down
        S.push_level(n)
        if not S.level_is_done():
          f.down(n,g.first_child(n))

    else:
      S.pop_level()
      n = S.current_node()
      f.post(n)
      S.advance_node_range()
      if not S.is_at_root_level():
        parent = S.parent_node()
        f.up(n, parent)
        if not S.level_is_done():
          w = S.current_node()
          f.down(parent, w)

  return None


def depth_first_search(g, f):
  is_ok, msg  = dfs_interface_report(g)
  assert is_ok, msg

  S = graph_traversal_stack(g)
  return depth_first_search_stack(S, g, f)


# adaptation of general algorithm to find,prune and scan {
#
## The general algorithm asks if it should step out/over/into
## So the visitor's `pre` function must return a value of `step` type
## Here, we handle simpler `pre` functions:
## - find: return true to step out, return false to continue
## - prune: return true to step over, return false to continue
## - scan : always continue
class depth_first_visitor_adaptor:
  def __init__(self, f, convert_to_step):
    self.f = f
    self.convert_to_step = convert_to_step

  def pre(self, n) -> step:
    return self.convert_to_step(self.f,n)
  def down(self, above, below):
    self.f.down(above,below)
  def up(self, below, above):
    self.f.up(below,above)
  def post(self, n):
    self.f.post(n)

def depth_first_find(g, f):
  def convert_to_step(f, n): return step.out if f.pre(n) else step.into
  vis = depth_first_visitor_adaptor(f,convert_to_step)
  return depth_first_search(g,vis)

def depth_first_prune(g, f):
  def convert_to_step(f, n): return step.over if f.pre(n) else step.into
  vis = depth_first_visitor_adaptor(f,convert_to_step)
  depth_first_search(g,vis)

def depth_first_scan(g, f):
  def convert_to_step(f, n):
    f.pre(n)
    return step.into
  vis = depth_first_visitor_adaptor(f,convert_to_step)
  depth_first_search(g,vis)

# adaptation of general algorithm to find, prune and scan }
