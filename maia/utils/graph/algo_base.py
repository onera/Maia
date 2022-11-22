from enum import Enum


def dfs_interface_report(g):
  report = ''
  is_ok = True

  #expected_attrs = [ \
  #  'first_child', \
  #  'children','roots', \
  #  'current_node', \
  #  'range_is_done','advance_node_range','advance_node_range_to_last' \
  #]
  #for attr in expected_attrs:
  #  if not getattr(g, attr, None):
  #    is_ok = False
  #    report += f'Attribute {attr} is missing'

  #if not is_ok:
  #  report = f'dfs_interface_report of type {type(g)}:\n'  + report

  return is_ok, report


class step(Enum):
  into = 0
  over = 1
  out = 2


def make_visitor(v):
  def do_nothing(*args): pass

  if not getattr(v, 'post', None):
    v.post = do_nothing
  if not getattr(v, 'down', None):
    v.down = do_nothing
  if not getattr(v, 'up', None):
    v.up = do_nothing
  return v


class graph_traversal_stack:
    def __init__(self, g):
      self.g = g

      r_iter = iter(g.roots())
      self.active_iter_stack = [r_iter]
      self.active_node_stack = [next(r_iter)]

    def push_level(self, n):
      r_iter = iter(self.g.children(n))
      self.active_iter_stack += [r_iter]
      self.active_node_stack += [next(r_iter)]

    def pop_level(self):
      self.active_iter_stack.pop(-1)
      self.active_node_stack.pop(-1)

    def current_node(self):
      return self.active_node_stack[-1]
    def parent_node(self):
      return self.active_node_stack[-2]

    def advance_node_range(self):
      self.active_node_stack[-1] = next(self.active_iter_stack[-1])
    def advance_node_range_to_last(self):
      self.active_node_stack[-1] = None

    def level_is_done(self) -> bool:
      return self.active_node_stack[-1] is None
    def is_at_root_level(self) -> bool:
      return len(self.active_node_stack) == 1
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
      if next_step == step.over: # prune
        S.push_level(n)
        S.advance_node_range_to_last()
      if next_step is None or next_step == step.into: # go down
        S.push_level(n)
        if not S.level_is_done():
          child = S.current_node()
          f.down(n,child)

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
  v = make_visitor(f)
  return depth_first_search_stack(S, g, v)
