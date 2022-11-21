import maia.pytree as PT
from maia.utils.graph import algo_base

class pytree_adaptor:
  def __init__(self, t):
    self.t = t

  def first_child(self, n):
    return PT.get_children(n)[0]

  @staticmethod
  def children(n):
    return [PT.get_children(n), 0]
  def roots(self):
    return [[self.t], 0]

  def current_node(self, ns):
    nodes,pos = ns
    return nodes[pos]

  @staticmethod
  def range_is_done(ns) -> bool:
    nodes,pos = ns
    return len(nodes) == pos

  @staticmethod
  def advance_node_range(ns):
    ns[1] += 1

  @staticmethod
  def advance_node_range_to_last(ns):
    ns[1] = len(ns[0])


def make_visitor(v):
  if not getattr(v, 'post', None):
    v.post = lambda n: 0
  if not getattr(v, 'down', None):
    v.down = lambda above, below: 0
  if not getattr(v, 'up', None):
    v.up = lambda below, above: 0
  return v

def depth_first_find(t,v):
  v = make_visitor(v)
  return algo_base.depth_first_find(pytree_adaptor(t),v)
def depth_first_prune(t,v):
  v = make_visitor(v)
  return algo_base.depth_first_prune(pytree_adaptor(t),v)
def depth_first_scan(t,v):
  v = make_visitor(v)
  return algo_base.depth_first_scan(pytree_adaptor(t),v)
def depth_first_search(t,v):
  v = make_visitor(v)
  return algo_base.depth_first_search(pytree_adaptor(t),v)


def _value_or_none(l, i):
  if i < len(l):
    return l[i]
  else:
    return None

def _zip_lists(ls):
  sz = max(map(len,ls))
  zipped = []
  for i in range(sz):
    zipped.append([_value_or_none(l, i) for l in ls])
  return zipped

def get_sorted_children(n):
  if n is None:
    return []
  else:
    return sorted(PT.get_children(n))


class pytree_zip_adaptor:
  def __init__(self, ts):
    self.ts = ts

  def first_child(self, ns):
    return [PT.get_children(t)[0] for t in self.ts]

  @staticmethod
  def children(ns):
    sorted_ns = [get_sorted_children(n) for n in ns]
    cs = _zip_lists(sorted_ns)
    return [cs, 0]
  def roots(self):
    return [ [self.ts] , 0]

  def current_node(self, ns):
    nodes,pos = ns
    return nodes[pos]

  @staticmethod
  def range_is_done(ns) -> bool:
    nodes,pos = ns
    return len(nodes) == pos

  @staticmethod
  def advance_node_range(ns):
    ns[1] += 1

  @staticmethod
  def advance_node_range_to_last(ns):
    ns[1] = len(ns[0])


def zip_depth_first_find(t,v):
  v = make_visitor(v)
  return algo_base.depth_first_find(pytree_zip_adaptor(t),v)
def zip_depth_first_prune(t,v):
  v = make_visitor(v)
  return algo_base.depth_first_prune(pytree_zip_adaptor(t),v)
def zip_depth_first_scan(t,v):
  v = make_visitor(v)
  return algo_base.depth_first_scan(pytree_zip_adaptor(t),v)
def zip_depth_first_search(t,v):
  v = make_visitor(v)
  return algo_base.depth_first_search(pytree_zip_adaptor(t),v)
