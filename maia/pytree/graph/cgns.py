import maia.pytree as PT
from maia.pytree.graph import algo
from maia.pytree.graph.algo import step
from maia.pytree.algo_utils import set_intersection_difference


class pytree_adaptor:
  def __init__(self, t):
    self.t = t

  def children(self, n):
    return iter(PT.get_children(n))
  def roots(self):
    return iter([self.t])


def depth_first_search(t, v, depth='node'):
  return algo.depth_first_search(pytree_adaptor(t), v, depth)


def _value_or_none(l, i):
  if i < len(l):
    return l[i]
  else:
    return None


def _zip_lists(ls):
  assert len(ls) == 2

  def less_by_name(x, y):
    return PT.get_name(x) < PT.get_name(y)
  inter_x, diff_x, inter_y, diff_y = set_intersection_difference(ls[0], ls[1], less_by_name)

  zipped = []

  for i in range(len(inter_x)):
    zipped.append( [inter_x[i], inter_y[i]] )

  sz = max(len(diff_x),len(diff_y))
  for i in range(sz):
    zipped.append( [_value_or_none(diff_x, i), _value_or_none(diff_y, i)] )

  return zipped

def _get_sorted_children(n):
  if n is None:
    return []
  else:
    return sorted(PT.get_children(n))


class pytree_zip_adaptor:
  def __init__(self, ts):
    self.ts = ts

  def children(self, ns):
    sorted_ns = [_get_sorted_children(n) for n in ns]
    cs = _zip_lists(sorted_ns)
    return iter(cs)
  def roots(self):
    return iter([self.ts])

def zip_depth_first_search(t, v, depth='node'):
  return algo.depth_first_search(pytree_zip_adaptor(t), v, depth)
