import maia.pytree as PT
from maia.utils.graph import algo_base
from maia.utils.graph.algo_base import step


class adjacency_iterator:
  def __init__(self, ts):
    self.ts = ts

  def __iter__(self):
    self.idx = 0
    return self

  def __next__(self):
    if self.idx < len(self.ts):
      t = self.ts[self.idx]
      self.idx += 1
      return t
    else:
      return None

class pytree_adaptor:
  def __init__(self, t):
    self.t = t

  def children(self, n):
    return adjacency_iterator(PT.get_children(n))
  def roots(self):
    return adjacency_iterator([self.t])


def depth_first_search(t,v):
  return algo_base.depth_first_search(pytree_adaptor(t),v)


def _value_or_none(l, i):
  if i < len(l):
    return l[i]
  else:
    return None

# TODO move from here
def set_intersection_difference(x, y, comp):
  """
    Returns 4 lists `(inter_x, diff_x, inter_y, diff_y)`:
      - `inter_x`, are `diff_x` are subsets of elements in `x`
      - `inter_y`, are `diff_y` are subsets of elements in `y`
    `inter_x` and `inter_y` contain elements both present in `x` and `y` according the `comp`
    `diff_x` and `diff_y` contain elements not present in both `x` and `y` according the `comp`

    `x` and `y` must be sorted according to `comp` 
    The returned lists will be sorted
  """
  i = 0
  j = 0
  inter_x = []
  inter_y = []
  diff_x  = []
  diff_y  = []
  while i<len(x) and j<len(y):
    if comp(x[i],y[j]):
      diff_x.append( x[i] )
      i += 1
    elif comp(y[j],x[i]):
      diff_y.append( y[j] )
      j += 1
    else: # x[i]==y[j]:
      inter_x.append( x[i] )
      inter_y.append( y[j] )
      i += 1
      j += 1
  diff_x += x[i:]
  diff_y += y[j:]
  return inter_x, diff_x, inter_y, diff_y

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
    return adjacency_iterator(cs)
  def roots(self):
    return adjacency_iterator([self.ts])


def zip_depth_first_search(t,v):
  return algo_base.depth_first_search(pytree_zip_adaptor(t),v)
