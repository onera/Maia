import operator

def find(seq, pred):
  i = 0
  while i<len(seq):
    if pred(seq[i]):
      return i
    i += 1
  return i

def find_not(seq, pred):
  def not_pred(x): return not pred(x)
  return find(seq, not_pred)


def mismatch(xs, ys):
  """ Finds the first index for which the elements of two sequences are not equal
  """
  i = 0
  for x, y in zip(xs, ys):
    if x != y:
      return i
    i += 1
  return i

def begins_with(seq, seq_begin):
  """ Returns True if the elements in `seq_begin` are the first elements of `seq` 
  """
  if len(seq) < len(seq_begin):
    return False
  return mismatch(seq, seq_begin) == len(seq_begin)

def partition_copy(xs, pred):
  """
  Returns two lists
    - the first with the elements of `xs` where `pred` is True
    - the second with the elements of `xs` where `pred` is False
  """
  xs_true, xs_false = [], []
  for x in xs:
    if pred(x):
      xs_true.append(x)
    else:
      xs_false.append(x)
  return xs_true, xs_false


def set_intersection_difference(x, y, comp = operator.lt):
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
