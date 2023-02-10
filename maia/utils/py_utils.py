from itertools import permutations

def to_nested_list(l, counts):
  """ Transform a flat list to a list of lists"""
  assert len(l) == sum(counts)
  nested = list()
  r_idx = 0
  for n_elts in counts:
    nested.append(l[r_idx:r_idx+n_elts])
    r_idx += n_elts
  return nested

def bucket_split(l, f, compress=False, size=None):
  """ Dispatch the elements of list l into n sublists, according to the result of function f """
  if size is None: 
    size = max(f(e) for e in l) + 1
  result = [ [] for i in range(size)]
  for e in l:
    result[f(e)].append(e)
  if compress:
    result = [sub_l for sub_l in result if sub_l]
  return result

def is_subset_l(subset, L):
  """Return True is subset list is included in L, allowing looping"""
  extended_l = list(L) + list(L)[:len(subset)-1]
  return max([subset == extended_l[i:i+len(subset)] for i in range(len(L))])

def append_unique(L, item):
  """ Add an item in a list only if not already present"""
  if item not in L:
    L.append(item)

def loop_from(L, i):
  """ Iterator over a list L, starting from element i (wrapping around at the end)"""
  assert 0 <= i and i < len(L)
  yield from L[i:]
  yield from L[:i]

def find_cartesian_vector_names(names):
  """
  Function to find basename of cartesian vectors
  In the SIDS (https://cgns.github.io/CGNS_docs_current/sids/dataname.html), a cartesian
  vector 'Vector' is describe by its 3 components 'VectorX', 'VectorY', 'VectorZ'
  > names : list of potential vectors components
  """
  to_index = {'X' : 0, 'Y' : 1 , 'Z' : 2}
  suffix_names = [set() for i in to_index]
  for name in names:
    last = name[-1]
    try:
      suffix_names[to_index[last]].add(name[0:-1])
    except KeyError:
      pass

  common = suffix_names[0].intersection(*suffix_names[1:])
  return sorted(common)

def get_ordered_subset(subset, L):
  """
  Check is one of the permutations of subset exists in L, allowing looping
  Return the permutation if existing, else None
  TODO if n=len(L) and k=len(subset), worst case complexity is k! * n. 
  TODO Replace by this algorithm (should be n * k ln(k))
    subset = sort(subset) # we don't care about the order of this one, might as well sort it
    extended_l = list(L) + list(L)[:len(subset)-1] # ugly: is there a way to create a lazy circular list easily?
    for i in range(len(extended_l)-len(subset)): # TODO: +/- 1 ?
      if subset[0]==extended_l[i]:
        if match(extended_l,i+1,subset[1:]) # is k ln(k) since will binary search extended_l[j] (which is ln k) k times in subset
          return extended_l[i:i+k]
    return None
  """
  extended_l = list(L) + list(L)[:len(subset)-1]
  for perm in permutations(subset, len(subset)):
    perm_l = list(perm)
    if max([perm_l == extended_l[i:i+len(perm_l)] for i in range(len(L))]) == True:
      return perm

def is_before(l, a, b):
  """Return True is element a is present in list l before element b"""
  for e in l:
    if e==a:
      return True
    if e==b:
      return False
  return False

def uniform_distribution_at(n_elt, i, n_interval):
  """
  """
  step      = n_elt // n_interval
  remainder = n_elt %  n_interval

  if i < remainder:
    inf = i * (step + 1)
    sup = inf + step + 1
  else:
    inf = i * step + remainder
    sup = inf + step

  return inf,sup

def str_to_bools(size, key):
  """
  Convert a keyword into a list of booleens of the given size
  """
  if key == "none":
    return size*[False]
  elif key == "all":
    return size*[True]
  elif key == "ancestors":
    return [True]*(size-1) + [False]
  elif key == "leaf":
    return [False]*(size-1) + [True]
  else:
    raise ValueError(f"key must be one of {{'none', 'all', 'ancestors' or 'leaf'}}")


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
