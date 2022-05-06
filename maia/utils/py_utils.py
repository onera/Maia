import sys
if sys.version_info.major == 3 and sys.version_info.major < 8:
  from collections.abc import Iterable  # < py38
else:
  from typing import Iterable
import re
from itertools import permutations

def camel_to_snake(text, keep_upper=False):
  """
  Return a snake_case string from a camelCase string.
  If keep_upper is True, upper case words in camelCase are keeped upper case
  """
  ptou    = re.compile(r'(2)([A-Z]+)([A-Z][a-z])')
  ptol    = re.compile(r'(2)([A-Z][a-z])')
  tmp = re.sub(ptol, r'_to_\2', re.sub(ptou, r'_to_\2', text))
  pupper = re.compile(r'([A-Z]+)([A-Z][a-z])')
  plower = re.compile(r'([a-z\d])([A-Z])')
  word = plower.sub(r'\1_\2', re.sub(pupper, r'\1_\2', tmp))
  if keep_upper:
    return '_'.join([w if all([i.isupper() for i in w]) else w.lower() for w in word.split('_')])
  else:
    return word.lower()

# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
def flatten(items):
  """Yield items from any nested iterable; see Reference."""
  for x in items:
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
      yield from flatten(x)
    else:
      yield x

def list_or_only_elt(l):
  return l[0] if len(l) == 1 else l

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

def are_overlapping(range1, range2, strict=False):
  """ Return True if range1 and range2 share a common element.
  If strict=True, case (eg) End1 == Start2 is not considered to overlap 
  https://is.gd/gTBuwu """
  assert range1[0] <= range1[1] and range2[0] <= range2[1]
  if strict:
    return range1[0] < range2[1] and range2[0] < range1[1]
  else:
    return range1[0] <= range2[1] and range2[0] <= range1[1]


def is_subset_l(subset, L):
  """Return True is subset list is included in L, allowing looping"""
  extended_l = list(L) + list(L)[:len(subset)-1]
  return max([subset == extended_l[i:i+len(subset)] for i in range(len(L))])

def append_unique(L, item):
  """ Add an item in a list only if not already present"""
  if item not in L:
    L.append(item)

def expects_one(L, err_msg=("elem", "list")):
  """
  Raise a RuntimeError if L does not contains exactly one element. Otherwise,
  return this element
  """
  assert isinstance(L, list)
  if len(L) == 0:
    raise RuntimeError(f"{err_msg[0]} not found in {err_msg[1]}")
  elif len(L) > 1:
    raise RuntimeError(f"Multiple {err_msg[0]} found in {err_msg[1]}")
  else:
    return L[0]


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

