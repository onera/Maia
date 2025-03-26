from itertools import permutations, product
from typing import List, TypeVar, Callable, Any, Sequence, Tuple, Optional

T = TypeVar('T')
U = TypeVar('U')

def to_nested_list(l: List[T], counts: List[int]) -> List[List[T]]:
  """ Transform a flat list to a list of lists"""
  assert len(l) == sum(counts)
  nested = list()
  r_idx = 0
  for n_elts in counts:
    nested.append(l[r_idx:r_idx+n_elts])
    r_idx += n_elts
  return nested

def to_flat_list(nested_list: List[List[T]]) -> List[T]:
  """ Transform a list of list to a flat list"""
  return [obj for l in nested_list for obj in l]

def bucket_split(l: List[T], f: Callable[[T], int], compress: bool = False, size: Optional[int] = None) -> List[List[T]]:
  """ Dispatch the elements of list l into n sublists, according to the result of function f """
  if size is None: 
    size = max(f(e) for e in l) + 1
  result = [ [] for i in range(size)]
  for e in l:
    result[f(e)].append(e)
  if compress:
    result = [sub_l for sub_l in result if sub_l]
  return result

def is_subset_l(subset: List[T], L: List[T]) -> bool:
  """Return True is subset list is included in L, allowing looping"""
  extended_l = list(L) + list(L)[:len(subset)-1]
  return max([subset == extended_l[i:i+len(subset)] for i in range(len(L))])

def append_unique(L: List[T], item: T) -> None:
  """ Add an item in a list only if not already present"""
  if item not in L:
    L.append(item)

def loop_from(L: List[T], i: int):
  """ Iterator over a list L, starting from element i (wrapping around at the end)"""
  assert 0 <= i and i < len(L)
  yield from L[i:]
  yield from L[:i]

def find_tensor_names(names: List[str], axis: List[str]) -> List[str]:
  """ Return the name of the fields appearing to be a tensor """
  assert len(axis) >= 1
  names = [name for name in names if len(name) > 2] #Exclude crazy cases

  # For tensor, we will search only diagonal components
  to_index = {f'{a}{a}':i for i,a in enumerate(axis)} 
  suffix_names = [set() for _ in to_index]

  for name in names:
    for suffix, index in to_index.items():
      if name.endswith(suffix):
        base_name = name[:-len(suffix)]
        suffix_names[index].add(base_name)
        break

  common = suffix_names[0].intersection(*suffix_names[1:])
  return sorted(common)

def find_vector_names(names: List[str], axis: List[str]) -> List[str]:
  """ Return the name of the fields appearing to be a vector """
  assert len(axis) >= 1

  # Exclude tensors
  tens_suffixes = [''.join(p) for p in product(axis, repeat=2)]
  tensor_comps = []
  for tensor_name in find_tensor_names(names, axis):
    tensor_comps.extend([tensor_name + s for s in tens_suffixes])
  names = [name for name in names if name not in tensor_comps]
  # Exclude single char names
  names = [name for name in names if len(name) > 1]

  to_index = {a:i for i,a in enumerate(axis)}
  suffix_names = [set() for _ in to_index]

  for name in names:
    for suffix, index in to_index.items():
      if name.endswith(suffix):
        base_name = name[:-len(suffix)]
        suffix_names[index].add(base_name)
        break

  common = suffix_names[0].intersection(*suffix_names[1:])
  return sorted(common)

def find_cartesian_vector_names(names: List[str], phy_dim: int = 3) -> List[str]:
  return find_vector_names(names, ['X', 'Y', 'Z'][:phy_dim])

def find_auxiliary_vector_names(names: List[str], phy_dim: int = 3) -> List[str]:
  return find_vector_names(names, ['Xi', 'Eta', 'Zeta'][:phy_dim])

def find_cylindric_vector_names(names: List[str], phy_dim: int = 3) -> List[str]:
  return find_vector_names(names, ['R', 'Theta', 'Z'][:phy_dim])

def find_spherical_vector_names(names: List[str], phy_dim: int = 3) -> List[str]:
  return find_vector_names(names, ['R', 'Theta', 'Phi'][:phy_dim])

def get_ordered_subset(subset: List[T], L: List[T]) -> Optional[Tuple[T, ...]]:
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

def is_before(l: List[T], a: T, b: T) -> bool:
  """Return True is element a is present in list l before element b"""
  for e in l:
    if e==a:
      return True
    if e==b:
      return False
  return False

def any_true(iterable: Sequence[T], predicate: Callable[[T], bool]) -> bool:
  return any(predicate(elem) for elem in iterable)

def all_true(iterable: Sequence[T], predicate: Callable[[T], bool]) -> bool:
  return all(predicate(elem) for elem in iterable)

def uniform_distribution_at(n_elt: int, i: int, n_interval: int) -> Tuple[int, int]:
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

def unique_idx(seq: Sequence[T]) -> List[int]:
  """ Indirect unique of a sequence : return an array of size len(seq)
  storing an unique id for each element occuring in sequence
  """

  size = len(seq)
  if size == 0:
    return []

  idx = sorted(range(size), key=seq.__getitem__)
  out = [-1] * size

  id = 0
  last = seq[idx[0]]
  for i in idx:
    if seq[i] != last:
      last = seq[i]
      id += 1
    out[i] = id
  return out

  

def str_to_bools(size: int, key: str) -> List[bool]:
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

def overlap_size(start1: int, end1: int, start2: int, end2: int) -> int:
  """ Number of common elements for two given intervals """
  return max(min(end1, end2) - max(start1, start2), 0)