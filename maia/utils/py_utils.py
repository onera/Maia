import re
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

def to_flat_list(nested_list):
  """ Transform a list of list to a flat list"""
  return [obj for l in nested_list for obj in l]

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

def find_vector_names(names, axis):
  """
  Function to find basename of any coordinates system
  In the SIDS (https://cgns.github.io/CGNS_docs_current/sids/dataname.html), a cartesian
  vector 'Vector' is describe by its 2 (resp. 3) components 'VectorX', 'VectorY', (resp. 'VectorZ')
  depending on the physical dimension of the mesh. 
  > names : list of potential vectors components
  > axis : Coordinates system of the mesh.
  """
  assert len(axis) > 1
  names = [name for name in names if len(name) > 1] #Exclude crazy cases
  tensors = [axis[0]+axis[0], axis[1]+axis[1]]
  to_index = {axis[0] : 0, axis[1] : 1}
  if len(axis) == 3:
    tensors.append(axis[2]+axis[2])
    to_index[axis[2]] = 2
  # tensors = sorted(tensors)

  basenames = [set() for i in to_index]

  suffix_names = []

  for name in names:
    is_lower = name[0].islower()
    if is_lower:
      name = name[0].upper() + name[1:]
    split_name = re.findall('[A-Z][^A-Z]*', name)
    suffix_names.append(''.join(split_name[-2:]))
    if len(split_name) > 1: 
      basename = ''.join(split_name[0:-1])
      if is_lower:
        basename = basename[0].lower() + basename[1:]
    try:
        basenames[to_index[split_name[-1]]].add(basename)
    except KeyError:
      pass
  is_tensors = sorted(list(set(suffix_names) & set(tensors)))
  if is_tensors != tensors :
    vectors = basenames[0].intersection(*basenames[1:])
  else:
    vectors = []
    Warning('Tensors are not included')
  return sorted(vectors)

def find_cartesian_vector_names(names, phy_dim=3):
  return find_vector_names(names, ['X', 'Y', 'Z'][:phy_dim])

def find_auxiliary_vector_names(names, phy_dim=3):
  return find_vector_names(names, ['Xi', 'Eta', 'Zeta'][:phy_dim])

def find_cylindric_vector_names(names, phy_dim=3):
  return find_vector_names(names, ['R', 'Theta', 'Z'][:phy_dim])

def find_spherical_vector_names(names, phy_dim=3):
  return find_vector_names(names, ['R', 'Theta', 'Phi'][:phy_dim])

names = ['ZSRR', 'ZSRTheta', 'ZSRZ','TotoR', 'TotoTheta', 'TotoZ', 'tat']
basename_vectors = find_cylindric_vector_names(names)
assert basename_vectors == ['Toto', 'ZSR']
# names = ['ZSRRR', 'ZSRRTheta', 'ZSRRZ', 'ZSRThetaR', 'ZSRThetaTheta', 'ZSRThetaZ', 'ZSRZR', 'ZSRZTheta', 'ZSRZZ']
# basename_tensors = find_cylindric_vector_names(names)
# assert basename_tensors == []
# names = ['ZSRRR', 'ZSRRTheta', 'ZSRRZ', 'ZSRThetaR', 'ZSRZR', 'ZSRZTheta', 'ZSRZZ']
# basename_tensors = find_cylindric_vector_names(names)
# assert basename_tensors == ['ZSRR', 'ZSRZ']
# names = ['ZSRRR', 'ZSRRTheta', 'ZSRRPhi', 'ZSRThetaR', 'ZSRThetaTheta', 'ZSRThetaPhi', 'ZSRPhiR', 'ZSRPhiTheta', 'ZSRPhiPhi']
# basename_tensors = find_spherical_vector_names(names)
# assert basename_tensors == []
# names = ['ZSRR', 'ZSRTheta', 'ZSRZ']
# basename_tensors = find_cylindric_vector_names(names)
# assert basename_tensors == ['ZSR']

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

def any_true(iterable, predicate):
  return any(predicate(elem) for elem in iterable)

def all_true(iterable, predicate):
  return all(predicate(elem) for elem in iterable)

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

def unique_idx(seq):
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
