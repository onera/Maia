import Converter.Internal as I
import numpy as np
from itertools import permutations

def list_or_only_elt(l):
  return l[0] if len(l) == 1 else l

def is_subset_l(subset, L):
  """Return True is subset list is included in L, allowing looping"""
  extended_l = list(L) + list(L)[:len(subset)-1]
  return max([subset == extended_l[i:i+len(subset)] for i in range(len(L))])

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

def interweave_arrays(array_list):
  #https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
  first  = array_list[0]
  number = len(array_list)
  output = np.empty(number*first.size, first.dtype)
  for i,array in enumerate(array_list):
    output[i::number] = array
  return output

def single_dim_pr_to_pl(pr, distrib=None):
  assert pr.shape[0] == 1
  if distrib is not None:
    return np.arange(pr[0,0]+distrib[0], pr[0,0]+distrib[1], dtype=pr.dtype).reshape((1,-1), order='F')
  else:
    return np.arange(pr[0,0], pr[0,1]+1, dtype=pr.dtype).reshape((1,-1), order='F')

def concatenate_point_list(point_lists, dtype=None):
  """
  Merge all the PointList arrays in point_lists list
  into a flat 1d array and an index array
  """
  sizes = [pl_n.size for pl_n in point_lists]

  merged_pl_idx = sizes_to_indices(sizes, dtype=np.int32)

  if dtype is None:
    dtype = point_lists[0].dtype if point_lists != [] else np.int

  merged_pl = np.empty(sum(sizes), dtype=dtype)
  for ipl, pl in enumerate(point_lists):
    merged_pl[merged_pl_idx[ipl]:merged_pl_idx[ipl+1]] = pl[0,:]

  return merged_pl_idx, merged_pl

def sizes_to_indices(nb_array, dtype=None):
  """ Create and offset array from a size array """
  nptype = dtype if dtype else np.asarray(nb_array).dtype
  offset_array = np.empty(len(nb_array)+1, dtype=nptype)
  offset_array[0] = 0
  np.cumsum(nb_array, out=offset_array[1:])
  return offset_array

def reverse_connectivity(ids, idx, array):
  """
  Reverse an strided array (idx+array) supported by some elements whose id is given by ids
  Return a strided array(r_idx+r_array) and the ids of (initially childs) elements
  supporting it
  """
  r_ids, counts = np.unique(array, return_counts=True)
  sort_idx = np.argsort(array)
  sizes = np.diff(idx)
  extended_ids = np.repeat(ids, sizes)
  r_array = extended_ids[sort_idx]
  r_idx = sizes_to_indices(counts)

  return (r_ids, r_idx, r_array)

def multi_arange(starts, stops):
  """
  Create concatenated np.arange of integers for multiple start/stop
  """
  assert len(starts)==len(stops)
  if len(starts)==0: return []
  return np.concatenate([np.arange(start,stop) for start,stop in zip(starts,stops)])

def arange_with_jumps(multi_interval,jumps):
  """
  Create an arange, but where sub-intervals are removed
  """
  multi_interval = np.asarray(multi_interval)
  jumps = np.asarray(jumps)
  return multi_arange(multi_interval[ :-1][~jumps],
                      multi_interval[1:  ][~jumps])

def roll_from(array, start_idx = None, start_value = None, reverse = False):
  """
  Return a new array starting from given index (or value), in normal or reversed order
  """
  assert (start_idx is None) != (start_value is None)
  if start_idx is None:
    start_idx = np.where(array == start_value)[0][0]

  return np.roll(array, -start_idx) if not reverse else np.roll(array[::-1], start_idx + 1)

def others_mask(array, ids):
  """
  Return a mask usefull to access elements of array whose local index *are not* in ids array
  """
  mask = np.ones(array.size, dtype=bool)
  mask[ids] = False
  return mask

def any_in_range(array, start, end, strict=False):
  """
  Return True if any element of array is in interval
  [start, end]. In is large by defaut and strict is strict==True
  """
  np_array = np.asarray(array)
  return ((start <  np_array) & (np_array <  end)).any() if strict\
    else ((start <= np_array) & (np_array <= end)).any()

def all_in_range(array, start, end, strict=False):
  """
  Return True if all the elements of array are in interval
  [start, end]. In is large by defaut and strict is strict==True
  """
  np_array = np.asarray(array)
  return ((start <  np_array) & (np_array <  end)).all() if strict\
    else ((start <= np_array) & (np_array <= end)).all()

