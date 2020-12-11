import numpy as np
import itertools as ITT

def search_match(L, target, max_tries=2**20, tol=0):
  """
  For a sorted list L of integer values, search a combination of indexes
  such that sum(L[Indexes]) == target

  Args:
      L (array, sorted) : List of values
      target      (int) : Wanted value
      max_tries    (int) : Give up and return None if the number of tested
                          permutations reaches max_tries
      tol       (float) : Allows sum(L[Indexes]) to be close but not exactly
                          equal to target

  Returns               : list of matching indexes or empty list
  """
  n_tries = 0

  # > The list must be sorted (ascending)
  if (np.diff(L)>=0).all() == False:
    LOG.warning('Error in search_match : input list must be sorted (ascending)')
    return

  # We can exclude all elems > target from the input list
  too_large_indexes = [i for i, elem in enumerate(L) if elem > target]
  if len(too_large_indexes) > 0:
    L = L[0:too_large_indexes[0]]
  N = len(L)

  # > Create the iteration list : N, 1, N-1, 2, N-2, 3 etc. in order to start with less
  #   expensive cases
  iter_order = np.empty(N, dtype=np.int32)
  iter_order[::2]  = range(N, N//2, -1)
  iter_order[1::2] = range(1,N//2+1)

  # > Loop around the combinations and search target
  for k in iter_order:
    #LOG.debug("Try to find combination using {0}-uplets, ntries is {1}".format(k, n_tries))
    for idx in ITT.combinations(range(N), k):
      n_tries += 1
      combination_sum = sum([L[i] for i in idx])
      # Since the list is sorted, if this sum is > target, all the others are too
      if combination_sum > (1+tol)*target:
        break
      if np.abs(combination_sum - target) <= tol*target:
        #LOG.debug("Found matching combinaton in search_match")
        return idx
    if n_tries > max_tries:
      #LOG.debug("Exit search_match because max_tries as been reached")
      return []

  # > No match have been found
  #LOG.debug("Exit search_match because no match have been found")
  return []

# ------------------------------------------------------------------
