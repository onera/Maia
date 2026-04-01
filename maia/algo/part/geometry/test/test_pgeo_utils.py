import pytest
import pytest_parallel
import warnings

import numpy as np
import time

def test_get_local_coordinates_perf():
  # We do not really test the function, but we try to check
  # that the heuristic used to switch between 2 implem is correct 

  from maia.utils import s_numbering
  def implem_1(a, idx):
    _idx = s_numbering.index_to_ijk(idx, a.shape)
    for array in _idx:
      array -= 1
    return a[_idx]
    
  def implem_2(a, idx):
    _a = a.flatten(order='F')
    _idx = idx - 1
    return _a[_idx]

  coord = np.asfortranarray(np.random.rand(100, 50, 20))

  large_ids = np.random.randint(1, coord.size+1, coord.size)
  small_ids = np.random.randint(1, coord.size+1, 1000)

  t = time.time()
  o1 = implem_1(coord, small_ids)
  t_impl1 = time.time() - t
  t = time.time()
  o2 = implem_2(coord, small_ids)
  t_impl2 = time.time() - t

  assert np.array_equal(o1,o2)
  if not (t_impl1 < t_impl2):
    warnings.warn('Strange timer result : algo 1 should be faster')

  t = time.time()
  o1 = implem_1(coord, large_ids)
  t_impl1 = time.time() - t
  t = time.time()
  o2 = implem_2(coord, large_ids)
  t_impl2 = time.time() - t

  assert np.array_equal(o1,o2)
  if not (t_impl1 > t_impl2):
    warnings.warn('Strange timer result : algo 2 should be faster')