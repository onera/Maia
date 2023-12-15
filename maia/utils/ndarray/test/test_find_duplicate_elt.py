
import pytest
import numpy as np

from maia.utils import np_utils

def test_find_duplicate_elt():
  size_elt = 3
  elt_ec = np.array([4,5,6, 3,1,2, 4,5,1, 5,4,6, 10,2,3, 1,2,3, 2,1,2, 5,1,1], dtype=np.int32)
  result = np.array([    0,     0,     1,     0,      1,     0,     1,     1], dtype=bool)

  mask = np_utils.is_unique_strided(elt_ec, size_elt, method='hash')
  assert np.array_equal(mask, result)

  mask = np_utils.is_unique_strided(elt_ec, size_elt, method='sort')
  assert np.array_equal(mask, result)



@pytest.mark.parametrize("n_vtx", [100, 100000])
@pytest.mark.parametrize("n_elt", [100, 100000])
def test_find_duplicate_elt_many(n_vtx, n_elt):
  import time

  print(f'\nn_vtx = {n_vtx}')
  print(f'n_elt = {n_elt}')
  elt_size = 3
  
  np.random.seed(seed=1)
  elt_ec = np.random.randint(1, n_vtx, n_elt*elt_size, dtype=np.int32)

  # > Count conflict number
  sum_vtx_per_elt = np.add.reduceat(elt_ec, np.arange(0,n_elt*elt_size,elt_size))
  conflict = np.unique(sum_vtx_per_elt)
  print(f'n_conflict = {conflict.size}')

  start = time.time()
  mask1 = np_utils.is_unique_strided(elt_ec, elt_size, method='hash')
  end   = time.time()
  print(f'TIME v1 = {end-start}')

  start = time.time()
  mask3 = np_utils.is_unique_strided(elt_ec, elt_size, method='sort')
  end   = time.time()
  print(f'TIME v3 = {end-start}')

  # where_diff = np.where(mask1!=mask2)[0]
  # print(f'np.where(mask1!=mask2) = ({where_diff.size}) {where_diff}')
  # print(f'mask1[where_diff] = {mask1[where_diff]}')
  # print(f'mask2[where_diff] = {mask2[where_diff]}')
  assert np.array_equal(mask1, mask3)