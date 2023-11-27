
import pytest
import pytest_parallel
import numpy as np

import cmaia.dist_algo as cdist_algo

@pytest_parallel.mark.parallel(1)
def test_find_duplicate_elt(comm):
  n_elt = 8
  size_elt = 3

  elt_ec = np.array([4,5,6, 3,1,2, 4,5,1, 5,4,6, 10,2,3, 1,2,3, 2,1,2, 5,1,1], dtype=np.int32)
  result = np.array([    0,     0,     1,     0,      1,     0,     1,     1], dtype=np.int32)

  mask = np.ones(n_elt, dtype=np.int32)
  cdist_algo.find_duplicate_elt(n_elt, size_elt, elt_ec, mask)
  assert np.array_equal(mask, result)

  mask = np.ones(n_elt, dtype=np.int32)
  cdist_algo.find_duplicate_elt2(n_elt, size_elt, elt_ec, mask)
  assert np.array_equal(mask, result)

  mask = np.ones(n_elt, dtype=np.int32)
  cdist_algo.find_duplicate_elt3(n_elt, size_elt, elt_ec, mask)
  assert np.array_equal(mask, result)


@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("n_vtx", [100, 1000000000])
@pytest.mark.parametrize("n_elt", [100, 100000])
def test_find_duplicate_elt_many(n_vtx, n_elt, comm):
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
  mask1 = np.ones(n_elt, dtype=np.int32)
  cdist_algo.find_duplicate_elt(n_elt, elt_size, elt_ec, mask1)
  end   = time.time()
  print(f'TIME v1 = {end-start}')

  start = time.time()
  mask2 = np.ones(n_elt, dtype=np.int32)
  cdist_algo.find_duplicate_elt2(n_elt, elt_size, elt_ec, mask2)
  end   = time.time()
  print(f'TIME v2 = {end-start}')

  start = time.time()
  mask3 = np.ones(n_elt, dtype=np.int32)
  cdist_algo.find_duplicate_elt3(n_elt, elt_size, elt_ec, mask3)
  end   = time.time()
  print(f'TIME v3 = {end-start}')

  # where_diff = np.where(mask1!=mask2)[0]
  # print(f'np.where(mask1!=mask2) = ({where_diff.size}) {where_diff}')
  # print(f'mask1[where_diff] = {mask1[where_diff]}')
  # print(f'mask2[where_diff] = {mask2[where_diff]}')
  assert np.array_equal(mask1, mask2)
  assert np.array_equal(mask1, mask3)
  assert np.array_equal(mask2, mask3)


@pytest_parallel.mark.parallel(1)
def test_find_tri_in_tetras(comm):
  n_tri = 5
  n_tet = 4
  tri_ec = np.array([1,2,3, 4,5,6, 3,2,5, 4,5,2, 5,1,3], dtype=np.int32)
  tet_ec = np.array([7,4,5,6, 3,2,4,1, 4,5,6,1, 3,4,5,1], dtype=np.int32)
  result = np.array([1, 1, 0, 0, 1], dtype=np.int32)

  mask = np.zeros(n_tri, dtype=np.int32)
  cdist_algo.find_tri_in_tetras(n_tri, n_tet, tri_ec, tet_ec, mask)

  assert np.array_equal(mask, result)
