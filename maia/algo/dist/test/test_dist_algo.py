
import pytest
import pytest_parallel
import numpy as np

import cmaia.dist_algo as cdist_algo

@pytest_parallel.mark.parallel(1)
def test_find_duplicate_elt(comm):
  n_elt = 8
  size_elt = 3

  elt_ec = np.array([1,2,3, 4,5,6, 3,1,2, 4,5,1, 5,4,6, 4,5,6, 2,1,2, 2,1,3], dtype=np.int32)
  result = np.array([    0,     0,     0,     1,     0,     0,     1,     0], dtype=np.int32)

  mask = np.ones(n_elt, dtype=np.int32)
  cdist_algo.find_duplicate_elt(n_elt, size_elt, elt_ec, mask)
  assert np.array_equal(mask, result)

  mask = np.ones(n_elt, dtype=np.int32)
  cdist_algo.find_duplicate_elt2(n_elt, size_elt, elt_ec, mask)
  assert np.array_equal(mask, result)


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
