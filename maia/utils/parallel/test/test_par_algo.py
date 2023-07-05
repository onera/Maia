import pytest_parallel

import numpy as np

from maia.utils import py_utils
from maia.utils.parallel import algo as par_algo

@pytest_parallel.mark.parallel(3)
def test_dist_set_difference(comm):
  if comm.Get_rank() == 0:
    ids = np.array([34,19])
    others = [np.array([4,9,2,11])]
    expected = np.array([12, 19])
  if comm.Get_rank() == 1:
    ids = np.array([11,12])
    others = [np.array([11,5])]
    expected = np.array([33])
  if comm.Get_rank() == 2:
    ids = np.array([5, 19, 33])
    others = [np.empty(0, dtype=int)]
    expected = np.array([34])

  assert (par_algo.dist_set_difference(ids, others, comm) == expected).all()

@pytest_parallel.mark.parallel(3)
def test_DistSorter(comm):
  if comm.Get_rank() == 0:
    key = np.array([34,19])
    data = np.array([1.,2.])
    expected = np.array([3., 2.])
  if comm.Get_rank() == 1:
    key = np.array([3,40,42,27])
    data = np.array([3., 4., 5., 6.])
    expected = np.array([6.,1.])
  if comm.Get_rank() == 2:
    key = np.empty(0, int)
    data = np.empty(0, float)
    expected = np.array([4.,5.])
  #Key order is 3,19,27,34,40,42

  sorter = par_algo.DistSorter(key, comm)
  assert (sorter.sort(data) == expected).all()


@pytest_parallel.mark.parallel([1,4])
def test_compute_gnum_s(comm):
  # Simple test with strings  
  keys = [['Pear', 'Banana', 'Peach', 'Banana'],
          [],
          ['Watermelon', 'Banana'],
          ['Melon', 'Peach', 'Tomato']]
  expected_gnum = [[4,0,3,0],
                   [],
                   [1,0],
                   [5,3,2]]

  if comm.Get_size() == 4:
    rank_keys = keys[comm.Get_rank()]
    rank_gnum = expected_gnum[comm.Get_rank()]
  else: #Results should be // independant
    rank_keys = py_utils.to_flat_list(keys)
    rank_gnum = py_utils.to_flat_list(expected_gnum)

  assert (par_algo.compute_gnum(rank_keys, comm) == rank_gnum).all()

@pytest_parallel.mark.parallel(2)
def test_compute_gnum_o(comm):
  # Test with various objects
  if comm.Get_rank() == 0:
    keys = [test_compute_gnum_s, 'Banana', np.array([1,2,3], np.int32)]
    expected_gnum = [4,0,1]
  elif comm.Get_rank() == 1:
    keys = [None, False, test_compute_gnum_s, 'Banana']
    expected_gnum = [3,2,4,0]

  assert (par_algo.compute_gnum(keys, comm) == expected_gnum).all()
