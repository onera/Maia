import pytest_parallel

import numpy as np

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

