import pytest_parallel

import numpy as np

from maia.utils.parallel import algo as par_algo

@pytest_parallel.mark.parallel(3)
def test_dist_set_difference(comm):
  if comm.Get_rank() == 0:
    ids = np.array([19,34])
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
