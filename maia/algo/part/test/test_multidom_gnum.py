import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import maia.algo.part.multidom_gnum as MGM

@mark_mpi_test(3)
def test_get_shifted_arrays(sub_comm):
  if sub_comm.Get_rank() == 0:
    arrays = [[np.array([1,8,4])],  [np.array([3,6,1,9])]]
    expected = [[np.array([1,8,4])],  [np.array([13,16,11,19])]]
  elif sub_comm.Get_rank() == 1:
    arrays = [[],  [np.array([9,3,1,4]), np.array([8,6])]]
    expected = [[],  [np.array([19,13,11,14]), np.array([18,16])]]
  elif sub_comm.Get_rank() == 2:
    arrays = [[np.array([10,2])],  [np.empty(0)]]
    expected = [[np.array([10,2])],  [np.empty(0)]]

  offset, shifted_arrays = MGM._get_shifted_arrays(arrays, sub_comm)
  assert (offset == [0,10,19]).all()
  for i in range(2):
    for t1, t2 in zip(shifted_arrays[i], expected[i]):
      assert (t1 == t2).all()


