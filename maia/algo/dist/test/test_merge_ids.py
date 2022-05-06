import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np

from maia.algo.dist import merge_ids

@mark_mpi_test(2)
def test_merge_distributed_ids(sub_comm):
  if sub_comm.Get_rank() == 0:
    distri = np.array([0,7,13], np.int32)
    ids     = np.array([6,12,9], np.int32)
    targets = np.array([1,10,7], np.int32)
    expected_old_to_new = [1,2,3,4,5,1,6]
  elif sub_comm.Get_rank() == 1:
    distri = np.array([7,13,13], dtype=np.int32)
    ids     = np.empty(0, dtype=np.int32)
    targets = np.empty(0, dtype=np.int32)
    expected_old_to_new = [7,6,8,9,8,10]

  old_to_new = merge_ids.merge_distributed_ids(distri, ids, targets, sub_comm)
  assert (old_to_new == expected_old_to_new).all()

  if sub_comm.Get_rank() == 0:
    distri = np.array([0,7,13])
    ids     = np.array([6,9])
    targets = np.array([1,7])
    expected_old_to_new = [1,2,3,4,5,1,6]
    expected_signed_old_to_new = [1,2,3,4,5,-1,6]
  elif sub_comm.Get_rank() == 1:
    distri = np.array([7,13,13])
    ids     = np.array([12])
    targets = np.array([10])
    expected_old_to_new = [7,6,8,9,8,10]
    expected_signed_old_to_new = [7,-6,8,9,-8,10]

  old_to_new = merge_ids.merge_distributed_ids(distri, ids, targets, sub_comm)
  assert (old_to_new == expected_old_to_new).all()

  signed_old_to_new = merge_ids.merge_distributed_ids(distri, ids, targets, sub_comm, True)
  assert (signed_old_to_new == expected_signed_old_to_new).all()

