import pytest
import pytest_parallel

import numpy as np

from maia.algo.dist import merge_ids

@pytest_parallel.mark.parallel(2)
def test_remove_distributed_ids(comm):
  if comm.Get_rank() == 0:
    distri = np.array([0,7,13], np.int32)
    ids     = np.array([1, 6,12,9], np.int32)
    expected_old_to_new = [-1,1,2,3,4,-1,5]
  elif comm.Get_rank() == 1:
    distri = np.array([7,13,13], dtype=np.int32)
    ids     = np.empty(0, dtype=np.int32)
    expected_old_to_new = [6,-1,7,8,-1,9]

  old_to_new = merge_ids.remove_distributed_ids(distri, ids, comm)
  assert (old_to_new == expected_old_to_new).all()

@pytest_parallel.mark.parallel(2)
def test_merge_distributed_ids(comm):
  if comm.Get_rank() == 0:
    distri = np.array([0,7,13], np.int32)
    ids     = np.array([6,12,9], np.int32)
    targets = np.array([1,10,7], np.int32)
    expected_old_to_new = [1,2,3,4,5,1,6]
  elif comm.Get_rank() == 1:
    distri = np.array([7,13,13], dtype=np.int32)
    ids     = np.empty(0, dtype=np.int32)
    targets = np.empty(0, dtype=np.int32)
    expected_old_to_new = [7,6,8,9,8,10]

  old_to_new = merge_ids.merge_distributed_ids(distri, ids, targets, comm)
  assert (old_to_new == expected_old_to_new).all()

  if comm.Get_rank() == 0:
    distri = np.array([0,7,13])
    ids     = np.array([6,9])
    targets = np.array([1,7])
    expected_old_to_new = [1,2,3,4,5,1,6]
    expected_signed_old_to_new = [1,2,3,4,5,-1,6]
  elif comm.Get_rank() == 1:
    distri = np.array([7,13,13])
    ids     = np.array([12])
    targets = np.array([10])
    expected_old_to_new = [7,6,8,9,8,10]
    expected_signed_old_to_new = [7,-6,8,9,-8,10]

  old_to_new = merge_ids.merge_distributed_ids(distri, ids, targets, comm)
  assert (old_to_new == expected_old_to_new).all()

  signed_old_to_new = merge_ids.merge_distributed_ids(distri, ids, targets, comm, True)
  assert (signed_old_to_new == expected_signed_old_to_new).all()

