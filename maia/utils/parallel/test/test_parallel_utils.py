from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np
from mpi4py import MPI

from maia.utils.parallel import utils

@mark_mpi_test(3)
def test_partial_to_full_distribution(sub_comm):
  if sub_comm.Get_rank() == 0:
    partial_distrib_32 = np.array([0, 25, 75], dtype=np.int32)
    partial_distrib_64 = np.array([0, 25, 75], dtype=np.int64)
    partial_distrib_hole = np.array([0, 25, 75])
    partial_distrib_void = np.array([0, 0, 0])
  if sub_comm.Get_rank() == 1:
    partial_distrib_32 = np.array([25, 55, 75], dtype=np.int32)
    partial_distrib_64 = np.array([25, 55, 75], dtype=np.int64)
    partial_distrib_hole = np.array([25, 25, 75])
    partial_distrib_void = np.array([0, 0, 0])
  if sub_comm.Get_rank() == 2:
    partial_distrib_32 = np.array([55, 75, 75], dtype=np.int32)
    partial_distrib_64 = np.array([55, 75, 75], dtype=np.int64)
    partial_distrib_hole = np.array([25, 75, 75])
    partial_distrib_void = np.array([0, 0, 0])

  full_distri_32 = utils.partial_to_full_distribution(partial_distrib_32, sub_comm)
  assert full_distri_32.dtype == np.int32
  assert (full_distri_32 == [0,25,55,75]).all()
  full_distri_64 = utils.partial_to_full_distribution(partial_distrib_64, sub_comm)
  assert full_distri_64.dtype == np.int64
  assert (full_distri_64 == [0,25,55,75]).all()
  full_distri_hole = utils.partial_to_full_distribution(partial_distrib_hole, sub_comm)
  assert (full_distri_hole == [0,25,25,75]).all()
  full_distri_void = utils.partial_to_full_distribution(partial_distrib_void, sub_comm)
  assert (full_distri_void == [0,0,0,0]).all()
