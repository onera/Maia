import pytest
import numpy  as     NPY
from   mpi4py import MPI

import maia.distribution as MID

@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
def test_uniform_int32(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return

  nelmt  = NPY.int32(10)
  distib = MID.uniform_distribution(nelmt, sub_comm)

  assert nelmt.dtype == 'int32'
  assert isinstance(distib, NPY.ndarray)
  assert distib.shape == (3,)
  assert distib[0]    == 0
  assert distib[1]    == 10
  assert distib[2]    == nelmt


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
def test_uniform_int64(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return

  nelmt  = NPY.int64(10)
  distib = MID.uniform_distribution(nelmt, sub_comm)

  assert nelmt.dtype == 'int64'
  assert isinstance(distib, NPY.ndarray)
  assert distib.shape == (3,)
  assert distib[0]    == 0
  assert distib[1]    == 10
  assert distib[2]    == nelmt


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("sub_comm", [2], indirect=['sub_comm'])
def test_uniform_int64_2p(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return

  nelmt  = NPY.int64(10)
  distib = MID.uniform_distribution(nelmt, sub_comm)

  pytest.assert_mpi(sub_comm, 0, nelmt.dtype == 'int64'          )
  pytest.assert_mpi(sub_comm, 0, isinstance(distib, NPY.ndarray) )
  pytest.assert_mpi(sub_comm, 0, distib.shape == (3,)            )
  pytest.assert_mpi(sub_comm, 0, distib[0]    == 0               )
  pytest.assert_mpi(sub_comm, 0, distib[1]    == 5               )
  pytest.assert_mpi(sub_comm, 0, distib[2]    == nelmt           )

  pytest.assert_mpi(sub_comm, 1, nelmt.dtype == 'int64'          )
  pytest.assert_mpi(sub_comm, 1, isinstance(distib, NPY.ndarray) )
  pytest.assert_mpi(sub_comm, 1, distib.shape == (3,)            )
  pytest.assert_mpi(sub_comm, 1, distib[0]    == 5               )
  pytest.assert_mpi(sub_comm, 1, distib[1]    == 10              )
  pytest.assert_mpi(sub_comm, 1, distib[2]    == nelmt           )
