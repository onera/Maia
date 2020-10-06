import pytest
from mpi4py import MPI

# --------------------------------------------------------------------------
@pytest.mark.bnr_lvl0
def test_filter_1():
  comm = MPI.COMM_WORLD
  assert comm.size > 0

  print(test_filter_1)

# --------------------------------------------------------------------------
# @pytest.mark.usefixtures('sub_comm', params=[1, 2, 3])
# @pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
def test_filter_2(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    print("ooox"*10)
    return

  assert( sub_comm.size == 1)
  # assert( sub_comm.size == 2)

  pytest.assert_mpi(sub_comm, 0, sub_comm.rank == 0)

  pass

# --------------------------------------------------------------------------
@pytest.mark.parametrize("sub_comm", [2], indirect=['sub_comm'])
def test_filter_all_reduce(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    print("ooox"*10)
    return

  assert( sub_comm.size == 2)
  res = sub_comm.allreduce(sub_comm.size, op=MPI.SUM)

  pytest.assert_mpi(sub_comm, 0, sub_comm.rank == 0)
  pytest.assert_mpi(sub_comm, 1, res           == 4)
  pytest.assert_mpi(sub_comm, 0, res           == 4)
