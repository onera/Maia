import pytest
from pytest_mpi_check._decorator import mark_mpi_test

from mpi4py import MPI

def test_simple_seq():
  """
  """
  print("test_simple_seq")
  assert 0 == 0

def test_simple_mpi2proc():
  """
  A nettoyer je pense qu'il est executé n procs fois
  """
  print("test_simple_mpi2proc")

  assert 0 == 0


@mark_mpi_test([1,2])
def test_simple_mpi_param(sub_comm):
  """
  """
  comm   = MPI.COMM_WORLD
  print("\n\n test_simple_mpi_param :: rank = ",sub_comm.Get_rank(),", n_rank = ",sub_comm.Get_size(), "on initial comm :: ", comm.Get_rank(), "/", comm.Get_size(), "\n\n")
  assert 0 == 0
  # assert 0 == 1
  # assert 1 == 0
  pytest.assert_mpi(sub_comm, 0, sub_comm.rank == 1)
  pytest.assert_mpi(sub_comm, 1, sub_comm.rank == 0)

# @pytest.mark.parametrize("val",[-3,-4])
# @mark_mpi_test([1,2])
# def test_simple_mpi_param_and_val(sub_comm, val):
#   """
#   """
#   comm   = MPI.COMM_WORLD
#   print("\n\n test_simple_mpi_param_and_val :: rank = ",sub_comm.Get_rank(),", n_rank = ",sub_comm.Get_size(), "on initial comm :: ", comm.Get_rank(), "/", comm.Get_size(), "\n\n")
#   assert 0 == 0
#   assert 0 == 0
#   assert 0 == 0
#   assert 0 == 0
#   assert 1 == 0

# @pytest.mark.parametrize("val",[-30,-40])
@mark_mpi_test(2)
def test_simple_mpi_param_and_val_t2(sub_comm):
  """
  """
  comm   = MPI.COMM_WORLD
  print("\n\n test_simple_mpi_param_and_val_t2 :: rank = ",sub_comm.Get_rank(),", n_rank = ",sub_comm.Get_size(), "on initial comm :: ", comm.Get_rank(), "/", comm.Get_size(), "\n\n")
  assert 0 == 0

  # pytest.assert_mpi(sub_comm, 0, sub_comm.rank == 1)
  # pytest.assert_mpi(sub_comm, 1, sub_comm.rank == 0)
