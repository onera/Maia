import pytest
from pytest_mpi_check._decorator import mark_mpi_test
from mpi4py import MPI

@mark_mpi_test(1)
def test_mark_mpi_decorator(sub_comm):
  print("\n\nrank = ",sub_comm.Get_rank(),", n_rank = ",sub_comm.Get_size(),"\n\n")

@mark_mpi_test([1,2])
def test_mark_mpi_decorator_with_list(sub_comm):
  print("\n\nrank = ",sub_comm.Get_rank(),", n_rank = ",sub_comm.Get_size(),"\n\n")

@pytest.mark.parametrize("val",[3,4])
@mark_mpi_test(2)
def test_mark_mpi_decorator_with_other_pytest_deco(sub_comm,val):
  print("\n\nrank = ",sub_comm.Get_rank(),", n_rank = ",sub_comm.Get_size()," , val = ",val,"\n\n")

@pytest.mark.parametrize("val",[3,4])
@mark_mpi_test([1,2])
def test_mark_mpi_decorator_with_list_and_other_pytest_deco(sub_comm,val):
  print("\n\nrank = ",sub_comm.Get_rank(),", n_rank = ",sub_comm.Get_size()," , val = ",val,"\n\n")


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

# @pytest.mark.parametrize("val",[-30,-40])
@mark_mpi_test(2)
def test_simple_mpi_param_and_val_t3(sub_comm):
  """
  """
  comm   = MPI.COMM_WORLD
  print("\n\n test_simple_mpi_param_and_val_t2 :: rank = ",sub_comm.Get_rank(),", n_rank = ",sub_comm.Get_size(), "on initial comm :: ", comm.Get_rank(), "/", comm.Get_size(), "\n\n")
  assert 0 == 0

  # pytest.assert_mpi(sub_comm, 0, sub_comm.rank == 1)
  # pytest.assert_mpi(sub_comm, 1, sub_comm.rank == 0)
