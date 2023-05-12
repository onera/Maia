import pytest
import pytest_parallel
from mpi4py import MPI

@pytest_parallel.mark.parallel(1)
def test_mark_mpi_decorator(comm):
  print("\n\n", MPI.COMM_WORLD.Get_rank(), " -> ", comm, MPI.COMM_NULL)
  assert(comm != MPI.COMM_NULL)
  print("\n\nrank = ",comm.Get_rank(),", n_rank = ",comm.Get_size(), "  ", MPI.COMM_WORLD.Get_rank(),"\n\n")

@pytest_parallel.mark.parallel([1,2])
def test_mark_mpi_decorator_with_list(comm):
  # print("\n\n", MPI.COMM_WORLD.Get_rank(), " -> ", comm)
  print("\n\nrank = ",comm.Get_rank(),", n_rank = ",comm.Get_size(), "  ", MPI.COMM_WORLD.Get_rank(),"\n\n")


@pytest.mark.parametrize("val",[3,4])
@pytest_parallel.mark.parallel(2)
def test_mark_mpi_decorator_with_other_pytest_deco(comm,val):
  print("\n\nrank = ",comm.Get_rank(),", n_rank = ",comm.Get_size()," , val = ",val,"\n\n")

@pytest.mark.parametrize("val",[3,4])
@pytest_parallel.mark.parallel([1,2])
def test_mark_mpi_decorator_with_list_and_other_pytest_deco(comm,val):
  print("\n\nrank = ",comm.Get_rank(),", n_rank = ",comm.Get_size()," , val = ",val,"\n\n")


# @pytest.mark.parametrize("val",[-30,-40])
@pytest_parallel.mark.parallel(2)
def test_simple_mpi_param_and_val_t2(comm):
  """
  """
  comm   = MPI.COMM_WORLD
  print("\n\n test_simple_mpi_param_and_val_t2 :: rank = ",comm.Get_rank(),", n_rank = ",comm.Get_size(), "on initial comm :: ", comm.Get_rank(), "/", comm.Get_size(), "\n\n")
  assert 0 == 0


# @pytest.mark.parametrize("val",[-30,-40])
@pytest_parallel.mark.parallel(2)
def test_simple_mpi_param_and_val_t3(comm):
  """
  """
  comm   = MPI.COMM_WORLD
  print("\n\n test_simple_mpi_param_and_val_t2 :: rank = ",comm.Get_rank(),", n_rank = ",comm.Get_size(), "on initial comm :: ", comm.Get_rank(), "/", comm.Get_size(), "\n\n")
  assert 0 == 0
