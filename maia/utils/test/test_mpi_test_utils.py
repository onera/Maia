import pytest
from maia.utils.mpi_test_utils import mark_mpi_test

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
