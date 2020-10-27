import pytest
import pytest_check as check

# --------------------------------------------------------------------------
# @pytest.mark.parametrize("memorder", 1, 2, 3, ids=lambda fixture_value:fixture_value.name)
@pytest.mark.parametrize("memorder", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float", "double", "int"])
def test_memory_key(dtype, memorder):
  print("dtype::"   ,dtype)
  print("memorder::",memorder)

# --------------------------------------------------------------------------
@pytest.mark.mpi_test(comm_size=1)
def test_filter_5():
  print("xxx"*100)

# --------------------------------------------------------------------------
@pytest.mark.mpi_test(comm_size=2)
def test_filter_6():
  print("xxx"*100)

# --------------------------------------------------------------------------
# @pytest.mark.parametrize("make_sub_comm", [1, 2], indirect=['make_sub_comm'])
# def test_filter_7(make_sub_comm):
#   print("yyy"*100, make_sub_comm.size)
#   if(make_sub_comm.size == 2):
#     # assert mamke_sub_comm.size == 2
#     # assert make_sub_comm.rank == 1
#     # assert make_sub_comm.rank == 1
#     # assert make_sub_comm.rank == 1
#     # assert make_sub_comm.rank == 1
#     check.equal(make_sub_comm.size, 2, " hehe" )
#     check.equal(make_sub_comm.rank, 1, " hehe" )
#     check.equal(make_sub_comm.rank, 1, " hehe" )
#     check.equal(make_sub_comm.rank, 1, " hehe" )
#     check.equal(make_sub_comm.rank, 1, " hehe" )
