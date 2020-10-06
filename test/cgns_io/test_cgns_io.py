import pytest

# --------------------------------------------------------------------------
@pytest.mark.bnr_lvl0
def test_filter_1():
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  assert comm.size > 0

  print(test_filter_1)

# --------------------------------------------------------------------------
# @pytest.mark.use_fixtures('cleandir')
# def test_filter_1():
#   pass
