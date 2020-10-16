import pytest
import numpy  as     np
from   mpi4py import MPI

import maia.distribution as MID

# TODO SUBCASE or equivalent in pytest?
class Test_uniform_distribution_at
def test_uniform_distribution_at(int_type):
  def test_exact():
    n_elt  = 15

    distib = MID.uniform_distribution_at(n_elt,0,3)
    assert distib[0]    == 0
    assert distib[1]    == 6
    distib = MID.uniform_distribution_at(n_elt,1,3)
    assert distib[0]    == 5
    assert distib[1]    == 10
    distib = MID.uniform_distribution_at(n_elt,2,3)
    assert distib[0]    == 10
    assert distib[1]    == 15

  def test_inexact():
    n_elt  = 17

    distib = MID.uniform_distribution_at(n_elt,0,3)
    assert distib[0]    == 0
    assert distib[1]    == 6
    distib = MID.uniform_distribution_at(n_elt,1,3)
    assert distib[0]    == 6
    assert distib[1]    == 12
    distib = MID.uniform_distribution_at(n_elt,2,3)
    assert distib[0]    == 12
    assert distib[1]    == 17


#@pytest.mark.mpi(min_size=1)
#@pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
#def test_uniform_int32(sub_comm):
#  if(sub_comm == MPI.COMM_NULL):
#    return
#
#  n_elt  = np.int32(10)
#  distib = MID.uniform_distribution(n_elt, sub_comm)
#
#  assert n_elt.dtype == 'int32'
#  assert isinstance(distib, np.ndarray)
#  assert distib.shape == (3,)
#  assert distib[0]    == 0
#  assert distib[1]    == 10
#  assert distib[2]    == n_elt
#
#
#@pytest.mark.mpi(min_size=1)
#@pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
#def test_uniform_int64(sub_comm):
#  if(sub_comm == MPI.COMM_NULL):
#    return
#
#  n_elt  = np.int64(10)
#  distib = MID.uniform_distribution(n_elt, sub_comm)
#
#  assert n_elt.dtype == 'int64'
#  assert isinstance(distib, np.ndarray)
#  assert distib.shape == (3,)
#  assert distib[0]    == 0
#  assert distib[1]    == 10
#  assert distib[2]    == n_elt
#
#
#@pytest.mark.mpi(min_size=2)
#@pytest.mark.parametrize("sub_comm", [2], indirect=['sub_comm'])
#def test_uniform_int64_2p(sub_comm):
#  if(sub_comm == MPI.COMM_NULL):
#    return
#
#  n_elt  = np.int64(11)
#  distib = MID.uniform_distribution(n_elt, sub_comm)
#
#  pytest.assert_mpi(sub_comm, 0, n_elt.dtype == 'int64'          )
#  pytest.assert_mpi(sub_comm, 0, isinstance(distib, np.ndarray) )
#  pytest.assert_mpi(sub_comm, 0, distib.shape == (3,)            )
#  pytest.assert_mpi(sub_comm, 0, distib[0]    == 0               )
#  pytest.assert_mpi(sub_comm, 0, distib[1]    == 6               )
#  pytest.assert_mpi(sub_comm, 0, distib[2]    == n_elt           )
#
#  pytest.assert_mpi(sub_comm, 1, n_elt.dtype == 'int64'          )
#  pytest.assert_mpi(sub_comm, 1, isinstance(distib, np.ndarray) )
#  pytest.assert_mpi(sub_comm, 1, distib.shape == (3,)            )
#  pytest.assert_mpi(sub_comm, 1, distib[0]    == 6               )
#  pytest.assert_mpi(sub_comm, 1, distib[1]    == 11              )
#  pytest.assert_mpi(sub_comm, 1, distib[2]    == n_elt           )
