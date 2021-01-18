import pytest
import numpy  as     np
import Converter.Internal as I

from   mpi4py     import MPI
from   maia.utils import parse_yaml_cgns
import maia.distribution as MID

class Test_uniform_distribution_at:
  def test_exact(self):
    n_elt  = 15

    distib = MID.uniform_distribution_at(n_elt,0,3)
    assert distib[0]    == 0
    assert distib[1]    == 5
    distib = MID.uniform_distribution_at(n_elt,1,3)
    assert distib[0]    == 5
    assert distib[1]    == 10
    distib = MID.uniform_distribution_at(n_elt,2,3)
    assert distib[0]    == 10
    assert distib[1]    == 15

  def test_inexact(self):
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

def test_clean_distribution_info():
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t [[27],[8],[0]]:
    Ngon Elements_t [22,0]:
      ElementConnectivity DataArray_t [1,2,3,4]:
      ElementConnectivity#Size DataArray_t [12]:
      :CGNS#Distribution UserDefinedData_t:
    ZBC ZoneBC_t:
      bc1 BC_t "Farfield":
        PointList IndexArray_t [[1,2]]:
        PointList#Size IndexArray_t [4]:
        :CGNS#Distribution UserDefinedData_t:
          Distribution DataArray_t [0,2,4]:
        bcds BCDataSet_t:
          PointList#Size IndexArray_t [2]:
          :CGNS#Distribution UserDefinedData_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [1,4,7,10]:
        PointListDonor IndexArray_t [13,16,7,10]:
        PointList#Size IndexArray_t [8]:
    :CGNS#Distribution UserDefinedData_t:
"""
  dist_tree = parse_yaml_cgns.to_complete_pytree(yt)
  MID.distribution_tree.clean_distribution_info(dist_tree)
  assert I.getNodeFromName(dist_tree, ':CGNS#Distribution') is None
  assert I.getNodeFromName(dist_tree, 'PointList#Size') is None
  assert len(I.getNodesFromName(dist_tree, 'PointList')) == 2
  assert I.getNodeFromName(dist_tree, 'ElementConnectivity#Size') is None

@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
def test_uniform_int32(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return

  n_elt  = np.int32(10)
  distib = MID.uniform_distribution(n_elt, sub_comm)

  assert n_elt.dtype == 'int32'
  assert isinstance(distib, np.ndarray)
  assert distib.shape == (3,)
  assert distib[0]    == 0
  assert distib[1]    == 10
  assert distib[2]    == n_elt


@pytest.mark.mpi(min_size=1)
@pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
def test_uniform_int64(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return

  n_elt  = np.int64(10)
  distib = MID.uniform_distribution(n_elt, sub_comm)

  assert n_elt.dtype == 'int64'
  assert isinstance(distib, np.ndarray)
  assert distib.shape == (3,)
  assert distib[0]    == 0
  assert distib[1]    == 10
  assert distib[2]    == n_elt


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("sub_comm", [2], indirect=['sub_comm'])
def test_uniform_int64_2p(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return

  n_elt  = np.int64(11)
  distib = MID.uniform_distribution(n_elt, sub_comm)

  pytest.assert_mpi(sub_comm, 0, n_elt.dtype == 'int64'          )
  pytest.assert_mpi(sub_comm, 0, isinstance(distib, np.ndarray) )
  pytest.assert_mpi(sub_comm, 0, distib.shape == (3,)            )
  pytest.assert_mpi(sub_comm, 0, distib[0]    == 0               )
  pytest.assert_mpi(sub_comm, 0, distib[1]    == 6               )
  pytest.assert_mpi(sub_comm, 0, distib[2]    == n_elt           )

  pytest.assert_mpi(sub_comm, 1, n_elt.dtype == 'int64'          )
  pytest.assert_mpi(sub_comm, 1, isinstance(distib, np.ndarray) )
  pytest.assert_mpi(sub_comm, 1, distib.shape == (3,)            )
  pytest.assert_mpi(sub_comm, 1, distib[0]    == 6               )
  pytest.assert_mpi(sub_comm, 1, distib[1]    == 11              )
  pytest.assert_mpi(sub_comm, 1, distib[2]    == n_elt           )
