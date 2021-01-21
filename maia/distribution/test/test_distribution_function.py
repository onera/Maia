import pytest
import numpy      as np
import mpi4py.MPI as MPI

import Converter.Internal as I
import maia.distribution.distribution_function as MID

def test_uniform_distribution_at():
  assert MID.uniform_distribution_at(15,0,3) == (0,5)
  assert MID.uniform_distribution_at(15,1,3) == (5,10)
  assert MID.uniform_distribution_at(15,2,3) == (10,15)

  assert MID.uniform_distribution_at(17,0,3) == (0,6)
  assert MID.uniform_distribution_at(17,1,3) == (6,12)
  assert MID.uniform_distribution_at(17,2,3) == (12,17)


@pytest.mark.mpi(min_size=3)
@pytest.mark.parametrize("sub_comm", [3], indirect=['sub_comm'])
def test_uniform_distribution(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return

  distrib = MID.uniform_distribution(np.int32(17), sub_comm)
  assert isinstance(distrib, np.ndarray)
  assert distrib.dtype == 'int32'
  assert (distrib[0:2] == MID.uniform_distribution_at(\
      17, sub_comm.Get_rank(), sub_comm.Get_size())).all()
  assert distrib[2] == 17

  distrib = MID.uniform_distribution(np.int64(17), sub_comm)
  assert isinstance(distrib, np.ndarray)
  assert distrib.dtype == 'int64'
  assert (distrib[0:2] == MID.uniform_distribution_at(\
      17, sub_comm.Get_rank(), sub_comm.Get_size())).all()
  assert distrib[2] == 17

@pytest.mark.mpi(min_size=3)
@pytest.mark.parametrize("sub_comm", [3], indirect=['sub_comm'])
def test_create_distribution_node(sub_comm):
  if(sub_comm == MPI.COMM_NULL):
    return

  node = I.createNode('ParentNode', 'AnyType_t')
  MID.create_distribution_node(100, sub_comm, 'MyDistribution', node)

  distri_ud   = I.getNodeFromName(node, ':CGNS#Distribution')
  assert distri_ud is not None
  assert I.getType(distri_ud) == 'UserDefinedData_t'
  distri_node = I.getNodeFromName1(distri_ud, 'MyDistribution')
  assert distri_node is not None
  assert (I.getValue(distri_node) == MID.uniform_distribution(100, sub_comm)).all()
  assert I.getType(distri_node) == 'DataArray_t'


def test_create_distribution_node_from_distrib():
  distri = np.array([10,20,30])
  node = I.createNode('ParentNode', 'AnyType_t')
  MID.create_distribution_node_from_distrib('MyDistribution', node, distri)
  distri_ud   = I.getNodeFromName(node, ':CGNS#Distribution')
  assert distri_ud is not None
  assert I.getType(distri_ud) == 'UserDefinedData_t'
  distri_node = I.getNodeFromName1(distri_ud, 'MyDistribution')
  assert distri_node is not None
  assert (I.getValue(distri_node) == distri).all()
  assert I.getType(distri_node) == 'DataArray_t'
