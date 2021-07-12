import pytest 
from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal      as I

from maia.utils        import parse_yaml_cgns
from maia.sids         import sids

from maia.distribution import distribute_nodes as DN

@mark_mpi_test(2)
def test_distribute_pl_node(sub_comm):
  yt = """
  bc BC_t:
    PointList IndexArray_t [[10,14,12,16]]:
    GridLocation GridLocation_t "CellCenter":
    BCDataSet BCDataSet_t:
      BCData BCData_t:
        Data DataArray_t [[1,2,11,11]]:
  """
  bc = parse_yaml_cgns.to_node(yt)
  dist_bc = DN.distribute_pl_node(bc, sub_comm)
  assert I.getNodeFromPath(dist_bc, ':CGNS#Distribution/Index') is not None
  assert I.getNodeFromPath(dist_bc, 'BCDataSet/:CGNS#Distribution/Index') is None
  assert I.getNodeFromName(dist_bc, 'PointList')[1].shape == (1,2)
  assert I.getNodeFromName(dist_bc, 'Data')[1].shape == (1,2)

  yt = """
  bc BC_t:
    PointList IndexArray_t [[10,14,12,16]]:
    GridLocation GridLocation_t "CellCenter":
    BCDataSet BCDataSet_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[100,200]]:
      BCData BCData_t:
        Data DataArray_t [[1,2]]:
  """
  bc = parse_yaml_cgns.to_node(yt)
  dist_bc = DN.distribute_pl_node(bc, sub_comm)
  assert I.getNodeFromPath(dist_bc, ':CGNS#Distribution/Index') is not None
  assert I.getNodeFromPath(dist_bc, 'BCDataSet/:CGNS#Distribution/Index') is not None
  assert I.getNodeFromName(dist_bc, 'PointList')[1].shape == (1,2)
  assert I.getNodeFromName(dist_bc, 'Data')[1].shape == (1,1)
  assert I.getNodeFromPath(dist_bc, 'BCDataSet/PointList')[1].shape == (1,1)

@mark_mpi_test(3)
def test_distribute_data_node(sub_comm):
  rank = sub_comm.Get_rank()
  fs = I.newFlowSolution(gridLocation='CellCenter')
  data1 = I.newDataArray('Data1', [2,4,6,8,10,12,14], fs)
  data2 = I.newDataArray('Data2', [-1,-2,-3,-4,-5,-6,-7], fs)

  dist_fs = DN.distribute_data_node(fs, sub_comm)
  distri_f = [0,3,5,7]

  assert (I.getNodeFromName(dist_fs, 'Data1')[1] == data1[1][distri_f[rank] : distri_f[rank+1]]).all()
  assert (I.getNodeFromName(dist_fs, 'Data2')[1] == data2[1][distri_f[rank] : distri_f[rank+1]]).all()

@mark_mpi_test(2)
def test_distribute_element(sub_comm):
  yt = """
  Element Elements_t [5,0]:
    ElementRange IndexRange_t [16,20]:
    ElementConnectivity DataArray_t [4,1,3, 8,2,1, 9,7,4, 11,4,2, 10,4,1]:
  """
  elem = parse_yaml_cgns.to_node(yt)
  dist_elem = DN.distribute_element_node(elem, sub_comm)

  assert (sids.ElementRange(dist_elem) == [16,20]).all()
  assert I.getNodeFromPath(dist_elem, ':CGNS#Distribution/Element') is not None
  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromName(dist_elem, 'ElementConnectivity')[1] == [4,1,3, 8,2,1, 9,7,4]).all()
  else:
    assert (I.getNodeFromName(dist_elem, 'ElementConnectivity')[1] == [11,4,2, 10,4,1]).all()

  yt = """
  Element Elements_t [22,0]:
    ElementRange IndexRange_t [1,4]:
    ElementStartOffset DataArray_t [0,4,8,11,16]:
    ElementConnectivity DataArray_t [4,1,3,8, 8,2,3,1, 9,7,4, 11,4,2,10,1]:
    ParentElements DataArray_t [[1,0], [2,3], [2,0], [3,0]]:
  """
  elem = parse_yaml_cgns.to_node(yt)
  dist_elem = DN.distribute_element_node(elem, sub_comm)

  assert (sids.ElementRange(dist_elem) == [1,4]).all()
  assert I.getNodeFromPath(dist_elem, ':CGNS#Distribution/Element') is not None
  assert I.getNodeFromPath(dist_elem, ':CGNS#Distribution/ElementConnectivity') is not None
  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromName(dist_elem, 'ElementConnectivity')[1] == [4,1,3,8, 8,2,3,1]).all()
    assert (I.getNodeFromName(dist_elem, 'ElementStartOffset')[1] == [0,4,8]).all()
    assert (I.getNodeFromName(dist_elem, 'ParentElements')[1] == [[1,0],[2,3]]).all()
  else:
    assert (I.getNodeFromName(dist_elem, 'ElementConnectivity')[1] == [9,7,4, 11,4,2,10,1]).all()
    assert (I.getNodeFromName(dist_elem, 'ElementStartOffset')[1] == [8,11,16]).all()
    assert (I.getNodeFromName(dist_elem, 'ParentElements')[1] == [[2,0],[3,0]]).all()
