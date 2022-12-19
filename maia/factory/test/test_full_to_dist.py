import pytest 
from pytest_mpi_check._decorator import mark_mpi_test

import maia.pytree        as PT

from maia.pytree.yaml   import parse_yaml_cgns

from maia.factory import full_to_dist

@mark_mpi_test(2)
def test_distribute_pl_node(sub_comm):
  yt = """
  bc BC_t:
    PointList IndexArray_t [[10,14,12,16]]:
    GridLocation GridLocation_t "CellCenter":
    BCDataSet BCDataSet_t:
      BCData BCData_t:
        Data DataArray_t [1,2,11,11]:
  """
  bc = parse_yaml_cgns.to_node(yt)
  dist_bc = full_to_dist.distribute_pl_node(bc, sub_comm)
  assert PT.get_node_from_path(dist_bc, ':CGNS#Distribution/Index') is not None
  assert PT.get_node_from_path(dist_bc, 'BCDataSet/:CGNS#Distribution/Index') is None
  assert PT.get_node_from_name(dist_bc, 'PointList')[1].shape == (1,2)
  assert PT.get_node_from_name(dist_bc, 'Data')[1].shape == (2,)

  yt = """
  bc BC_t:
    PointList IndexArray_t [[10,14,12,16]]:
    GridLocation GridLocation_t "CellCenter":
    BCDataSet BCDataSet_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[100,200]]:
      BCData BCData_t:
        Data DataArray_t [1,2]:
  """
  bc = parse_yaml_cgns.to_node(yt)
  dist_bc = full_to_dist.distribute_pl_node(bc, sub_comm)
  assert PT.get_node_from_path(dist_bc, ':CGNS#Distribution/Index') is not None
  assert PT.get_node_from_path(dist_bc, 'BCDataSet/:CGNS#Distribution/Index') is not None
  assert PT.get_node_from_name(dist_bc, 'PointList')[1].shape == (1,2)
  assert PT.get_node_from_name(dist_bc, 'Data')[1].shape == (1,)
  assert PT.get_node_from_path(dist_bc, 'BCDataSet/PointList')[1].shape == (1,1)

@mark_mpi_test(3)
def test_distribute_data_node(sub_comm):
  rank = sub_comm.Get_rank()
  fs = PT.new_FlowSolution(loc='CellCenter')
  data1 = PT.new_DataArray('Data1', [2,4,6,8,10,12,14], parent=fs)
  data2 = PT.new_DataArray('Data2', [-1,-2,-3,-4,-5,-6,-7], parent=fs)

  dist_fs = full_to_dist.distribute_data_node(fs, sub_comm)
  distri_f = [0,3,5,7]

  assert (PT.get_node_from_name(dist_fs, 'Data1')[1] == data1[1][distri_f[rank] : distri_f[rank+1]]).all()
  assert (PT.get_node_from_name(dist_fs, 'Data2')[1] == data2[1][distri_f[rank] : distri_f[rank+1]]).all()

@mark_mpi_test(2)
def test_distribute_element(sub_comm):
  yt = """
  Element Elements_t [5,0]:
    ElementRange IndexRange_t [16,20]:
    ElementConnectivity DataArray_t [4,1,3, 8,2,1, 9,7,4, 11,4,2, 10,4,1]:
  """
  elem = parse_yaml_cgns.to_node(yt)
  dist_elem = full_to_dist.distribute_element_node(elem, sub_comm)

  assert (PT.Element.Range(dist_elem) == [16,20]).all()
  assert PT.get_node_from_path(dist_elem, ':CGNS#Distribution/Element') is not None
  if sub_comm.Get_rank() == 0:
    assert (PT.get_node_from_name(dist_elem, 'ElementConnectivity')[1] == [4,1,3, 8,2,1, 9,7,4]).all()
  else:
    assert (PT.get_node_from_name(dist_elem, 'ElementConnectivity')[1] == [11,4,2, 10,4,1]).all()

  for elt_type in ['22', '20']:
    yt = f"""
    Element Elements_t [{elt_type},0]:
      ElementRange IndexRange_t [1,4]:
      ElementStartOffset DataArray_t [0,4,8,11,16]:
      ElementConnectivity DataArray_t [4,1,3,8, 8,2,3,1, 9,7,4, 11,4,2,10,1]:
      ParentElements DataArray_t [[1,0], [2,3], [2,0], [3,0]]:
    """
    elem = parse_yaml_cgns.to_node(yt)
    dist_elem = full_to_dist.distribute_element_node(elem, sub_comm)

    assert (PT.Element.Range(dist_elem) == [1,4]).all()
    assert PT.get_node_from_path(dist_elem, ':CGNS#Distribution/Element') is not None
    assert PT.get_node_from_path(dist_elem, ':CGNS#Distribution/ElementConnectivity') is not None
    if sub_comm.Get_rank() == 0:
      assert (PT.get_child_from_name(dist_elem, 'ElementConnectivity')[1] == [4,1,3,8, 8,2,3,1]).all()
      assert (PT.get_child_from_name(dist_elem, 'ElementStartOffset')[1] == [0,4,8]).all()
      assert (PT.get_child_from_name(dist_elem, 'ParentElements')[1] == [[1,0],[2,3]]).all()
    else:
      assert (PT.get_child_from_name(dist_elem, 'ElementConnectivity')[1] == [9,7,4, 11,4,2,10,1]).all()
      assert (PT.get_child_from_name(dist_elem, 'ElementStartOffset')[1] == [8,11,16]).all()
      assert (PT.get_child_from_name(dist_elem, 'ParentElements')[1] == [[2,0],[3,0]]).all()

@mark_mpi_test(2)
def test_distribute_tree(sub_comm):
  yt = """
  Zone Zone_t [[18, 4, 0]]:
    ZoneType ZoneType_t "Unstructured":
    Element Elements_t [22,0]:
      ElementRange IndexRange_t [1,4]:
      ElementStartOffset DataArray_t [0,4,8,11,16]:
      ElementConnectivity DataArray_t [4,1,3,8, 8,2,3,1, 9,7,4, 11,4,2,10,1]:
      ParentElements DataArray_t [[1,0], [2,3], [2,0], [3,0]]:
    ZoneBC ZoneBC_t:
      bc BC_t:
        PointList IndexArray_t [[10,14,12,16]]:
        GridLocation GridLocation_t "CellCenter":
        BCDataSet BCDataSet_t:
          GridLocation GridLocation_t "Vertex":
          PointList IndexArray_t [[100,200]]:
          BCData BCData_t:
            Data DataArray_t [1,2]:
    SolNoPl FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      Array DataArray_t [2.2, 3.3, 1.1, 0.0]:
    SolPl FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[3]]:
      Array DataArray_t [1000]:
    UnrelatedZSR ZoneSubRegion_t:
      PointList IndexArray_t [[2, 1]]:
      Array DataArray_t [500, 550]:
    RelatedZSR ZoneSubRegion_t:
      Array DataArray_t [21, 12, 20, 12]:
      BCRegionName Descriptor_t "bc":
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  dist_tree = full_to_dist.distribute_tree(tree, sub_comm)

  zone = PT.get_all_Zone_t(dist_tree)[0]
  if sub_comm.Get_rank() == 0:
    assert (PT.get_node_from_name(zone, 'ElementConnectivity')[1] == [4,1,3,8, 8,2,3,1]).all()
    assert (PT.get_node_from_path(zone, 'SolPl/Array')[1] == [1000]).all()
  if sub_comm.Get_rank() == 1:
    assert (PT.get_node_from_name(zone, 'ElementConnectivity')[1] == [9,7,4, 11,4,2,10,1]).all()
    assert (PT.get_node_from_path(zone, 'SolPl/Array')[1].size == 0)
  assert len(PT.get_nodes_from_name(zone, ':CGNS#Distribution')) == 6

