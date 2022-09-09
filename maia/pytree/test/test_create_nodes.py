import pytest
import numpy as np
from maia.pytree import create_nodes
from maia.pytree import nodes_attr as NA
from maia.pytree import walk
from maia.utils.yaml     import parse_yaml_cgns
from maia.pytree.compare import is_same_tree

def test_new_node():
  node = create_nodes.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  assert NA.get_name(node) == 'match'
  assert NA.get_label(node) == 'GridConnectivity1to1_t'
  assert NA.get_value(node) == 'OtherZone'
  assert NA.get_children(node) == []

  child = create_nodes.new_node('Transform', 'Transform_t', [1,2,3], parent=node)
  assert (NA.get_value(child) == np.array([1,2,3])).all()
  assert NA.get_children(node)[0] == child

  with pytest.warns(RuntimeWarning):
    create_nodes.new_node('ANodeNameDefinitivelyTooLongForTheCGNSStandard', 'Transform_t', [1,2,3], parent=node)

def test_update_node():
  node = ['Node', None, [], 'UserDefined_t']
  create_nodes.update_node(node, 'NewName', value=[6.])
  assert NA.get_name(node) ==  'NewName'
  assert NA.get_value(node) == np.array([6.])
  create_nodes.update_node(node, ..., 'BC_t')
  assert NA.get_name(node)  == 'NewName'
  assert NA.get_label(node) == 'BC_t'
  with pytest.raises(Exception):
    create_nodes.update_node(node, children=12)

# def test_create_child():
  # node  = create_nodes.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  # child = create_nodes.create_child(node, 'Transform', 'Transform_t', [1,2,3])
  # assert NA.get_children(node)[0] == child
  # otherchild = create_nodes.create_child(node, 'Transform', label='UserDefinedData_t')
  # assert len(NA.get_children(node)) == 1
  # assert NA.get_children(node)[0] == otherchild
  # assert NA.get_value(otherchild) is None #Old values are erased

def test_new_child():
  node = create_nodes.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  child = create_nodes.new_child(node, 'Transform', 'Transform_t', [1,2,3])
  assert NA.get_children(node)[0] == child
  with pytest.raises(Exception):
    child = create_nodes.new_child(node, 'Transform', 'Transform_t', [3,2,1])

def test_update_child():
  node = create_nodes.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  child = create_nodes.update_child(node, 'Transform', 'Transform_t', [1,2,3])
  assert NA.get_children(node)[0] == child
  child = create_nodes.update_child(node, 'Transform', value=[3,2,1])
  assert len(NA.get_children(node)) == 1
  assert NA.get_children(node)[0] == child
  assert np.array_equal(NA.get_value(child), [3,2,1])
  assert NA.get_label(child) == 'Transform_t' #Old value are preserved

def test_shallow_copy():
  node = create_nodes.new_node('match', 'GridConnectivity1to1_t')
  child = create_nodes.update_child(node, 'Transform', 'Transform_t', [1,2,3])

  copied   = create_nodes.shallow_copy(node)
  copied_c = NA.get_children(copied)[0]
  assert NA.get_name(copied) == "match"
  assert np.array_equal(NA.get_value(copied_c), [1,2,3])

  NA.set_label(copied, "GridConnectivity_t")
  assert NA.get_label(node) == 'GridConnectivity1to1_t' #Source label is not modified
  copied_c[1] += 1
  assert np.array_equal(NA.get_value(child), [2,3,4]) #Source value is modified
  copied_c[1] = np.array([1,2,3])
  assert np.array_equal(NA.get_value(child), [2,3,4]) #Hard set break the link

  NA.set_children(copied, [])
  assert len(NA.get_children(copied)) == 0
  assert len(NA.get_children(node)) == 1 # Source structure is not modified

def test_deep_copy():
  node = create_nodes.new_node('match', 'GridConnectivity1to1_t')
  child = create_nodes.update_child(node, 'Transform', 'Transform_t', [1,2,3])

  copied   = create_nodes.deep_copy(node)
  copied_c = NA.get_children(copied)[0]
  assert NA.get_name(copied) == "match"
  assert np.array_equal(NA.get_value(copied_c), [1,2,3])

  NA.set_label(copied, "GridConnectivity_t")
  assert NA.get_label(node) == 'GridConnectivity1to1_t' #Source label is not modified
  copied_c[1] += 1
  assert np.array_equal(NA.get_value(child), [1,2,3]) #Source value is not modified

  NA.set_children(copied, [])
  assert len(NA.get_children(copied)) == 0
  assert len(NA.get_children(node)) == 1 # Source structure is not modified


def test_new_CGNSTree():
  tree = create_nodes.new_CGNSTree(version=3.1)
  expected = parse_yaml_cgns.to_node("""
  CGNSTree CGNSTree_t:
    CGNSLibraryVersion CGNSLibraryVersion_t 3.1:
  """)
  assert is_same_tree(expected, tree)

def test_new_CGNSBase():
  base = create_nodes.new_CGNSBase(phy_dim=2)
  assert np.array_equal(NA.get_value(base), [3,2])

def test_new_Zone():
  zone = create_nodes.new_Zone('SomeZone', type='Unstructured', family='Family', size=[[11, 10, 0]])
  expected = parse_yaml_cgns.to_node("""
  SomeZone Zone_t I4 [[11,10,0]]:
    FamilyName FamilyName_t "Family":
    ZoneType ZoneType_t "Unstructured":
  """)
  assert is_same_tree(expected, zone)
  
  with pytest.raises(AssertionError):
    create_nodes.new_Zone('SomeZone', type='Toto')

def test_new_Elements():
  elem = create_nodes.new_Elements('Tri', 'TRI_3')
  assert np.array_equal(NA.get_value(elem), [5,0])
  elem = create_nodes.new_Elements('Tri', 5)
  assert np.array_equal(NA.get_value(elem), [5,0])
  elem = create_nodes.new_Elements('Tri', [5,1], erange=[1,10], econn=[1,2,7])
  expected = parse_yaml_cgns.to_node("""
  Tri Elements_t I4 [5,1]:
    ElementRange ElementRange_t I4 [1,10]:
    ElementConnectivity DataArray_t I4 [1,2,7]:
  """)
  assert is_same_tree(expected, elem)

def test_new_NGonElements():
  elem = create_nodes.new_NGonElements('NGon', erange=[1,10], ec=[1,2,7], eso=[0,3], pe=[[1,0], [2,1], [2,0]])
  expected = parse_yaml_cgns.to_node("""
  NGon Elements_t I4 [22,0]:
    ElementRange ElementRange_t I4 [1,10]:
    ElementStartOffset DataArray_t I4 [0,3]:
    ElementConnectivity DataArray_t I4 [1,2,7]:
    ParentElements DataArray_t I4 [[1,0], [2,1], [2,0]]:
  """)
  assert NA.get_value(walk.get_child_from_name(elem, "ParentElements")).shape == (3,2)
  assert is_same_tree(expected, elem)

def test_new_BC():
  bc = create_nodes.new_BC('MyBC', 'FamilySpecified', family='MyFamily', point_list=[1,2,3], loc="FaceCenter")
  expected = parse_yaml_cgns.to_node("""
  MyBC BC_t "FamilySpecified":
    FamilyName FamilyName_t "MyFamily":
    PointList IndexArray_t I4 [1,2,3]:
    GridLocation GridLocation_t "FaceCenter":
  """)
  assert is_same_tree(expected, bc)
  with pytest.raises(AssertionError):
    create_nodes.new_BC(point_list=[1,2,3], point_range=[[1,5], [1,5]], loc='Vertex')

def test_new_PointList():
  node = create_nodes.new_PointList('MyPointList', [1,2,3])
  assert NA.get_name(node)  == 'MyPointList'
  assert NA.get_label(node) == 'IndexArray_t'
  assert np.array_equal(NA.get_value(node), [1,2,3])

def test_new_PointRange():
  node = create_nodes.new_PointRange('MyPointRange', [[1,10], [1,10], [1,1]])
  assert NA.get_name(node)  == 'MyPointRange'
  assert NA.get_label(node) == 'IndexRange_t'
  assert NA.get_value(node).shape == (3,2)
  node2 = create_nodes.new_PointRange('MyPointRange', [1,10, 1,10, 1,1])
  assert is_same_tree(node, node2)

def test_new_GridLocation():
  loc = create_nodes.new_GridLocation('JFaceCenter')
  assert NA.get_name(loc) == 'GridLocation'
  assert NA.get_value(loc) == 'JFaceCenter'
  with pytest.raises(AssertionError):
    create_nodes.new_GridLocation('Toto')

def test_new_DataArray():
  data = create_nodes.new_DataArray('Array', np.array([1,2,3], np.int32))
  assert NA.get_label(data) == 'DataArray_t'
  assert np.array_equal(NA.get_value(data), [1,2,3])
  assert NA.get_value(data).dtype == np.int32
  data = create_nodes.new_DataArray('Array', np.array([1,2,3]), dtype="R8")
  #assert NA.get_value(data).dtype == np.float64

def test_new_FlowSolution():
  sol = create_nodes.new_FlowSolution('MySol', loc='CellCenter', \
      fields={'data1' : [1,2,3], 'data2' : [1.,2,3]})
  expected = parse_yaml_cgns.to_node("""
  MySol FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    data1 DataArray_t I4 [1,2,3]:
    data2 DataArray_t R8 [1,2,3]:
  """)
  assert is_same_tree(expected, sol)
