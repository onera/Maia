import pytest
import numpy as np

from maia.pytree.yaml    import parse_yaml_cgns
from maia.pytree.compare import is_same_tree

from maia.pytree import node as N
from maia.pytree import walk as W

from maia.pytree.node import presets

def test_new_CGNSTree():
  tree = presets.new_CGNSTree(version=3.1)
  expected = parse_yaml_cgns.to_node("""
  CGNSTree CGNSTree_t:
    CGNSLibraryVersion CGNSLibraryVersion_t 3.1:
  """)
  assert is_same_tree(expected, tree)

def test_new_CGNSBase():
  base = presets.new_CGNSBase(phy_dim=2)
  assert np.array_equal(N.get_value(base), [3,2])

def test_new_Family():
  fam = presets.new_Family('WALL', family_bc='BCWall')
  expected = parse_yaml_cgns.to_node("""
  WALL Family_t:
    FamilyBC FamilyBC_t "BCWall":
  """)
  assert is_same_tree(expected, fam)

def test_new_Zone():
  zone = presets.new_Zone('SomeZone', type='Unstructured', family='Family', size=[[11, 10, 0]])
  expected = parse_yaml_cgns.to_node("""
  SomeZone Zone_t I4 [[11,10,0]]:
    FamilyName FamilyName_t "Family":
    ZoneType ZoneType_t "Unstructured":
  """)
  assert is_same_tree(expected, zone)
  
  with pytest.raises(AssertionError):
    presets.new_Zone('SomeZone', type='Toto')

def test_new_Elements():
  elem = presets.new_Elements('Tri', 'TRI_3')
  assert np.array_equal(N.get_value(elem), [5,0])
  elem = presets.new_Elements('Tri', 5)
  assert np.array_equal(N.get_value(elem), [5,0])
  elem = presets.new_Elements('Tri', [5,1], erange=[1,10], econn=[1,2,7])
  expected = parse_yaml_cgns.to_node("""
  Tri Elements_t I4 [5,1]:
    ElementRange IndexRange_t I4 [1,10]:
    ElementConnectivity DataArray_t I4 [1,2,7]:
  """)
  assert is_same_tree(expected, elem)

def test_new_NGonElements():
  elem = presets.new_NGonElements('NGon', erange=[1,10], ec=[1,2,7], eso=[0,3], pe=[[1,0], [2,1], [2,0]])
  expected = parse_yaml_cgns.to_node("""
  NGon Elements_t I4 [22,0]:
    ElementRange IndexRange_t I4 [1,10]:
    ElementStartOffset DataArray_t I4 [0,3]:
    ElementConnectivity DataArray_t I4 [1,2,7]:
    ParentElements DataArray_t I4 [[1,0], [2,1], [2,0]]:
  """)
  assert N.get_value(W.get_child_from_name(elem, "ParentElements")).shape == (3,2)
  assert is_same_tree(expected, elem)

def test_new_NFaceElements():
  elem = presets.new_NFaceElements('NFace', erange=[1,10], ec=[1,2,7], eso=[0,3])
  expected = parse_yaml_cgns.to_node("""
  NFace Elements_t I4 [23,0]:
    ElementRange IndexRange_t I4 [1,10]:
    ElementStartOffset DataArray_t I4 [0,3]:
    ElementConnectivity DataArray_t I4 [1,2,7]:
  """)
  assert is_same_tree(expected, elem)

def test_new_ZoneBC():
  zbc = presets.new_ZoneBC()
  assert N.get_name(zbc) == 'ZoneBC'
  assert N.get_label(zbc) == 'ZoneBC_t'
  assert N.get_value(zbc) is None

def test_new_ZoneGridConnectivity():
  zbc = presets.new_ZoneGridConnectivity('ZoneGC')
  assert N.get_name(zbc) == 'ZoneGC'
  assert N.get_label(zbc) == 'ZoneGridConnectivity_t'
  assert N.get_value(zbc) is None

def test_new_BC():
  bc = presets.new_BC('MyBC', 'FamilySpecified', family='MyFamily', point_list=[1,2,3], loc="FaceCenter")
  expected = parse_yaml_cgns.to_node("""
  MyBC BC_t "FamilySpecified":
    FamilyName FamilyName_t "MyFamily":
    PointList IndexArray_t I4 [1,2,3]:
    GridLocation GridLocation_t "FaceCenter":
  """)
  assert is_same_tree(expected, bc)
  with pytest.raises(AssertionError):
    presets.new_BC(point_list=[1,2,3], point_range=[[1,5], [1,5]], loc='Vertex')

def test_new_GCType():
  gctype = presets.new_GridConnectivityType('Abutting1to1')
  assert N.get_name(gctype) == 'GridConnectivityType'
  assert N.get_label(gctype) == 'GridConnectivityType_t'
  assert N.get_value(gctype) == 'Abutting1to1'
  with pytest.raises(AssertionError):
    gctype = presets.new_GridConnectivityType('WrongType')

def test_new_GridConnectivityProperty():
  props = presets.new_GridConnectivityProperty()
  assert N.get_name(props) == 'GridConnectivityProperty'
  assert N.get_value(props) is None

  perio = {'rotation_angle' : [30.,0,0], 'translation' : [0.,0.,1.]}
  props = presets.new_GridConnectivityProperty(periodic=perio)
  expected = parse_yaml_cgns.to_node("""
  GridConnectivityProperty GridConnectivityProperty_t:
    Periodic Periodic_t:
      RotationAngle DataArray_t [30., 0, 0]:
      RotationCenter DataArray_t [0., 0., 0]:
      Translation DataArray_t [0., 0, 1]:
  """)
  assert is_same_tree(expected, props)

def test_new_Periodic():
  perio = presets.new_Periodic(rotation_angle=[0,45.,0])
  expected = parse_yaml_cgns.to_node("""
  Periodic Periodic_t:
    RotationAngle DataArray_t [0, 45., 0]:
    RotationCenter DataArray_t [0., 0., 0]:
    Translation DataArray_t [0., 0, 0]:
  """)
  assert is_same_tree(expected, perio)

def test_new_GC():
  gc = presets.new_GridConnectivity('MyGC', 'OppZone', 'Abutting1to1', loc="FaceCenter")
  expected = parse_yaml_cgns.to_node("""
  MyGC GridConnectivity_t "OppZone":
    GridConnectivityType GridConnectivityType_t "Abutting1to1":
    GridLocation GridLocation_t "FaceCenter":
  """)
  assert is_same_tree(expected, gc)

  gc = presets.new_GridConnectivity('MyGC', 'OppZone', 'Abutting1to1', point_list=[[1,2,3]],
      point_list_donor=[[4,5,6]])
  expected = parse_yaml_cgns.to_node("""
  MyGC GridConnectivity_t "OppZone":
    GridConnectivityType GridConnectivityType_t "Abutting1to1":
    PointList IndexArray_t [[1,2,3]]:
    PointListDonor IndexArray_t [[4,5,6]]:
  """)
  assert is_same_tree(expected, gc)

  with pytest.raises(AssertionError):
    gc = presets.new_GridConnectivity('MyGC', 'OppZone', 'Abutting1to1', point_list=[[1,2,3]],
        point_range=[[4,5,6]])

def test_new_GC1to1():
  gc = presets.new_GridConnectivity1to1('MyGC', 'OppZone', transform=[2,1,-3], \
      point_range=[[1,1], [1,5], [1,5]])
  expected = parse_yaml_cgns.to_node("""
  MyGC GridConnectivity1to1_t "OppZone":
    Transform "int[IndexDimension]" [2,1,-3]:
    PointRange IndexRange_t [[1,1], [1,5], [1,5]]:
  """)
  assert is_same_tree(expected, gc)
  assert is_same_tree(expected, gc)

def test_new_PointList():
  node = presets.new_PointList('MyPointList', [1,2,3])
  assert N.get_name(node)  == 'MyPointList'
  assert N.get_label(node) == 'IndexArray_t'
  assert np.array_equal(N.get_value(node), [1,2,3])

def test_new_PointRange():
  node = presets.new_PointRange('MyPointRange', [[1,10], [1,10], [1,1]])
  assert N.get_name(node)  == 'MyPointRange'
  assert N.get_label(node) == 'IndexRange_t'
  assert N.get_value(node).shape == (3,2)
  node2 = presets.new_PointRange('MyPointRange', [1,10, 1,10, 1,1])
  assert is_same_tree(node, node2)

def test_new_GridLocation():
  loc = presets.new_GridLocation('JFaceCenter')
  assert N.get_name(loc) == 'GridLocation'
  assert N.get_value(loc) == 'JFaceCenter'
  with pytest.raises(AssertionError):
    presets.new_GridLocation('Toto')

def test_new_DataArray():
  data = presets.new_DataArray('Array', np.array([1,2,3], np.int32))
  assert N.get_label(data) == 'DataArray_t'
  assert np.array_equal(N.get_value(data), [1,2,3])
  assert N.get_value(data).dtype == np.int32
  data = presets.new_DataArray('Array', np.array([1,2,3]), dtype="R8")
  assert N.get_value(data).dtype == np.float64

def test_new_GridCoordinates():
  gco = presets.new_GridCoordinates('MyGridCoords', fields={'CX' : [1,2,3], 'CY' : np.array([1.,2,3])})
  expected = parse_yaml_cgns.to_node("""
  MyGridCoords GridCoordinates_t:
    CX DataArray_t I4 [1,2,3]:
    CY DataArray_t R8 [1,2,3]:
  """)
  assert is_same_tree(expected, gco)

def test_new_FlowSolution():
  sol = presets.new_FlowSolution('MySol', loc='CellCenter', \
      fields={'data1' : [1,2,3], 'data2' : [1.,2,3]})
  expected = parse_yaml_cgns.to_node("""
  MySol FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    data1 DataArray_t I4 [1,2,3]:
    data2 DataArray_t R4 [1,2,3]:
  """)
  assert is_same_tree(expected, sol)

def test_new_ZoneSubRegion():
    #name='ZoneSubRegion', *, loc=None, point_range=None, point_list=None, bc_name=None, gc_name=None, \
    #family=None, fields={}, parent=None):
  zsr = presets.new_ZoneSubRegion(loc='Vertex', point_list=[[1,4,6]], fields={'data1' : [1,2,3], 'data2' : [1.,2,3]})
  expected = parse_yaml_cgns.to_node("""
  ZoneSubRegion ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[1,4,6]]:
    data1 DataArray_t I4 [1,2,3]:
    data2 DataArray_t R4 [1,2,3]:
  """)
  assert is_same_tree(expected, zsr)

  zsr = presets.new_ZoneSubRegion("MyZSR", loc='FaceCenter', family='Fam', bc_name="SomeBC")
  expected = parse_yaml_cgns.to_node("""
  MyZSR ZoneSubRegion_t:
    GridLocation GridLocation_t "FaceCenter":
    BCRegionName Descriptor_t "SomeBC":
    FamilyName FamilyName_t "Fam":
  """)
  assert is_same_tree(expected, zsr)

  with pytest.raises(AssertionError):
    zsr = presets.new_ZoneSubRegion(point_list=[[1,4,6]], bc_name="SomeBC")
