import pytest
import numpy as np
import maia.pytree        as PT

from maia.pytree.yaml import parse_yaml_cgns

def test_empty_tree():
  yt = ""
  nodes = parse_yaml_cgns.to_nodes(yt)
  assert nodes == []

  node = parse_yaml_cgns.to_node(yt)
  assert node is None

  complete_t = parse_yaml_cgns.to_cgns_tree(yt)
  assert complete_t == ["CGNSTree",None,[['CGNSLibraryVersion', np.array([4.2], np.float32), [], 'CGNSLibraryVersion_t']],"CGNSTree_t"]


def test_simple_tree():
  yt = """
Base0 CGNSBase_t [3,3]:
  Zone0 Zone_t [[24],[6],[0]]:
  Zone with  spaces in name     Zone_t R8 [[4,3,2],[3,2,1],[0,0,0]]:
"""
  t = parse_yaml_cgns.to_cgns_tree(yt)
  bs = PT.get_children_from_label(t,"CGNSBase_t")
  assert len(bs) == 1
  assert PT.get_name(bs[0]) == "Base0"
  assert PT.get_label(bs[0]) == "CGNSBase_t"
  assert np.all(PT.get_value(bs[0]) == [3,3])

  zs = PT.get_children_from_label(bs[0],"Zone_t")
  assert np.all(PT.get_value(zs[0]) == [[24],[6],[0]])
  assert PT.get_children(zs[0]) == []
  assert PT.get_node_from_name(t, 'Zone with  spaces in name')[1].dtype == np.float64

  yt = """
  ReferenceState ReferenceState_t:
  Zone0 Zone_t [[24],[6],[0]]:
"""
  t = parse_yaml_cgns.to_cgns_tree(yt)
  bs = PT.get_children_from_label(t,"CGNSBase_t")
  assert len(bs) == 1
  assert PT.get_name(bs[0]) == "Base"
  assert PT.get_label(bs[0]) == "CGNSBase_t"
  assert (PT.get_value(bs[0]) == [3,3]).all()

  yt = """
  CGNSTree CGNSTree_t:
    Base CGNSBase_t [3,3]:
      Zone0 Zone_t [[24],[6],[0]]:
"""
  t = parse_yaml_cgns.to_cgns_tree(yt)
  assert len(PT.get_all_Zone_t(t)) == 1

  yt = """
  BC BC_t "BCWall":
    GridLocation GridLocation_t "FaceCenter":
"""
  with pytest.raises(ValueError):
    t = parse_yaml_cgns.to_cgns_tree(yt)

  yt = """
  BC BC_t "BCWall":
    GridLocation Wronglabel "FaceCenter":
"""
  with pytest.raises(ValueError):
    t = parse_yaml_cgns.to_node(yt)


def test_multi_line_value():
  yt = """
CoordinateX DataArray_t:
  R8 : [ 0,1,2,3,
         0,1,2,3 ]
"""
  nodes = parse_yaml_cgns.to_nodes(yt)
  assert len(nodes) == 1

  node = nodes[0]
  assert PT.get_name(node) == "CoordinateX"
  assert PT.get_label(node) == "DataArray_t"
  assert (PT.get_value(node) == np.array([0,1,2,3,0,1,2,3],dtype=np.float64)).all()
  assert PT.get_value(node).dtype == np.float64
  assert PT.get_children(node) == []

  single_node = parse_yaml_cgns.to_node(yt)
  assert single_node[0] == node[0]
  assert (single_node[1] == node[1]).all()
  assert single_node[2] == node[2]
  assert single_node[3] == node[3]


def test_base_dim():
  yt_2D = """
ZoneU Zone_t [[6,0,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t:
    CoordinateY DataArray_t:
  """
  t = parse_yaml_cgns.to_cgns_tree(yt_2D)
  b = PT.get_child_from_label(t,"CGNSBase_t")
  assert (PT.get_value(b) == [2,2]).all()
  
  yt_2D_curv_S = """
ZoneS Zone_t [[6,5,0],[2,1,0]]:
  ZoneType ZoneType_t "Structured":
  GridCoordinates GridCoordinates_t:
    CoordinateXi DataArray_t:
    CoordinateEta DataArray_t:
    CoordinateZeta DataArray_t:
  """
  t = parse_yaml_cgns.to_cgns_tree(yt_2D_curv_S)
  b = PT.get_child_from_label(t,"CGNSBase_t")
  assert (PT.get_value(b) == [2,3]).all()
  
  yt_2D_curv_U_ngon = """
ZoneU Zone_t:
  ZoneType ZoneType_t "Unstructured":
  GridCoordinates GridCoordinates_t:
    CoordinateR DataArray_t:
    CoordinateTheta DataArray_t:
    CoordinateZ DataArray_t:
    CoordinateTransform DataArray_t:
  Ngon Elements_t [22,0]:
  """
  t = parse_yaml_cgns.to_cgns_tree(yt_2D_curv_U_ngon)
  b = PT.get_child_from_label(t,"CGNSBase_t")
  assert (PT.get_value(b) == [2,3]).all()
  
  yt_2D_curv_U_elem = """
ZoneU Zone_t:
  ZoneType ZoneType_t "Unstructured":
  GridCoordinates GridCoordinates_t:
    CoordinateR DataArray_t:
    CoordinateTheta DataArray_t:
    CoordinateZ DataArray_t:
  Quad Elements_t [7,0]:
  """
  t = parse_yaml_cgns.to_cgns_tree(yt_2D_curv_U_elem)
  b = PT.get_child_from_label(t,"CGNSBase_t")
  assert (PT.get_value(b) == [2,3]).all()
  
  yt_3D_S = """
ZoneS Zone_t [[6,5,0],[2,1,0],[2,1,0]]:
  ZoneType ZoneType_t "Structured":
  GridCoordinates GridCoordinates_t:
    CoordinateXi DataArray_t:
    CoordinateEta DataArray_t:
    CoordinateZeta DataArray_t:
  """
  t = parse_yaml_cgns.to_cgns_tree(yt_3D_S)
  b = PT.get_child_from_label(t,"CGNSBase_t")
  assert (PT.get_value(b) == [3,3]).all()
  
  yt_3D_U = """
ZoneU Zone_t [[6,0,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateR DataArray_t:
    CoordinateTheta DataArray_t:
    CoordinateZ DataArray_t:
  """
  t = parse_yaml_cgns.to_cgns_tree(yt_3D_U)
  b = PT.get_child_from_label(t,"CGNSBase_t")
  assert (PT.get_value(b) == [3,3]).all()
  
  yt = """
ZoneU Zone_t [[6,0,0]]:
  """
  t = parse_yaml_cgns.to_cgns_tree(yt)
  b = PT.get_child_from_label(t,"CGNSBase_t")
  assert (PT.get_value(b) == [3,3]).all()
