import Converter.Internal as I
import numpy as np
import maia.utils.py_utils as py_utils
from   maia.utils        import parse_yaml_cgns

def test_list_or_only_elt():
  assert py_utils.list_or_only_elt([42]) == 42
  input = [1,2,3, "nous irons aux bois"]
  assert py_utils.list_or_only_elt(input) is input

def test_interweave_arrays():
  first  = np.array([1,2,3], dtype=np.int32)
  second = np.array([11,22,33], dtype=np.int32)
  third  = np.array([111,222,333], dtype=np.int32)
  assert (py_utils.interweave_arrays([first]) == [1,2,3]).all()
  assert (py_utils.interweave_arrays([second, third]) == \
      [11,111,22,222,33,333]).all()
  assert (py_utils.interweave_arrays([first, second, third]) == \
      [1,11,111,2,22,222,3,33,333]).all()

def test_nb_to_offset():
  assert(py_utils.nb_to_offset([]) == np.zeros(1))
  assert(py_utils.nb_to_offset([5,3,5,10]) == np.array([0,5,8,13,23])).all()
  assert(py_utils.nb_to_offset([5,0,0,10]) == np.array([0,5,5,5,15])).all()
  assert py_utils.nb_to_offset([5,0,0,10], np.int32).dtype == np.int32
  assert py_utils.nb_to_offset([5,0,0,10], np.int64).dtype == np.int64

def test_concatenate_point_list():
  pl1 = np.array([[2, 4, 6, 8]])
  pl2 = np.array([[10, 20, 30, 40, 50, 60]])
  pl3 = np.array([[100]])
  plvoid = np.empty((1,0))

  #No pl at all in the mesh
  none_idx, none = py_utils.concatenate_point_list([])
  assert none_idx == [0]
  assert isinstance(none, np.ndarray)
  assert none.shape == (0,)

  #A pl, but with no data
  empty_idx, empty = py_utils.concatenate_point_list([plvoid])
  assert (none_idx == [0,0]).all()
  assert isinstance(empty, np.ndarray)
  assert empty.shape == (0,)

  # A pl with data
  one_idx, one = py_utils.concatenate_point_list([pl1])
  assert (one_idx == [0,4]).all()
  assert (one     == pl1[0]).all()

  # Several pl
  merged_idx, merged = py_utils.concatenate_point_list([pl1, pl2, pl3])
  assert (merged_idx == [0, pl1.size, pl1.size+pl2.size, pl1.size+pl2.size+pl3.size]).all()
  assert (merged[0:pl1.size]                 == pl1[0]).all()
  assert (merged[pl1.size:pl1.size+pl2.size] == pl2[0]).all()
  assert (merged[pl1.size+pl2.size:]         == pl3[0]).all()
  # Several pl, some with no data
  merged_idx, merged = py_utils.concatenate_point_list([pl1, plvoid, pl2])
  assert (merged_idx == [0, 4, 4, 10]).all()
  assert (merged[0:4 ] == pl1[0]).all()
  assert (merged[4:10] == pl2[0]).all()

def test_getNodesFromTypePath():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index_i IndexArray_t:
    bc2 BC_t:
      Index_ii IndexArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index_iii IndexArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index_iv IndexArray_t:
      Index_v IndexArray_t:
      Index_vi IndexArray_t:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)
  zoneI = I.getNodeFromName(root, 'ZoneI')
  zbcB  = I.getNodeFromName(root, 'ZBCB' )

  assert list(py_utils.getNodesFromTypePath(zoneI, '')) == []
  assert list(py_utils.getNodesFromTypePath(zoneI, 'BC_t')) == []

  onelvl = py_utils.getNodesFromTypePath(zoneI, 'ZoneBC_t')
  assert [I.getName(node) for node in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = py_utils.getNodesFromTypePath(zbcB, 'BC_t')
  assert [I.getName(node) for node in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = py_utils.getNodesFromTypePath(zoneI, 'ZoneBC_t/BC_t')
  assert [I.getName(node) for node in twolvl] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  twolvl = py_utils.getNodesFromTypePath(zbcB, 'BC_t/IndexArray_t')
  assert [I.getName(node) for node in twolvl] == ['Index_iii','Index_iv','Index_v','Index_vi']

  threelvl = py_utils.getNodesFromTypePath(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  assert [I.getName(node) for node in threelvl] == ['Index_i', 'Index_ii','Index_iii','Index_iv','Index_v','Index_vi']

def test_getNodesWithParentsFromTypePath():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index_i IndexArray_t:
    bc2 BC_t:
      Index_ii IndexArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index_iii IndexArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index_iv IndexArray_t:
      Index_v IndexArray_t:
      Index_vi IndexArray_t:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)
  zoneI = I.getNodeFromName(root, 'ZoneI')
  zbcB  = I.getNodeFromName(root, 'ZBCB' )

  assert list(py_utils.getNodesWithParentsFromTypePath(zoneI, '')) == []
  assert list(py_utils.getNodesWithParentsFromTypePath(zoneI, 'BC_t')) == []

  onelvl = py_utils.getNodesWithParentsFromTypePath(zoneI, 'ZoneBC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = py_utils.getNodesWithParentsFromTypePath(zbcB, 'BC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = py_utils.getNodesWithParentsFromTypePath(zoneI, 'ZoneBC_t/BC_t')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCA', 'bc1'), ('ZBCA', 'bc2'), ('ZBCB', 'bc3'), ('ZBCB', 'bc4'), ('ZBCB', 'bc5')]
  twolvl = py_utils.getNodesWithParentsFromTypePath(zbcB, 'BC_t/IndexArray_t')
  for bc in I.getNodesFromType1(zbcB, 'BC_t'):
    for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
      assert next(twolvl) == (bc, idx)

  threelvl = py_utils.getNodesWithParentsFromTypePath(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
        assert next(threelvl) == (zbc, bc, idx)

def test_getNodesByMatching():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index_i IndexArray_t:
    bc2 BC_t:
      Index_i IndexArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index_i IndexArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index_i IndexArray_t:
      Index_ii IndexArray_t:
      Index_iii IndexArray_t:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)
  zoneI = I.getNodeFromName(root, 'ZoneI')
  zbcB  = I.getNodeFromName(root, 'ZBCB' )

  assert list(py_utils.getNodesByMatching(zoneI, '')) == []
  assert list(py_utils.getNodesByMatching(zoneI, 'BC_t')) == []
  assert list(py_utils.getNodesByMatching(zoneI, 'Index_ii')) == []

  onelvl = py_utils.getNodesByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(node) for node in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = py_utils.getNodesByMatching(zbcB, 'bc5')
  assert [I.getName(node) for node in onelvl] == ['bc5']

  twolvl = py_utils.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [I.getName(node) for node in twolvl] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  twolvl = py_utils.getNodesByMatching(zoneI, 'ZoneBC_t/bc5')
  assert [I.getName(node) for node in twolvl] == ['bc5']
  twolvl = py_utils.getNodesByMatching(zbcB, 'BC_t/IndexArray_t')
  assert [I.getName(node) for node in twolvl] == ['Index_i','Index_i','Index_ii','Index_iii']

  threelvl = py_utils.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  assert [I.getName(node) for node in threelvl] == ['Index_i', 'Index_i','Index_i','Index_i','Index_ii','Index_iii']
  assert len(list(py_utils.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t/Index_i'))) == 4

def test_getNodesWithParentsByMatching():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index_i IndexArray_t:
      PL DataArray_t:
    bc2 BC_t:
      Index_ii IndexArray_t:
      PL DataArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index_iii IndexArray_t:
      PL DataArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index_iv IndexArray_t:
      Index_v IndexArray_t:
      Index_vi IndexArray_t:
      PL DataArray_t:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)
  zoneI = I.getNodeFromName(root, 'ZoneI')
  zbcB  = I.getNodeFromName(root, 'ZBCB' )

  assert list(py_utils.getNodesWithParentsByMatching(zoneI, '')) == []
  assert list(py_utils.getNodesWithParentsByMatching(zoneI, 'BC_t')) == []

  onelvl = py_utils.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = py_utils.getNodesWithParentsByMatching(zbcB, 'BC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = py_utils.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCA', 'bc1'), ('ZBCA', 'bc2'), ('ZBCB', 'bc3'), ('ZBCB', 'bc4'), ('ZBCB', 'bc5')]
  twolvl = py_utils.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/bc3')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCB', 'bc3')]
  twolvl = py_utils.getNodesWithParentsByMatching(zbcB, 'BC_t/IndexArray_t')
  for bc in I.getNodesFromType1(zbcB, 'BC_t'):
    for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
      assert next(twolvl) == (bc, idx)

  threelvl = py_utils.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
        assert next(threelvl) == (zbc, bc, idx)

  threelvl = py_utils.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/PL')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL'):
        assert next(threelvl) == (zbc, bc, pl)
