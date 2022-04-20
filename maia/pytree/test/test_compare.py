import pytest
import os
import numpy as np

import Converter.Internal as I

from maia.utils.yaml   import parse_yaml_cgns

from maia.pytree import compare as CP


dir_path = os.path.dirname(os.path.realpath(__file__))

def test_is_valid_name():
  assert CP.is_valid_name('MyName') == True
  assert CP.is_valid_name(123) == False

def test_check_name():
  assert CP.check_name('MyName') == 'MyName'
  with pytest.raises(TypeError):
    CP.check_name(123)

def test_is_valid_value():
  assert CP.is_valid_value(123) == False
  assert CP.is_valid_value('Coucou') == False
  assert CP.is_valid_value(np.array(tuple('Coucou'), '|S1')) == True
  assert CP.is_valid_value(np.array([123])) == True
  assert CP.is_valid_value(np.ones((3,3), order='F')) == True
  assert CP.is_valid_value(np.ones((3,3), order='C')) == False

def test_check_value():
  assert CP.check_value(np.array([123])) == np.array([123])
  with pytest.raises(TypeError):
    CP.check_value(123)

def test_is_valid_children():
  assert CP.is_valid_children(['BC', 'BCWall', [], 'BC_t']) == True
  assert CP.is_valid_children(123) == False

def test_check_children():
  assert CP.check_children(['BC', 'BCWall', [], 'BC_t']) == ['BC', 'BCWall', [], 'BC_t']
  with pytest.raises(TypeError):
    CP.check_children(123)

def test_is_valid_label():
  assert CP.is_valid_label('') == False
  assert CP.is_valid_label('BC') == False
  assert CP.is_valid_label('BC_t') == True
  assert CP.is_valid_label('BC_toto') == False
  assert CP.is_valid_label('FakeLabel_t', only_sids=False) == True
  assert CP.is_valid_label('FakeLabel_t', only_sids=True) == False

def test_check_label():
  assert CP.check_label('BC_t') == 'BC_t'
  with pytest.raises(TypeError):
    CP.check_label('BC')

def test_is_valid_node():
  assert CP.is_valid_node(['BC', np.array(tuple('BCWall'), '|S1'), [], 'BC_t']) == True
  assert CP.is_valid_node(123) == False
  assert CP.is_valid_node(['BC', np.array(tuple('BCWall'), '|S1'), [], 'BC_t', 'Coucou']) == False


def test_check_is_label():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  @CP.check_is_label('Zone_t')
  def apply_zone(node):
    pass

  for zone in I.getZones(tree):
    apply_zone(zone)

  with pytest.raises(CP.CGNSLabelNotEqualError):
    for zone in I.getBases(tree):
      apply_zone(zone)

def test_check_in_labels():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  @CP.check_in_labels(['Zone_t', 'CGNSBase_t'])
  def foo(node):
    pass

  for zone in I.getZones(tree):
    foo(zone)
  for zone in I.getBases(tree):
    foo(zone)
  with pytest.raises(CP.CGNSLabelNotEqualError):
    foo(tree)

def test_is_same_value_type():
  node1 = I.createNode('Data', 'DataArray_t', value=None)
  node2 = I.createNode('Data', 'DataArray_t', value=None)
  assert CP.is_same_value_type(node1, node2)
  I.setValue(node1, np.array([1,2,3], dtype=np.int64))
  assert not CP.is_same_value_type(node1, node2)
  I.setValue(node2, np.array([1,2,3], np.int32))
  assert CP.is_same_value_type(node1, node2, strict=False)
  assert not CP.is_same_value_type(node1, node2, strict=True)

def test_is_same_value():
  node1 = I.createNode('Data', 'DataArray_t', value=np.array([1,2,3]))
  node2 = I.createNode('Data', 'DataArray_t', value=np.array([1,2,3]))
  assert CP.is_same_value(node1, node2)
  I.setValue(node1, np.array([1,2,3], float))
  I.setValue(node2, np.array([1,2,3], float))
  assert CP.is_same_value(node1, node2)
  I.getVal(node2)[1] += 1E-8
  assert not CP.is_same_value(node1, node2)
  assert CP.is_same_value(node1, node2, abs_tol=1E-6)

def test_is_same_node():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)
  node1 = I.getNodeFromName(tree, 'bc3')
  node2 = I.getNodeFromName(tree, 'bc5')
  assert not CP.is_same_node(node1, node2)
  node2[0] = 'bc3'
  assert CP.is_same_node(node1, node2) #Children are not compared
  
def test_is_same_tree():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)
  node1 = I.getNodeFromName(tree, 'bc5')
  node2 = I.copyTree(node1)
  assert CP.is_same_tree(node1, node2)
  #Position of child does not matter
  node2[2][1], node2[2][2] = node2[2][2], node2[2][1]
  assert CP.is_same_tree(node1, node2)
  # But node must have same children names
  I.newIndexArray('Index_vii', parent=node2)
  assert not CP.is_same_tree(node1, node2)
  #And those one should be equal
  I.newIndexArray('Index_vii', value=[[1,2]], parent=node1)
  assert not CP.is_same_tree(node1, node2)
