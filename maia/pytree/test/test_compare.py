import pytest
import os
import numpy as np

import maia.pytree as PT

from maia.pytree.yaml   import parse_yaml_cgns

from maia.pytree import compare as CP


dir_path = os.path.dirname(os.path.realpath(__file__))

def test_check_is_label():
  with open(os.path.join(dir_path, "minimal_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  @CP.check_is_label('Zone_t')
  def apply_zone(node):
    pass

  for zone in PT.get_all_Zone_t(tree):
    apply_zone(zone)

  with pytest.raises(CP.CGNSLabelNotEqualError):
    for zone in PT.get_all_CGNSBase_t(tree):
      apply_zone(zone)

def test_check_in_labels():
  with open(os.path.join(dir_path, "minimal_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  @CP.check_in_labels(['Zone_t', 'CGNSBase_t'])
  def foo(node):
    pass

  for zone in PT.get_all_Zone_t(tree):
    foo(zone)
  for zone in PT.get_all_CGNSBase_t(tree):
    foo(zone)
  with pytest.raises(CP.CGNSLabelNotEqualError):
    foo(tree)

def test_is_same_value_type():
  node1 = PT.new_node('Data', 'DataArray_t', value=None)
  node2 = PT.new_node('Data', 'DataArray_t', value=None)
  assert CP.is_same_value_type(node1, node2)
  PT.set_value(node1, np.array([1,2,3], dtype=np.int64))
  assert not CP.is_same_value_type(node1, node2)
  PT.set_value(node2, np.array([1,2,3], np.int32))
  assert CP.is_same_value_type(node1, node2, strict=False)
  assert not CP.is_same_value_type(node1, node2, strict=True)

def test_is_same_value():
  node1 = PT.new_node('Data', 'DataArray_t', value=np.array([1,2,3]))
  node2 = PT.new_node('Data', 'DataArray_t', value=np.array([1,2,3]))
  assert CP.is_same_value(node1, node2)
  PT.set_value(node1, np.array([1,2,3], float))
  PT.set_value(node2, np.array([1,2,3], float))
  assert CP.is_same_value(node1, node2)
  PT.get_value(node2)[1] += 1E-8
  assert not CP.is_same_value(node1, node2)
  assert CP.is_same_value(node1, node2, abs_tol=1E-6)

def test_is_same_node():
  with open(os.path.join(dir_path, "minimal_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)
  node1 = PT.get_node_from_name(tree, 'gc3')
  node2 = PT.get_node_from_name(tree, 'gc5')
  assert not CP.is_same_node(node1, node2)
  node2[0] = 'gc3'
  assert CP.is_same_node(node1, node2) #Children are not compared
  
def test_is_same_tree():
  with open(os.path.join(dir_path, "minimal_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)
  node1 = PT.get_node_from_name(tree, 'gc5')
  node2 = PT.deep_copy(node1)
  assert CP.is_same_tree(node1, node2)
  #Position of child does not matter
  node2[2][1], node2[2][2] = node2[2][2], node2[2][1]
  assert CP.is_same_tree(node1, node2)
  # But node must have same children names
  PT.new_node('Index_vii', 'IndexArray_t', parent=node2)
  assert not CP.is_same_tree(node1, node2)
  #And those one should be equal
  PT.new_node('Index_vii', 'IndexArray_t', value=[[1,2]])
  assert not CP.is_same_tree(node1, node2)
