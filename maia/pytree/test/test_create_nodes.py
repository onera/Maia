import pytest
import numpy as np
from maia.pytree import create_nodes
from maia.pytree import nodes_attr as NA

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

def test_create_child():
  node  = create_nodes.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  child = create_nodes.create_child(node, 'Transform', 'Transform_t', [1,2,3])
  assert NA.get_children(node)[0] == child
  otherchild = create_nodes.create_child(node, 'Transform', label='UserDefinedData_t')
  assert len(NA.get_children(node)) == 1
  assert NA.get_children(node)[0] == otherchild
  assert NA.get_value(otherchild) is None #Old values are erased

def test_update_child():
  node = create_nodes.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  child = create_nodes.update_child(node, 'Transform', 'Transform_t', [1,2,3])
  assert NA.get_children(node)[0] == child
  child = create_nodes.update_child(node, 'Transform', value=[3,2,1])
  assert len(NA.get_children(node)) == 1
  assert NA.get_children(node)[0] == child
  assert np.array_equal(NA.get_value(child), [3,2,1])
  assert NA.get_label(child) == 'Transform_t' #Old value are preserved
