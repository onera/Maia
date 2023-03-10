import pytest
import numpy as np
from maia.pytree.node import create
from maia.pytree.node import access as NA

def test_new_node():
  node = create.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  assert NA.get_name(node) == 'match'
  assert NA.get_label(node) == 'GridConnectivity1to1_t'
  assert NA.get_value(node) == 'OtherZone'
  assert NA.get_children(node) == []

  child = create.new_node('Transform', 'Transform_t', [1,2,3], parent=node)
  assert (NA.get_value(child) == np.array([1,2,3])).all()
  assert NA.get_children(node)[0] == child

  with pytest.warns(RuntimeWarning):
    create.new_node('ANodeNameDefinitivelyTooLongForTheCGNSStandard', 'Transform_t', [1,2,3], parent=node)

def test_update_node():
  node = ['Node', None, [], 'UserDefined_t']
  create.update_node(node, 'NewName', value=[6.])
  assert NA.get_name(node) ==  'NewName'
  assert NA.get_value(node) == np.array([6.])
  create.update_node(node, ..., 'BC_t')
  assert NA.get_name(node)  == 'NewName'
  assert NA.get_label(node) == 'BC_t'
  with pytest.raises(Exception):
    create.update_node(node, children=12)

# def test_create_child():
  # node  = create.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  # child = create.create_child(node, 'Transform', 'Transform_t', [1,2,3])
  # assert NA.get_children(node)[0] == child
  # otherchild = create.create_child(node, 'Transform', label='UserDefinedData_t')
  # assert len(NA.get_children(node)) == 1
  # assert NA.get_children(node)[0] == otherchild
  # assert NA.get_value(otherchild) is None #Old values are erased

def test_new_child():
  node = create.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  child = create.new_child(node, 'Transform', 'Transform_t', [1,2,3])
  assert NA.get_children(node)[0] == child
  with pytest.raises(Exception):
    child = create.new_child(node, 'Transform', 'Transform_t', [3,2,1])

def test_update_child():
  node = create.new_node('match', 'GridConnectivity1to1_t', "OtherZone")
  child = create.update_child(node, 'Transform', 'Transform_t', [1,2,3])
  assert NA.get_children(node)[0] == child
  child = create.update_child(node, 'Transform', value=[3,2,1])
  assert len(NA.get_children(node)) == 1
  assert NA.get_children(node)[0] == child
  assert np.array_equal(NA.get_value(child), [3,2,1])
  assert NA.get_label(child) == 'Transform_t' #Old value are preserved

def test_shallow_copy():
  node = create.new_node('match', 'GridConnectivity1to1_t')
  child = create.update_child(node, 'Transform', 'Transform_t', [1,2,3])

  copied   = create.shallow_copy(node)
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
  node = create.new_node('match', 'GridConnectivity1to1_t')
  child = create.update_child(node, 'Transform', 'Transform_t', [1,2,3])

  copied   = create.deep_copy(node)
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


