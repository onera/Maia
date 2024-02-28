import pytest
import os
import numpy as np

import maia.pytree as PT

from maia.pytree.yaml   import parse_yaml_cgns

from maia.pytree import compare as CP


dir_path = os.path.dirname(os.path.realpath(__file__))

def test_is_same_value_type():
  node1 = PT.new_node('Data', 'DataArray_t', value=None)
  node2 = PT.new_node('Data', 'DataArray_t', value=None)
  assert CP.is_same_value_type(node1, node2)
  PT.set_value(node1, np.array([1,2,3], dtype=np.int64))
  assert not CP.is_same_value_type(node1, node2)
  PT.set_value(node2, np.array([1,2,3], np.int32))
  assert CP.is_same_value_type(node1, node2, strict=False)
  assert not CP.is_same_value_type(node1, node2, strict=True)

def test_is_same_value_shape():
  node1 = PT.new_node('Data', 'DataArray_t', value=[1,2,3])
  node2 = PT.new_node('Data', 'DataArray_t', value=[4,5,6])
  assert CP.is_same_value_shape(node1, node2)
  node1 = PT.new_node('Data', 'DataArray_t', value=[1,2,3])
  node2 = PT.new_node('Data', 'DataArray_t', value=[4,5,6,7])
  assert not CP.is_same_value_shape(node1, node2)
  node1 = PT.new_node('Data', 'DataArray_t', value=[1,2,3])
  node2 = PT.new_node('Data', 'DataArray_t', value=[[4,5,6]])
  assert not CP.is_same_value_shape(node1, node2)


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

def test_report_diff():
  x = np.array([1., 2., 3])
  ref = np.array([1., 2., 3.])
  is_equal = np.ones(3, bool)
  assert CP._report_diff(x, ref, is_equal) == (True, '', '')
  x[2] = 4.
  is_equal[2] = False
  assert CP._report_diff(x, ref, is_equal) == (False, '[1. 2. 4.] <> [1. 2. 3.]', '')
  x = np.ones(20) # Values does not matter for this test
  is_equal = np.ones(20, bool)
  is_equal[::2] = False
  assert CP._report_diff(x, x, is_equal) == (False, '10 values are different', '')

def test_unzip_pairs():
  first, second = CP._unzip_pairs([['a', 1], ['b', 2], ['c', None]])
  assert first == ['a', 'b', 'c']
  assert second == [1,2,None]

def test_str_comp():
  node_1 = PT.new_node('Des', 'Descriptor_t', 'VAL')
  node_2 = PT.new_node('Des', 'Descriptor_t', 'VALUE')
  assert CP.str_comp([node_1], [node_2]) == (False, 'VAL <> VALUE', '')
  PT.set_value(node_1, ["Array", "of", "str"])
  PT.set_value(node_2, ["Array", "of", "str"])
  assert CP.str_comp([node_1], [node_2]) == (True, '', '')
  PT.set_value(node_2, ["array", "of", "strs"])
  assert CP.str_comp([node_1], [node_2]) == (False, "['Array', 'of', 'str'] <> ['array', 'of', 'strs']", '')

def test_is_same_tree():
  with open(os.path.join(dir_path, "minimal_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)
  t1 = PT.get_node_from_name(tree, 'gc5')
  t2 = PT.deep_copy(t1)
  assert CP.is_same_tree(t1, t2)

  # Position of child does not matter
  t2 = PT.deep_copy(t1)
  t2[2][1], t2[2][2] = t2[2][2], t2[2][1]
  assert CP.is_same_tree(t1, t2)

  # But node must have same children names
  t2 = PT.deep_copy(t1)
  PT.new_node('Index_vii', 'IndexArray_t', parent=t2)
  assert not CP.is_same_tree(t1, t2)

  # And those one should be equal
  t2 = PT.deep_copy(t1)
  t3 = PT.deep_copy(t1)
  PT.new_node('Index_vii', 'IndexArray_t', value=[0], parent=t2)
  PT.new_node('Index_vii', 'IndexArray_t', value=[1], parent=t3)
  assert not CP.is_same_tree(t2, t3)

def test_diff_tree():
  with open(os.path.join(dir_path, "minimal_tree.yaml"), 'r') as yt:
    t1 = parse_yaml_cgns.to_cgns_tree(yt)
  t2 = PT.deep_copy(t1)
  assert CP.diff_tree(t1, t2)[1] == ''

  # Position of child does not matter
  t2 = PT.deep_copy(t1)
  gc5_t3 = PT.get_node_from_name(t2, 'gc5')
  gc5_t3[2][1], gc5_t3[2][2] = gc5_t3[2][2], gc5_t3[2][1]
  assert CP.diff_tree(t1, t2)[1] == ''

  # But node must have the same name...
  t2 = PT.deep_copy(t1)
  gc5_t2 = PT.get_node_from_name(t2, 'gc5')
  PT.set_name(gc5_t2, 'gc6')
  assert CP.diff_tree(t1, t2)[1] == '< /CGNSTree/Base/ZoneI/ZGCB/gc5\n' \
                                    '> /CGNSTree/Base/ZoneI/ZGCB/gc6\n'

  # ... Same label ...
  t2 = PT.deep_copy(t1)
  gc5_t2 = PT.get_node_from_name(t2, 'gc5')
  PT.set_label(gc5_t2, 'IndexRange_t')
  assert CP.diff_tree(t1, t2)[1] == '/CGNSTree/Base/ZoneI/ZGCB/gc5 -- Labels differ: GridConnectivity_t <> IndexRange_t\n'

  # ... Same children ...
  t2 = PT.deep_copy(t1)
  gc5_t2 = PT.get_node_from_name(t2, 'gc5')
  PT.new_node('Index_vii', 'IndexArray_t', parent=gc5_t2)
  assert CP.diff_tree(t1, t2)[1] == '> /CGNSTree/Base/ZoneI/ZGCB/gc5/Index_vii\n'

  # ... And values should be equal
  t2 = PT.deep_copy(t1)
  t3 = PT.deep_copy(t1)
  gc5_t2 = PT.get_node_from_name(t2, 'gc5')
  gc5_t3 = PT.get_node_from_name(t3, 'gc5')
  PT.new_node('Index_vii', 'IndexArray_t', value=[0], parent=gc5_t2)
  PT.new_node('Index_vii', 'IndexArray_t', value=[1], parent=gc5_t3)
  assert CP.diff_tree(t2, t3)[1] == '/CGNSTree/Base/ZoneI/ZGCB/gc5/Index_vii -- Values differ: [0] <> [1]\n'

