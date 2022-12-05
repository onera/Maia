import pytest
import os
import numpy as np

import maia.pytree as PT

from maia.pytree.yaml   import parse_yaml_cgns

from maia.pytree import compare as CP
from maia.pytree.compare_arrays import field_comparison, tensor_field_comparison
from mpi4py import MPI


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


def test_diff_tree_field_comp():
  t_ref = """
FlowSolution FlowSolution_t []:
  MomentumX DataArray_t [1.,1.,1.,1.]:
  MomentumY DataArray_t [2.,2.,2.,2.]:
  MomentumZ DataArray_t [3.,3.,3.,3.]:
  StressXX DataArray_t [1.,1.]:
  StressXY DataArray_t [0.,0.]:
  StressXZ DataArray_t [0.,0.]:
  StressYX DataArray_t [0.,0.]:
  StressYY DataArray_t [0.,0.]:
  StressYZ DataArray_t [0.,0.]:
  StressZX DataArray_t [0.,0.]:
  StressZY DataArray_t [0.,0.]:
  StressZZ DataArray_t [0.,0.]:
"""
  t = """
FlowSolution FlowSolution_t []:
  MomentumX DataArray_t [0.,1.,1.,1.]: # DIFF HERE
  MomentumY DataArray_t [2.,2.,2.,2.]:
  MomentumZ DataArray_t [3.,3.,3.,3.]:
  StressXX DataArray_t [1.,1.]:
  StressXY DataArray_t [0.,1.e-20]: # DIFF HERE
  StressXZ DataArray_t [0.,0.]:
  StressYX DataArray_t [0.,0.]:
  StressYY DataArray_t [0.,0.]:
  StressYZ DataArray_t [0.,0.]:
  StressZX DataArray_t [0.,0.]:
  StressZY DataArray_t [0.,0.]:
  StressZZ DataArray_t [0.,0.]:
"""
  t_ref = parse_yaml_cgns.to_node(t_ref)
  t     = parse_yaml_cgns.to_node(t)

  is_ok, err_report, warn_report = CP.diff_tree(t, t_ref, comp = field_comparison(1.e-12, MPI.COMM_SELF))
  assert not is_ok
  assert err_report == '/FlowSolution/MomentumX -- Values differ: RMS mean diff: 5.000e-01, RMS ref mean: 1.000e+00, rel error: 5.000e-01\n' \
                       '/FlowSolution/StressXY -- Values differ: RMS mean diff: 7.071e-21, RMS ref mean: 0.000e+00, rel error: inf\n'

  is_ok, err_report, warn_report = CP.diff_tree(t, t_ref, comp = tensor_field_comparison(1.e-12, MPI.COMM_SELF))
  assert not is_ok
  assert err_report == '/FlowSolution/Momentum -- Values differ: RMS mean diff: 5.000e-01, RMS ref mean: 3.742e+00, rel error: 1.336e-01\n'
