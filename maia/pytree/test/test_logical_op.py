import numpy  as np
import pytest

import maia.pytree            as PT
import maia.pytree.logical_op as PLO

from maia.pytree.yaml import parse_yaml_cgns

yt1a = """
CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
BaseA CGNSBase_t:
  Zone1 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "Zone3":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
  Zone2 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "Zone4":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
  Zone3 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match1 GridConnectivity_t "BaseA/Zone1":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
      match2 GridConnectivity_t "BaseB/Zone6":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
  Zone4 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "Zone2":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
"""
yt1b = """
CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
BaseB CGNSBase_t:
  Zone5 Zone_t:
  Zone6 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "BaseA/Zone3":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
"""
yt2a = """
CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
BaseA CGNSBase_t:
  Zone1 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "Zone3":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
  Zone3 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match2 GridConnectivity_t "BaseB/Zone6":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
  Zone4 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "Zone2":
"""
yt2b = """
CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
BaseA CGNSBase_t:
  Zone1 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "Zone3":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
  Zone2 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "Zone4":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
  Zone3 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match1 GridConnectivity_t "BaseA/Zone1":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
  Zone4 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "Zone2":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
BaseB CGNSBase_t:
  Zone5 Zone_t:
  Zone6 Zone_t:
    ZGC ZoneGridConnectivity_t:
      match GridConnectivity_t "BaseA/Zone3":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
"""

def test_union():
  union_yt = """
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
  BaseA CGNSBase_t:
    Zone1 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone2 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone4":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "BaseA/Zone1":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
        match2 GridConnectivity_t "BaseB/Zone6":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone4 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone2":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  BaseB CGNSBase_t:
    Zone5 Zone_t:
    Zone6 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "BaseA/Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """
  ref_union_tree = parse_yaml_cgns.to_cgns_tree(union_yt)
  
  tree1a = parse_yaml_cgns.to_cgns_tree(yt1a)
  tree1b = parse_yaml_cgns.to_cgns_tree(yt1b)
  
  union_tree1ab = PT.deep_copy(PLO.union(tree1a, tree1b))
  assert PT.is_same_tree(union_tree1ab, ref_union_tree)
  
  union_tree1ba = PT.deep_copy(PLO.union(tree1b, tree1a))
  assert PT.is_same_tree(union_tree1ba, ref_union_tree)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  union_tree2ab = PT.deep_copy(PLO.union(tree2a, tree2b))
  assert PT.is_same_tree(union_tree2ab, ref_union_tree)
  
  union_tree2ba = PT.deep_copy(PLO.union(tree2b, tree2a))
  assert PT.is_same_tree(union_tree2ba, ref_union_tree)


def test_intersection():
  intersect_yt1 = """
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
  """
  intersect_yt2 = """
  BaseA CGNSBase_t:
    Zone1 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
    Zone4 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone2":
  """
  ref_intersect_tree1 = parse_yaml_cgns.to_cgns_tree(intersect_yt1)
  ref_intersect_tree2 = parse_yaml_cgns.to_cgns_tree(intersect_yt2)
  
  tree1a = parse_yaml_cgns.to_cgns_tree(yt1a)
  tree1b = parse_yaml_cgns.to_cgns_tree(yt1b)
  
  intersect_tree1ab = PT.deep_copy(PLO.intersection(tree1a, tree1b))
  assert PT.is_same_tree(intersect_tree1ab, ref_intersect_tree1)
  
  intersect_tree1ba = PT.deep_copy(PLO.intersection(tree1b, tree1a))
  assert PT.is_same_tree(intersect_tree1ba, ref_intersect_tree1)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  intersect_tree2ab = PT.deep_copy(PLO.intersection(tree2a, tree2b))
  assert PT.is_same_tree(intersect_tree2ab, ref_intersect_tree2)
  
  intersect_tree2ba = PT.deep_copy(PLO.intersection(tree2b, tree2a))
  assert PT.is_same_tree(intersect_tree2ba, ref_intersect_tree2)


def test_diff():
  diff_yt1ab = """
  BaseA CGNSBase_t:
    Zone1 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone2 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone4":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "BaseA/Zone1":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
        match2 GridConnectivity_t "BaseB/Zone6":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone4 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone2":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """
  diff_yt1ba = """
  BaseB CGNSBase_t:
    Zone5 Zone_t:
    Zone6 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "BaseA/Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """
  diff_yt2ab = """
  BaseA CGNSBase_t:
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match2 GridConnectivity_t "BaseB/Zone6":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """
  diff_yt2ba = """
  BaseA CGNSBase_t:
    Zone2 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone4":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "BaseA/Zone1":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone4 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone2":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  BaseB CGNSBase_t:
    Zone5 Zone_t:
    Zone6 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "BaseA/Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """
  ref_diff_tree1ab = parse_yaml_cgns.to_cgns_tree(diff_yt1ab)
  ref_diff_tree1ba = parse_yaml_cgns.to_cgns_tree(diff_yt1ba)
  ref_diff_tree2ab = parse_yaml_cgns.to_cgns_tree(diff_yt2ab)
  ref_diff_tree2ba = parse_yaml_cgns.to_cgns_tree(diff_yt2ba)
  #Need to delete CGNSLibraryVersion_t because to_cgns_tree creates it
  PT.rm_node_from_path(ref_diff_tree1ab, 'CGNSLibraryVersion')
  PT.rm_node_from_path(ref_diff_tree1ba, 'CGNSLibraryVersion')
  PT.rm_node_from_path(ref_diff_tree2ab, 'CGNSLibraryVersion')
  PT.rm_node_from_path(ref_diff_tree2ba, 'CGNSLibraryVersion')
  
  tree1a = parse_yaml_cgns.to_cgns_tree(yt1a)
  tree1b = parse_yaml_cgns.to_cgns_tree(yt1b)
  
  diff_tree1ab = PT.deep_copy(PLO.diff(tree1a, tree1b))
  assert PT.is_same_tree(diff_tree1ab, ref_diff_tree1ab)
  
  diff_tree1ba = PT.deep_copy(PLO.diff(tree1b, tree1a))
  assert PT.is_same_tree(diff_tree1ba, ref_diff_tree1ba)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  diff_tree2ab = PT.deep_copy(PLO.diff(tree2a, tree2b))
  assert PT.is_same_tree(diff_tree2ab, ref_diff_tree2ab)
  
  diff_tree2ba = PT.deep_copy(PLO.diff(tree2b, tree2a))
  assert PT.is_same_tree(diff_tree2ba, ref_diff_tree2ba)


def test_label_differ():
  tree1 = PT.new_CGNSBase()
  tree2 = PT.new_Zone()
  
  with pytest.raises(TypeError):
    union_tree = PT.deep_copy(PLO.union(tree1, tree2))
  
  with pytest.raises(TypeError):
    intersect_tree = PT.deep_copy(PLO.intersection(tree1, tree2))
  
  with pytest.raises(TypeError):
    diff_tree = PT.deep_copy(PLO.diff(tree1, tree2))


def test_with_check():
  yt1 = """
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
  BaseA CGNSBase_t:
    Zone3 Zone_t:
      ZoneBC ZoneBC_t:
        BC1 BC_t "BCInflow":
  """
  yt2 = """
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
  BaseA CGNSBase_t:
    Zone3 Zone_t:
      ZoneBC ZoneBC_t:
        BC1 UserDefinedData_t "Test":
  """
  tree1 = parse_yaml_cgns.to_cgns_tree(yt1)
  tree2 = parse_yaml_cgns.to_cgns_tree(yt2)
  
  root_tree = PT.new_node('CGNSTree', 'CGNSTree_t')
  
  ref_intersect_tree = PT.shallow_copy(tree1)
  PT.rm_nodes_from_name(ref_intersect_tree, 'BC1')
  
  ref_diff_tree = PT.shallow_copy(tree1)
  PT.rm_node_from_path(ref_diff_tree, 'CGNSLibraryVersion')
  
  intersect_tree = PT.deep_copy(PLO.intersection(tree1, tree2))
  assert PT.is_same_tree(intersect_tree, tree1)
  
  intersect_tree = PT.deep_copy(PLO.intersection(tree1, tree2, check_label=True))
  PT.print_tree(intersect_tree)
  assert PT.is_same_tree(intersect_tree, ref_intersect_tree)
  
  intersect_tree = PT.deep_copy(PLO.intersection(tree1, tree2, check_value=True))
  assert PT.is_same_tree(intersect_tree, ref_intersect_tree)
  
  
  diff_tree = PT.deep_copy(PLO.diff(tree1, tree2))
  assert PT.is_same_tree(diff_tree, root_tree)
  
  diff_tree = PT.deep_copy(PLO.diff(tree1, tree2, check_label=True))
  assert PT.is_same_tree(diff_tree, ref_diff_tree)
  
  diff_tree = PT.deep_copy(PLO.diff(tree1, tree2, check_value=True))
  assert PT.is_same_tree(diff_tree, ref_diff_tree)


@pytest.mark.parametrize("CHECK_LABEL", [False, True])
@pytest.mark.parametrize("CHECK_VALUE", [False, True])
def test_combination1(CHECK_LABEL,CHECK_VALUE):
  # As reminder, in logical operations, we have:
  # a-b = a-intersect(a,b) = union(a,b)-b
  
  tree1a = parse_yaml_cgns.to_cgns_tree(yt1a)
  tree1b = parse_yaml_cgns.to_cgns_tree(yt1b)
  
  diff_tree1ab              = PT.deep_copy(PLO.diff(        tree1a,        tree1b,            check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  intersect_tree1ab         = PT.deep_copy(PLO.intersection(tree1a,        tree1b,            check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  union_tree1ab             = PT.deep_copy(PLO.union(       tree1a,        tree1b                                                             ))
  diff_tree1a_intersect1ab  = PT.deep_copy(PLO.diff(        tree1a,        intersect_tree1ab, check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  diff_union_tree1ab_tree1b = PT.deep_copy(PLO.diff(        union_tree1ab, tree1b,            check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  
  assert PT.is_same_tree(diff_tree1ab, diff_tree1a_intersect1ab)
  assert PT.is_same_tree(diff_tree1ab, diff_union_tree1ab_tree1b)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  diff_tree2ab              = PT.deep_copy(PLO.diff(        tree2a,        tree2b,            check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  intersect_tree2ab         = PT.deep_copy(PLO.intersection(tree2a,        tree2b,            check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  union_tree2ab             = PT.deep_copy(PLO.union(       tree2a,        tree2b                                                             ))
  diff_tree2a_intersect2ab  = PT.deep_copy(PLO.diff(        tree2a,        intersect_tree2ab, check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  diff_union_tree2ab_tree2b = PT.deep_copy(PLO.diff(        union_tree2ab, tree2b,            check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  
  assert PT.is_same_tree(diff_tree2ab, diff_tree2a_intersect2ab)
  assert PT.is_same_tree(diff_tree2ab, diff_union_tree2ab_tree2b)
  


@pytest.mark.parametrize("CHECK_LABEL", [False, True])
@pytest.mark.parametrize("CHECK_VALUE", [False, True])
def test_combination2(CHECK_LABEL,CHECK_VALUE):
  # As reminder, in logical operations, we have:
  # union(a,b) = union(union(a-b, b-a),intersect(a,b))
  
  tree1a = parse_yaml_cgns.to_cgns_tree(yt1a)
  tree1b = parse_yaml_cgns.to_cgns_tree(yt1b)
  
  union_tree1ab                = PT.deep_copy(PLO.union(       tree1a,                tree1b                                                            ))
  diff_tree1ab                 = PT.deep_copy(PLO.diff(        tree1a,                tree1b,           check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  diff_tree1ba                 = PT.deep_copy(PLO.diff(        tree1b,                tree1a,           check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  union_diff1ab_diff1ba        = PT.deep_copy(PLO.union(       diff_tree1ab,          diff_tree1ba                                                      ))
  intersect_tree1ab            = PT.deep_copy(PLO.intersection(tree1a,                tree1b,           check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  union_uniond1abd1ba_inter1ab = PT.deep_copy(PLO.union(       union_diff1ab_diff1ba, intersect_tree1ab                                                 ))
  
  assert PT.is_same_tree(union_tree1ab, union_uniond1abd1ba_inter1ab)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  union_tree2ab                = PT.deep_copy(PLO.union(       tree2a,                tree2b                                                            ))
  diff_tree2ab                 = PT.deep_copy(PLO.diff(        tree2a,                tree2b,           check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  diff_tree2ba                 = PT.deep_copy(PLO.diff(        tree2b,                tree2a,           check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  union_diff2ab_diff2ba        = PT.deep_copy(PLO.union(       diff_tree2ab,          diff_tree2ba                                                      ))
  intersect_tree2ab            = PT.deep_copy(PLO.intersection(tree2a,                tree2b,           check_label=CHECK_LABEL, check_value=CHECK_VALUE))
  union_uniond2abd2ba_inter2ab = PT.deep_copy(PLO.union(       union_diff2ab_diff2ba, intersect_tree2ab                                                 ))
  
  assert PT.is_same_tree(union_tree2ab, union_uniond2abd2ba_inter2ab)
  
  
