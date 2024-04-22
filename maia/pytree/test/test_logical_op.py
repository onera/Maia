import pytest

import maia.pytree            as PT
import maia.pytree.compare    as PTC
import maia.pytree.logical_op as PLO


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
  ref_union_tree = PT.yaml.to_cgns_tree("""
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
  """)
  
  tree1a = PT.yaml.to_cgns_tree(yt1a)
  tree1b = PT.yaml.to_cgns_tree(yt1b)
  
  assert PT.is_same_tree(PLO.union(tree1a, tree1b), ref_union_tree)
  assert PT.is_same_tree(PLO.union(tree1b, tree1a), ref_union_tree)
  # Input should not be modified 
  assert PT.is_same_tree(tree1a, PT.yaml.to_cgns_tree(yt1a)) and \
         PT.is_same_tree(tree1b, PT.yaml.to_cgns_tree(yt1b))
  
  tree2a = PT.yaml.to_cgns_tree(yt2a)
  tree2b = PT.yaml.to_cgns_tree(yt2b)
  
  assert PT.is_same_tree(PLO.union(tree2a, tree2b), ref_union_tree)
  assert PT.is_same_tree(PLO.union(tree2b, tree2a), ref_union_tree)


def test_intersection():
  ref_intersect_tree1 = PT.yaml.to_cgns_tree("""
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
  """)
  ref_intersect_tree2 = PT.yaml.to_cgns_tree("""
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
  """)
  
  tree1a = PT.yaml.to_cgns_tree(yt1a)
  tree1b = PT.yaml.to_cgns_tree(yt1b)
  
  assert PT.is_same_tree(PLO.intersection(tree1a, tree1b), ref_intersect_tree1)
  assert PT.is_same_tree(PLO.intersection(tree1b, tree1a), ref_intersect_tree1)
  # Input should not be modified 
  assert PT.is_same_tree(tree1a, PT.yaml.to_cgns_tree(yt1a)) and \
         PT.is_same_tree(tree1b, PT.yaml.to_cgns_tree(yt1b))
  
  tree2a = PT.yaml.to_cgns_tree(yt2a)
  tree2b = PT.yaml.to_cgns_tree(yt2b)
  
  assert PT.is_same_tree(PLO.intersection(tree2a, tree2b), ref_intersect_tree2)
  assert PT.is_same_tree(PLO.intersection(tree2b, tree2a), ref_intersect_tree2)


def test_difference():
  ref_diff_tree1ab = PT.yaml.to_cgns_tree("""
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
  """)
  ref_diff_tree1ba = PT.yaml.to_cgns_tree("""
  BaseB CGNSBase_t:
    Zone5 Zone_t:
    Zone6 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "BaseA/Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """)
  ref_diff_tree2ab = PT.yaml.to_cgns_tree("""
  BaseA CGNSBase_t:
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match2 GridConnectivity_t "BaseB/Zone6":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """)
  ref_diff_tree2ba = PT.yaml.to_cgns_tree("""
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
  """)
  #Need to delete CGNSLibraryVersion_t because to_cgns_tree creates it
  for tree in [ref_diff_tree1ab, ref_diff_tree1ba, 
               ref_diff_tree2ab, ref_diff_tree2ba]:
    PT.rm_node_from_path(tree, 'CGNSLibraryVersion')
  
  tree1a = PT.yaml.to_cgns_tree(yt1a)
  tree1b = PT.yaml.to_cgns_tree(yt1b)
  
  assert PT.is_same_tree(PLO.difference(tree1a, tree1b), ref_diff_tree1ab)
  assert PT.is_same_tree(PLO.difference(tree1b, tree1a), ref_diff_tree1ba)
  # Input should not be modified 
  assert PT.is_same_tree(tree1a, PT.yaml.to_cgns_tree(yt1a)) and \
         PT.is_same_tree(tree1b, PT.yaml.to_cgns_tree(yt1b))
  
  tree2a = PT.yaml.to_cgns_tree(yt2a)
  tree2b = PT.yaml.to_cgns_tree(yt2b)
  
  assert PT.is_same_tree(PLO.difference(tree2a, tree2b), ref_diff_tree2ab)
  assert PT.is_same_tree(PLO.difference(tree2b, tree2a), ref_diff_tree2ba)


def test_names_differ():
  tree1 = PT.new_CGNSBase()
  tree2 = PT.new_Zone()
  
  with pytest.raises(ValueError):
    PLO.union(tree1, tree2)
  
  with pytest.raises(ValueError):
    PLO.intersection(tree1, tree2)
  
  with pytest.raises(ValueError):
    PLO.difference(tree1, tree2)


def test_comp_func():
  tree1 = PT.yaml.to_cgns_tree("""
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
  BaseA CGNSBase_t:
    Zone3 Zone_t:
      ZoneBC ZoneBC_t:
        BC1 BC_t "BCInflow":
  """)
  tree2 = PT.yaml.to_cgns_tree("""
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
  BaseA CGNSBase_t:
    Zone3 Zone_t:
      ZoneBC ZoneBC_t:
        BC1 UserDefinedData_t "Test":
  """)
  
  ref_intersect_tree = PT.shallow_copy(tree1)
  PT.rm_nodes_from_name(ref_intersect_tree, 'BC1')
  
  # Name comparing : BC1 is preserved
  assert PT.is_same_tree(PLO.intersection(tree1, tree2, comp_func=PTC.is_same_name),
                         tree1)
  
  # Other comparaisons : BC1 is removed
  for func in [PTC.is_same_node, PTC.is_same_label, PTC.is_same_value]:
    assert PT.is_same_tree(PLO.intersection(tree1, tree2, comp_func=func),
                          ref_intersect_tree)
  
  
  ref_diff_tree = PT.shallow_copy(tree1)
  PT.rm_node_from_path(ref_diff_tree, 'CGNSLibraryVersion')
  
  # Name comparing : tree are identical -> return root
  assert PT.is_same_tree(PLO.difference(tree1, tree2, comp_func=PTC.is_same_name),
                         PT.new_node('CGNSTree', 'CGNSTree_t'))
  
  # Other comparaisons : only LibVersion is removed
  for func in [PTC.is_same_node, PTC.is_same_label, PTC.is_same_value]:
    assert PT.is_same_tree(PLO.difference(tree1, tree2, comp_func=func),
                          ref_diff_tree)
  

@pytest.mark.parametrize("COMP_FUNC", [PTC.is_same_node, PTC.is_same_name, PTC.is_same_label])
def test_combination(COMP_FUNC):
  # As reminder, in logical operations, we have:
  # a-b = a-intersect(a,b) = union(a,b)-b
  
  tree1a = PT.yaml.to_cgns_tree(yt1a)
  tree1b = PT.yaml.to_cgns_tree(yt1b)
  
  diff_tree1ab              = PLO.difference  (tree1a,        tree1b,            comp_func=COMP_FUNC)
  intersect_tree1ab         = PLO.intersection(tree1a,        tree1b,            comp_func=COMP_FUNC)
  union_tree1ab             = PLO.union       (tree1a,        tree1b                                )
  diff_tree1a_intersect1ab  = PLO.difference  (tree1a,        intersect_tree1ab, comp_func=COMP_FUNC)
  diff_union_tree1ab_tree1b = PLO.difference  (union_tree1ab, tree1b,            comp_func=COMP_FUNC)
  
  assert PT.is_same_tree(diff_tree1ab, diff_tree1a_intersect1ab)
  assert PT.is_same_tree(diff_tree1ab, diff_union_tree1ab_tree1b)

  # As reminder, in logical operations, we have:
  # union(a,b) = union(union(a-b, b-a),intersect(a,b))
  
  tree2a = PT.yaml.to_cgns_tree(yt2a)
  tree2b = PT.yaml.to_cgns_tree(yt2b)
  
  union_tree2ab                = PLO.union       (tree2a,                tree2b                               )
  diff_tree2ab                 = PLO.difference  (tree2a,                tree2b,           comp_func=COMP_FUNC)
  diff_tree2ba                 = PLO.difference  (tree2b,                tree2a,           comp_func=COMP_FUNC)
  union_diff2ab_diff2ba        = PLO.union       (diff_tree2ab,          diff_tree2ba                         )
  intersect_tree2ab            = PLO.intersection(tree2a,                tree2b,           comp_func=COMP_FUNC)
  union_uniond2abd2ba_inter2ab = PLO.union       (union_diff2ab_diff2ba, intersect_tree2ab                    )
  
  assert PT.is_same_tree(union_tree2ab, union_uniond2abd2ba_inter2ab)
