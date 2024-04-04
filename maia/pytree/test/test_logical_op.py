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
  
  union_tree1ab = PLO.union(tree1a, tree1b, copy=True)
  assert PT.is_same_tree(union_tree1ab, ref_union_tree)
  union_tree1ba = PLO.union(tree1b, tree1a)
  assert PT.is_same_tree(union_tree1ba, ref_union_tree)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  union_tree2ab = PLO.union(tree2a, tree2b, copy=True)
  assert PT.is_same_tree(union_tree2ab, ref_union_tree)
  union_tree2ba = PLO.union(tree2b, tree2a)
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
  
  intersect_tree1ab = PLO.intersection(tree1a, tree1b, copy=True)
  assert PT.is_same_tree(intersect_tree1ab, ref_intersect_tree1)
  intersect_tree1ba = PLO.intersection(tree1b, tree1a)
  assert PT.is_same_tree(intersect_tree1ba, ref_intersect_tree1)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  intersect_tree2ab = PLO.intersection(tree2a, tree2b, copy=True)
  assert PT.is_same_tree(intersect_tree2ab, ref_intersect_tree2)
  intersect_tree2ba = PLO.intersection(tree2b, tree2a)
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
  
  diff_tree1ab = PLO.diff(tree1a, tree1b, copy=True)
  assert PT.is_same_tree(diff_tree1ab, ref_diff_tree1ab)
  diff_tree1ba = PLO.diff(tree1b, tree1a)
  assert PT.is_same_tree(diff_tree1ba, ref_diff_tree1ba)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  diff_tree2ab = PLO.diff(tree2a, tree2b, copy=True)
  assert PT.is_same_tree(diff_tree2ab, ref_diff_tree2ab)
  diff_tree2ba = PLO.diff(tree2b, tree2a)
  assert PT.is_same_tree(diff_tree2ba, ref_diff_tree2ba)


def test_combination1():
  # As reminder in logical operations, we have:
  # a-b = a-intersect(a,b) = union(a,b)-b
  
  tree1a = parse_yaml_cgns.to_cgns_tree(yt1a)
  tree1b = parse_yaml_cgns.to_cgns_tree(yt1b)
  
  diff_tree1ab              = PLO.diff(        tree1a,        tree1b,            copy=True)
  intersect_tree1ab         = PLO.intersection(tree1a,        tree1b,            copy=True)
  union_tree1ab             = PLO.union(       tree1a,        tree1b,            copy=True)
  diff_tree1a_intersect1ab  = PLO.diff(        tree1a,        intersect_tree1ab, copy=True)
  diff_union_tree1ab_tree1b = PLO.diff(        union_tree1ab, tree1b,            copy=True)
  
  assert PT.is_same_tree(diff_tree1ab, diff_tree1a_intersect1ab)
  assert PT.is_same_tree(diff_tree1ab, diff_union_tree1ab_tree1b)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  diff_tree2ab              = PLO.diff(        tree2a,        tree2b,            copy=True)
  intersect_tree2ab         = PLO.intersection(tree2a,        tree2b,            copy=True)
  union_tree2ab             = PLO.union(       tree2a,        tree2b,            copy=True)
  diff_tree2a_intersect2ab  = PLO.diff(        tree2a,        intersect_tree2ab, copy=True)
  diff_union_tree2ab_tree2b = PLO.diff(        union_tree2ab, tree2b,            copy=True)
  
  assert PT.is_same_tree(diff_tree2ab, diff_tree2a_intersect2ab)
  assert PT.is_same_tree(diff_tree2ab, diff_union_tree2ab_tree2b)
  


def test_combination2():
  # As reminder in logical operations, we have:
  # union(a,b) = union(union(a-b, b-a),intersect(a,b))
  
  tree1a = parse_yaml_cgns.to_cgns_tree(yt1a)
  tree1b = parse_yaml_cgns.to_cgns_tree(yt1b)
  
  union_tree1ab                = PLO.union(       tree1a,                tree1b,            copy=True)
  diff_tree1ab                 = PLO.diff(        tree1a,                tree1b,            copy=True)
  diff_tree1ba                 = PLO.diff(        tree1b,                tree1a,            copy=True)
  union_diff1ab_diff1ba        = PLO.union(       diff_tree1ab,          diff_tree1ba,      copy=True)
  intersect_tree1ab            = PLO.intersection(tree1a,                tree1b,            copy=True)
  union_uniond1abd1ba_inter1ab = PLO.union(       union_diff1ab_diff1ba, intersect_tree1ab, copy=True)
  
  assert PT.is_same_tree(union_tree1ab, union_uniond1abd1ba_inter1ab)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  union_tree2ab                = PLO.union(       tree2a,                tree2b,            copy=True)
  diff_tree2ab                 = PLO.diff(        tree2a,                tree2b,            copy=True)
  diff_tree2ba                 = PLO.diff(        tree2b,                tree2a,            copy=True)
  union_diff2ab_diff2ba        = PLO.union(       diff_tree2ab,          diff_tree2ba,      copy=True)
  intersect_tree2ab            = PLO.intersection(tree2a,                tree2b,            copy=True)
  union_uniond2abd2ba_inter2ab = PLO.union(       union_diff2ab_diff2ba, intersect_tree2ab, copy=True)
  
  assert PT.is_same_tree(union_tree2ab, union_uniond2abd2ba_inter2ab)
