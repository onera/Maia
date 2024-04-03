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
  
  union_tree1 = PLO.union(tree1a, tree1b)
  assert PT.is_same_tree(union_tree1, ref_union_tree)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  union_tree2 = PLO.union(tree2a, tree2b)
  assert PT.is_same_tree(union_tree2, ref_union_tree)

def test_intersection():
  intersect_yt1 = """
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
  """
  intersect_yt2 = """
  CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
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
  
  intersect_tree1 = PLO.intersection(tree1a, tree1b)
  assert PT.is_same_tree(intersect_tree1, ref_intersect_tree1)
  
  tree2a = parse_yaml_cgns.to_cgns_tree(yt2a)
  tree2b = parse_yaml_cgns.to_cgns_tree(yt2b)
  
  intersect_tree2 = PLO.intersection(tree2a, tree2b)
  PT.print_tree(intersect_tree2)
  PT.print_tree(ref_intersect_tree2)
  assert PT.is_same_tree(intersect_tree2, ref_intersect_tree2)
