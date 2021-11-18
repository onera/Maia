import pytest
import numpy as np

import Converter.Internal as I

from maia.sids.cgns_keywords import Label as CGL
from maia.sids import pytree as PT
from maia.sids.pytree import predicate as PD

from maia.utils import parse_yaml_cgns

yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    NgonI Elements_t [22,0]:
    ZBCAI ZoneBC_t:
      bc1I BC_t:
        Index_i IndexArray_t:
        PL1I DataArray_t:
      bc2 BC_t:
        Index_ii IndexArray_t:
        PL2 DataArray_t:
  ZoneJ Zone_t:
    NgonJ Elements_t [22,0]:
    ZBCAJ ZoneBC_t:
      bc1J BC_t:
        Index_j IndexArray_t:
        PL1J DataArray_t:
    ZBCBJ ZoneBC_t:
      bc3J BC_t:
        Index_jjj IndexArray_t:
        PL3J DataArray_t:
"""

@pytest.mark.parametrize("search", ['dfs', 'bfs'])
def test_node_walker_auto(search):
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  # Search with name
  walker = PT.NodeWalker(tree, 'Zone*', search=search)
  n = walker()
  assert(PT.get_name(n) == "ZoneI")

  # Search with label as str
  walker = PT.NodeWalker(tree, 'Zone_t', search=search)
  n = walker()
  assert(PT.get_name(n) == "ZoneI")

  # Search with label as CGL
  walker = PT.NodeWalker(tree, CGL.Zone_t, search=search)
  n = walker()
  assert(PT.get_name(n) == "ZoneI")

  # Search with value
  walker = PT.NodeWalker(tree, np.array([22,0]), search=search)
  n = walker()
  assert(PT.get_name(n) == "NgonI")

  # Search with callable
  walker = PT.NodeWalker(tree, lambda n: PT.get_label(n) == "Zone_t", search=search)
  n = walker()
  assert(PT.get_name(n) == "ZoneI")


@pytest.mark.parametrize("search", ['dfs', 'bfs'])
def test_node_walker_depth(search):
  yt = """
Base CGNSBase_t:
  FamilyBCOut Family_t:
    FamilyBC FamilyBC_t:
      FamilyBCDataSet FamilyBCDataSet_t:
        RefStateFamilyBCDataSet ReferenceState_t:
          Density DataArray_t [1.1]:
          MomentumX DataArray_t [1.1]:
          MomentumY DataArray_t [0.]:
          MomentumZ DataArray_t [0.]:
          EnergyStagnationDensity DataArray_t [2.51]:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  # Test depth max
  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "RefStateFamilyBCDataSet"), depth=[1,2], search=search)()
  assert(n is None)

  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "RefStateFamilyBCDataSet"), depth=[1,3], search=search)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

  # Test depth min
  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "FamilyBCDataSet"), depth=[2,4], search=search)()
  assert(PT.get_name(n) == "FamilyBCDataSet")

  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "FamilyBCDataSet"), depth=[2,None], search=search)()
  assert(PT.get_name(n) == "FamilyBCDataSet")

  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "FamilyBCDataSet"), depth=[3,4], search=search)()
  assert(n is None)

  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "FamilyBCDataSet"), depth=[3,None], search=search)()
  assert(n is None)

  # Test change root
  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "Density"), depth=[3,4], search=search)()
  assert(PT.get_name(n) == "Density")

  root = PT.get_node_from_label(tree, "CGNSBase_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "Density"), depth=[3,4], search=search)()
  assert(n is None)


@pytest.mark.parametrize("search", ['dfs', 'bfs'])
def test_node_walker_pattern(search):
  yt = """
FamilyBCDataSet FamilyBCDataSet_t:
  RefStateFamilyBCDataSet ReferenceState_t:
    Density DataArray_t [1.1]:
    MomentumX DataArray_t [1.1]:
    MomentumY DataArray_t [0.]:
    MomentumZ DataArray_t [0.]:
    EnergyStagnationDensity DataArray_t [2.51]:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  check_name = lambda name : lambda n: PT.get_node_from_name(n, name, depth=1) is not None
  pattern  = lambda n : PT.get_label(n) == 'ReferenceState_t' and [check_name(i) for i in ["Density", "MomentumX"]]

  root = PT.get_node_from_label(tree, "FamilyBCDataSet_t")
  n = PT.NodeWalker(root, pattern, depth=0, search=search)()
  assert(n is None)

  root = PT.get_node_from_label(tree, "ReferenceState_t")
  n = PT.NodeWalker(root, pattern, depth=0, search=search)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

  root = PT.get_node_from_label(tree, "FamilyBCDataSet_t")
  n = PT.NodeWalker(root, pattern, depth=1, search=search)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

  root = PT.get_node_from_label(tree, "FamilyBCDataSet_t")
  n = PT.NodeWalker(root, pattern, depth=None, search=search)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

  root = PT.get_node_from_label(tree, "FamilyBCDataSet_t")
  n = PT.NodeWalker(root, pattern, search=search)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")
