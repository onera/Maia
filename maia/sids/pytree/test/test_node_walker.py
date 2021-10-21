import pytest

import Converter.Internal as I

from maia.sids import pytree as PT
from maia.sids.pytree import predicate as PD

from maia.utils import parse_yaml_cgns

def test_node_walker_pattern():
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
  n = PT.NodeWalker(root, pattern, depth=0)()
  assert(n is None)

  root = PT.get_node_from_label(tree, "ReferenceState_t")
  n = PT.NodeWalker(root, pattern, depth=0)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

  root = PT.get_node_from_label(tree, "FamilyBCDataSet_t")
  n = PT.NodeWalker(root, pattern, depth=1)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

  root = PT.get_node_from_label(tree, "FamilyBCDataSet_t")
  n = PT.NodeWalker(root, pattern, depth=None)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

  root = PT.get_node_from_label(tree, "FamilyBCDataSet_t")
  n = PT.NodeWalker(root, pattern)()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

def test_node_walker_depth():
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
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "RefStateFamilyBCDataSet"), depth=[1,2])()
  assert(n is None)

  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "RefStateFamilyBCDataSet"), depth=[1,3])()
  assert(PT.get_name(n) == "RefStateFamilyBCDataSet")

  # Test depth min
  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "FamilyBCDataSet"), depth=[2,4])()
  assert(PT.get_name(n) == "FamilyBCDataSet")

  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "FamilyBCDataSet"), depth=[3,4])()
  assert(n is None)

  # Test change root
  root = PT.get_node_from_label(tree, "Family_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "Density"), depth=[3,4])()
  assert(PT.get_name(n) == "Density")

  root = PT.get_node_from_label(tree, "CGNSBase_t")
  n = PT.NodeWalker(root, lambda n: PD.match_name(n, "Density"), depth=[3,4])()
  assert(n is None)

if __name__ == "__main__":
  # test_node_walker_pattern()
  test_node_walker_depth()
