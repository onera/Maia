import pytest

import Converter.Internal as I

from maia.sids import pytree as PT

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

if __name__ == "__main__":
  test_node_walker_pattern()
