import pytest

import maia.pytree as PT
from maia.utils.yaml   import parse_yaml_cgns

yt = """
Base CGNSBase_t:
  Zone Zone_t:
    ZBC ZoneBC_t:
      bc1 BC_t:
        FamilyName FamilyName_t 'BC1':
        Index_i IndexArray_t:
      bc2 BC_t:
        FamilyName FamilyName_t 'BC2':
        Index_ii IndexArray_t:
    FamilyName FamilyName_t 'ROW1':
"""

def values(nodes):
  return [PT.get_value(n) for n in nodes]

def test_rm_children_from_predicate():
  tree = parse_yaml_cgns.to_node(yt)
  for bc_node in PT.iterNodesFromLabel(tree, "BC_t"):
    PT.rm_children_from_predicate(bc_node, lambda n: PT.match_label(n, "FamilyName_t") and int(PT.get_value(n)[-1]) > 1)
  assert values(PT.getNodesFromLabel(tree, "FamilyName_t")) == ['BC1', 'ROW1']

def test_keep_children_from_predicate():
  tree = parse_yaml_cgns.to_node(yt)
  for bc_node in PT.iterNodesFromLabel(tree, "BC_t"):
    PT.keep_children_from_predicate(bc_node, lambda n: PT.match_label(n, "FamilyName_t") and int(PT.get_value(n)[-1]) > 1)
  assert values(PT.getNodesFromLabel(tree, "FamilyName_t")) == ['BC2', 'ROW1']

def test_rm_nodes_from_predicate():
  tree = parse_yaml_cgns.to_node(yt)
  PT.rm_nodes_from_predicate(tree, lambda n: PT.match_label(n, "FamilyName_t"))
  assert len(PT.getNodesFromLabel(tree, "FamilyName_t")) == 0

  tree = parse_yaml_cgns.to_node(yt)
  PT.rm_nodes_from_predicate(tree, lambda n: PT.match_label(n, "FamilyName_t"), depth=1)
  assert len(PT.getNodesFromLabel(tree, "FamilyName_t")) == 3

  tree = parse_yaml_cgns.to_node(yt)
  PT.rm_nodes_from_predicate(tree, lambda n: PT.match_label(n, "FamilyName_t"), depth=3)
  assert len(PT.getNodesFromLabel(tree, "FamilyName_t")) == 2

