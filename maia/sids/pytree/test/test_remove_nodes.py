import pytest

import Converter.Internal as I
from maia.sids import pytree as PT

from maia.utils        import parse_yaml_cgns

yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZBCA ZoneBC_t:
      bca1 BC_t:
        FamilyName FamilyName_t 'BCC1':
        Index_i IndexArray_t:
      bcd2 BC_t:
        FamilyName FamilyName_t 'BCA2':
        Index_ii IndexArray_t:
    FamilyName FamilyName_t 'ROW1':
    ZBCB ZoneBC_t:
      bcb3 BC_t:
        FamilyName FamilyName_t 'BCD3':
        Index_iii IndexArray_t:
      bce4 BC_t:
        FamilyName FamilyName_t 'BCE4':
      bcc5 BC_t:
        FamilyName FamilyName_t 'BCB5':
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""

def test_rmChildrenFromPredicate():
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  for bc_node in PT.iterNodesFromLabel(tree, "BC_t"):
    PT.rmChildrenFromPredicate(bc_node, lambda n: I.getType(n) == "FamilyName_t" and int(I.getValue(n)[-1]) > 4)
  assert [I.getValue(n) for n in PT.getNodesFromLabel(tree, "FamilyName_t", search='dfs')] == ['BCC1', 'BCA2', 'ROW1', 'BCD3', 'BCE4']

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  for bc_node in PT.iterNodesFromLabel(tree, "BC_t"):
    PT.keepChildrenFromPredicate(bc_node, lambda n: I.getType(n) == "FamilyName_t" and int(I.getValue(n)[-1]) > 4)
  assert [I.getValue(n) for n in PT.getNodesFromLabel(tree, "FamilyName_t")] == ['ROW1', 'BCB5']

def test_rmNodesFromPredicate():
  # Camel case
  # ==========
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rmNodesFromPredicate(tree, lambda n: I.getType(n) == "FamilyName_t")
  assert [I.getName(n) for n in PT.getNodesFromLabel(tree, "FamilyName_t")] == []

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rmNodesFromPredicate3(tree, lambda n: I.getType(n) == "FamilyName_t")
  assert [I.getName(n) for n in PT.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

  # Name
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rmNodesFromName(tree, "FamilyName")
  assert [I.getName(n) for n in I.getNodesFromName(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rmNodesFromName3(tree, "FamilyName")
  assert [I.getName(n) for n in I.getNodesFromName(tree, "FamilyName")] == ["FamilyName"]*5

  # Label
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rmNodesFromLabel(tree, "FamilyName_t")
  assert [I.getName(n) for n in I.getNodesFromName(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rmNodesFromLabel3(tree, "FamilyName_t")
  assert [I.getName(n) for n in PT.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

  # Snake case
  # ==========
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rm_nodes_from_predicate(tree, lambda n: I.getType(n) == "FamilyName_t")
  assert [I.getValue(n) for n in PT.get_nodes_from_label(tree, "FamilyName_t")] == []

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rm_nodes_from_predicate3(tree, lambda n: I.getType(n) == "FamilyName_t")
  assert [I.getValue(n) for n in PT.get_nodes_from_label(tree, "FamilyName_t")] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # Name
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rm_nodes_from_name(tree, "FamilyName")
  assert [I.getValue(n) for n in PT.get_nodes_from_name(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rm_nodes_from_name3(tree, "FamilyName")
  assert [I.getValue(n) for n in PT.get_nodes_from_name(tree, "FamilyName")] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # Label
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rm_nodes_from_label(tree, "FamilyName_t")
  assert [I.getName(n) for n in PT.get_nodes_from_label(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  PT.rm_nodes_from_label3(tree, "FamilyName_t")
  assert [I.getName(n) for n in PT.get_nodes_from_label(tree, "FamilyName_t")] == ["FamilyName"]*5
