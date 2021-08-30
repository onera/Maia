import pytest

import Converter.Internal as I

from maia.sids import pytree as PT

from maia.utils        import parse_yaml_cgns

def test_NodesWalkers():
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
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  walker = PT.NodesWalker(tree, lambda n: I.getType(n) == "FamilyName_t")
  assert [I.getValue(n) for n in walker()] == ['ROW1', 'BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # All predicates have the same options
  # ------------------------------------
  # Avoid Node : FamilyName FamilyName_t 'ROW1'
  predicates = [lambda n : I.getType(n) == "BC_t", lambda n : I.getType(n) == "FamilyName_t"]
  walker = PT.NodesWalkers(tree, predicates)
  print(f"nodes = {[I.getValue(n) for n in walker()]}")
  assert [I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in walker.cache] == []

  # Avoid Node : FamilyName FamilyName_t 'ROW1', with caching
  walker = PT.NodesWalkers(tree, predicates, caching=True)
  print(f"nodes = {[I.getValue(n) for n in walker()]}")
  assert [I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in walker.cache] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  fpathv = lambda nodes: '/'.join([I.getName(n) for n in nodes[:-1]]+[I.getValue(nodes[-1])])
  # ... with ancestors
  walker = PT.NodesWalkers(tree, predicates, ancestors = True)
  # for nodes in walker():
  #   print(f"nodes = {nodes}")
  print(f"nodes = {[[I.getName(n) for n in nodes] for nodes in walker()]}")
  assert [fpathv(nodes) for nodes in walker()] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in walker.cache] == []

  # ... with ancestors and caching
  walker = PT.NodesWalkers(tree, predicates, caching=True, ancestors=True)
  # for nodes in walker():
  #   print(f"nodes = {nodes}")
  print(f"nodes = {[[I.getName(n) for n in nodes] for nodes in walker()]}")
  assert [fpathv(nodes) for nodes in walker()] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in walker.cache] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # Each predicate has theirs owns options
  # --------------------------------------
  predicates = [{'predicate': lambda n : I.getType(n) == "BC_t", 'explore':'shallow'}, 
                {'predicate': lambda n : I.getType(n) == "FamilyName_t", 'depth':1}]

  # Avoid Node : FamilyName FamilyName_t 'ROW1'
  walker = PT.NodesWalkers(tree, predicates)
  print(f"nodes = {[I.getValue(n) for n in walker()]}")
  assert [I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in walker.cache] == []

  # Avoid Node : FamilyName FamilyName_t 'ROW1', with caching
  walker = PT.NodesWalkers(tree, predicates, caching=True)
  print(f"nodes = {[I.getValue(n) for n in walker()]}")
  assert [I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in walker.cache] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  fpathv = lambda nodes: '/'.join([I.getName(n) for n in nodes[:-1]]+[I.getValue(nodes[-1])])
  # ... with ancestors
  walker = PT.NodesWalkers(tree, predicates, ancestors = True)
  # for nodes in walker():
  #   print(f"nodes = {nodes}")
  print(f"nodes = {[[I.getName(n) for n in nodes] for nodes in walker()]}")
  assert [fpathv(nodes) for nodes in walker()] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in walker.cache] == []

  # ... with ancestors and caching
  walker = PT.NodesWalkers(tree, predicates, caching=True, ancestors=True)
  # for nodes in walker():
  #   print(f"nodes = {nodes}")
  print(f"nodes = {[[I.getName(n) for n in nodes] for nodes in walker()]}")
  assert [fpathv(nodes) for nodes in walker()] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in walker.cache] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
