import pytest
import fnmatch
import numpy as np

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

yt_two = """
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
    ZBCBI ZoneBC_t:
      bc3I BC_t:
        Index_iii IndexArray_t:
        PL3I DataArray_t:
      bc4 BC_t:
      bc5 BC_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
        PL4 DataArray_t:
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

def test_getNodesFromPredicate():
  tree = parse_yaml_cgns.to_cgns_tree(yt_two)


  # Camel case
  # ----------
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])

  ngon = np.array([22,0], order='F')
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon))] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='bfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=2)] == [])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=3)] == ['NgonI', 'NgonJ'])

  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in PT.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])

  # Snake case
  # ----------
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])

  ngon = np.array([22,0], order='F')
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon))] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='bfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=2)] == [])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=3)] == ['NgonI', 'NgonJ'])

  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in PT.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])


def test_iterNodesFromPredicate():
  tree = parse_yaml_cgns.to_cgns_tree(yt_two)


  # Camel case
  # ----------
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])

  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'ZoneBC_t')] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # alias for shallow exploring
  assert([I.getName(n) for n in PT.siterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.siterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.siterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in PT.siterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in PT.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])


  # Snake case
  # ----------
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])

  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'ZoneBC_t')] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # alias for shallow exploring
  assert([I.getName(n) for n in PT.siter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.siter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.siter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in PT.siter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in PT.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])



def test_getNodesFromPredicates():

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  base = PT.getNodesFromLabelCGNSBase(tree)[0]

  results = PT.getNodesFromPredicates(tree, [lambda n: I.getType(n) == "FamilyName_t"])
  assert [I.getValue(n) for n in results] == ['ROW1', 'BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # All predicates have the same options
  # ------------------------------------
  # Avoid Node : FamilyName FamilyName_t 'ROW1', traversal=deep
  results = PT.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"])
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  fpathv = lambda nodes: '/'.join([I.getName(n) for n in nodes[:-1]]+[I.getValue(nodes[-1])])
  results = PT.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # Avoid Node : FamilyName FamilyName_t 'ROW1', traversal=shallow
  results = PT.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow')
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  results = PT.sgetNodesFromPredicates(tree, ["BC_t", "FamilyName_t"])
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  results = PT.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow', ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  results = PT.sgetNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # With search='dfs', depth=1 -> iterNodesByMatching
  results = PT.getNodesFromPredicates(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', depth=1)
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  results = PT.getNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs')
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  results = PT.getNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"])
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  results = PT.getNodesFromPredicates(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', depth=1, ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]
  results = PT.getNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]
  results = PT.getNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]

  # Each predicate has theirs owns options
  # --------------------------------------
  predicates = [{'predicate':"BC_t", 'explore':'shallow'}, {'predicate':"FamilyName_t", 'depth':1}]

  # Avoid Node : FamilyName FamilyName_t 'ROW1'
  results = PT.getNodesFromPredicates(tree, predicates)
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  results = PT.getNodesFromPredicates(tree, predicates, ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

def test_iterNodesFromPredicates():
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  base = PT.getNodeFromLabel(tree, 'CGNSBase_t') # get the first base

  assert [I.getValue(n) for n in PT.iterNodesFromPredicates(tree, [lambda n: I.getType(n) == "FamilyName_t"])] == ['ROW1', 'BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # All predicates have the same options
  # ------------------------------------
  # Avoid Node : FamilyName FamilyName_t 'ROW1', traversal=deep
  assert [I.getValue(n) for n in PT.iterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"])] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  fpathv = lambda nodes: '/'.join([I.getName(n) for n in nodes[:-1]]+[I.getValue(nodes[-1])])
  assert [fpathv(nodes) for nodes in PT.iterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], ancestors=True)] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # Avoid Node : FamilyName FamilyName_t 'ROW1', traversal=shallow
  assert [I.getValue(n) for n in PT.iterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow')] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in PT.siterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"])] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  assert [fpathv(nodes) for nodes in PT.iterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow', ancestors=True)] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in PT.siterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], ancestors=True)] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # With search='dfs', depth=1 -> iterNodesByMatching
  assert [I.getValue(n) for n in PT.iterNodesFromPredicates(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', depth=1)] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in PT.iterNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs')] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in PT.iterNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"])] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  assert [fpathv(nodes) for nodes in PT.iterNodesFromPredicates(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', depth=1, ancestors=True)] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in PT.iterNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', ancestors=True)] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in PT.iterNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], ancestors=True)] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]

  # Each predicate has theirs owns options
  # --------------------------------------
  predicates = [{'predicate':"BC_t", 'explore':'shallow'}, {'predicate':"FamilyName_t", 'depth':1}]

  # Avoid Node : FamilyName FamilyName_t 'ROW1'
  assert [I.getValue(n) for n in PT.iterNodesFromPredicates(tree, predicates)] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  assert [fpathv(nodes) for nodes in PT.iterNodesFromPredicates(tree, predicates, ancestors=True)] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
