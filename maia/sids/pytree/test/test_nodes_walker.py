import pytest
import fnmatch
import numpy as np

import Converter.Internal as I

from maia.sids import pytree as PT

from maia.utils        import parse_yaml_cgns

def test_NodesWalker():
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
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  base = PT.getNodeFromLabel(tree, 'CGNSBase_t') # get the first base

  walker = PT.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='bfs'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='dfs'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='bfs'; walker.depth=2
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='dfs'; walker.depth=2
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='bfs'; walker.explore='shallow'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='dfs'; walker.explore='shallow'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])

  walker = PT.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  walker.search='bfs'
  assert([I.getName(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  walker.predicate = lambda n: fnmatch.fnmatch(n[0], 'Zone*')
  walker.search='dfs'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  walker.search='bfs'; walker.depth=4
  assert([I.getName(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  walker.predicate = lambda n: fnmatch.fnmatch(n[0], 'Zone*')
  walker.search='dfs'; walker.depth=4
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  walker.search='bfs'; walker.depth=4
  assert([I.getName(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  walker.predicate = lambda n: fnmatch.fnmatch(n[0], 'Zone*')
  walker.search='dfs'; walker.depth=4
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])

  # Test parent
  walker = PT.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.root = base; walker.depth = 1
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])

  # Test predicate
  walker = PT.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  walker.explore='shallow'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  assert([I.getName(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # Test search
  walker = PT.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.explore='shallow'; walker.search='bfs'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])
  assert(walker.search == 'bfs')
  walker.search = 'dfs'; walker.explore = 'shallow'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])

  # Test explore
  walker = PT.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.explore='deep'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])
  assert(walker.explore == 'deep')
  walker.explore = 'shallow'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])

  # Test depth
  walker = PT.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.depth = 2
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])
  assert(walker.depth == 2)
  walker.depth = 1
  assert([I.getName(n) for n in walker()] == [])
  assert([I.getName(n) for n in walker.cache] == [])

def test_NodesWalker_sort():
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
  # I.printTree(tree)

  # Snake case
  # ==========
  walker = PT.NodesWalker(tree, lambda n: I.getType(n) == "BC_t", search='bfs')
  assert([I.getName(n) for n in walker()] == ['bca1', 'bcd2', 'bcb3', 'bce4', 'bcc5'])
  walker.sort = lambda children:reversed(children)
  assert([I.getName(n) for n in walker()] == ['bcc5', 'bce4', 'bcb3', 'bcd2', 'bca1'])
  walker.sort = PT.NodesWalker.BACKWARD
  assert([I.getName(n) for n in walker()] == ['bcc5', 'bce4', 'bcb3', 'bcd2', 'bca1'])

  fsort = lambda children : sorted(children, key=lambda n : I.getName(n)[2])
  walker.sort = fsort
  # for n in walker(search='bfs', sort=fsort):
  #   print(f"n = {I.getName(n)}")
  assert([I.getName(n) for n in walker()] == ['bca1', 'bcd2', 'bcb3', 'bcc5', 'bce4'])

  walker = PT.NodesWalker(tree, lambda n: I.getType(n) == "FamilyName_t")
  # walker.search = 'dfs'
  # for n in walker():
  #   print(f"n = {I.getValue(n)}")
  # walker.search = 'bfs'
  # for n in walker(search='bfs'):
  #   print(f"n = {I.getValue(n)}")
  walker.search = 'dfs'
  assert([I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'ROW1', 'BCD3', 'BCE4', 'BCB5'])
  walker.search = 'bfs'
  assert([I.getValue(n) for n in walker()] == ['ROW1', 'BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5'])

  walker.search = 'dfs'
  walker.sort = PT.NodesWalker.BACKWARD
  assert([I.getValue(n) for n in walker()] == ['BCB5', 'BCE4', 'BCD3', 'ROW1', 'BCA2', 'BCC1'])
  # walker.search = 'bfs'
  # walker.sort = I.NodesWalker.BACKWARD
  # for n in walker(search='bfs', sort=I.NodesWalker.BACKWARD):
  #   print(f"n = {I.getValue(n)}")
  walker.search = 'bfs'
  walker.sort = PT.NodesWalker.BACKWARD
  assert([I.getValue(n) for n in walker()] == ['ROW1', 'BCB5', 'BCE4', 'BCD3', 'BCA2', 'BCC1'])

def test_NodesWalker_apply():
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
  # I.printTree(tree)

  walker = PT.NodesWalker(tree, lambda n: I.getType(n) == "BC_t", search='bfs')
  walker.apply(lambda n : I.setName(n, I.getName(n).upper()))
  assert([I.getName(n) for n in walker()] == ['BCA1', 'BCD2', 'BCB3', 'BCE4', 'BCC5'])
  assert([I.getName(n) for n in walker.cache] == [])

  walker = PT.NodesWalker(tree, lambda n: I.getType(n) == "BC_t", search='dfs', caching=True)
  walker.apply(lambda n : I.setName(n, f"_{I.getName(n).upper()}"))
  assert([I.getName(n) for n in walker()] == ['_BCA1', '_BCD2', '_BCB3', '_BCE4', '_BCC5'])
  assert([I.getName(n) for n in walker.cache] == ['_BCA1', '_BCD2', '_BCB3', '_BCE4', '_BCC5'])