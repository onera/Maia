import pytest
import re
import fnmatch
import numpy as np
from Converter import Internal     as I
from maia.sids import Internal_ext as IE
from maia.sids.cgns_keywords import Label as CGL

from   maia.utils        import parse_yaml_cgns

def test_is_valid_label():
  assert IE.is_valid_label('') == True
  assert IE.is_valid_label('BC') == False
  assert IE.is_valid_label('BC_t') == True
  assert IE.is_valid_label('BC_toto') == False
  assert IE.is_valid_label('FakeLabel_t') == False

  assert IE.is_valid_label(CGL.BC_t.name) == True

def test_check_is_label():
  yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZBCA ZoneBC_t:
      bc1 BC_t:
        Index_i IndexArray_t:
      bc2 BC_t:
        Index_ii IndexArray_t:
    ZBCB ZoneBC_t:
      bc3 BC_t:
        Index_iii IndexArray_t:
      bc4 BC_t:
      bc5 BC_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""
  tree = parse_yaml_cgns.to_complete_pytree(yt)

  @IE.check_is_label('Zone_t')
  def apply_zone(node):
    pass

  for zone in I.getZones(tree):
    apply_zone(zone)

  with pytest.raises(IE.CGNSLabelNotEqualError):
    for zone in I.getBases(tree):
      apply_zone(zone)

def test_getValue():
  node = I.newDataArray('Test', [1,2])
  assert (I.getVal(node) == [1,2]).all()
  assert isinstance(I.getVal(node), np.ndarray)
  node = I.newDataArray('Test', np.array([1]))
  # Result from Cassiopee
  assert I.getValue(node) == 1
  assert isinstance(I.getValue(node), int)
  # Result expected
  assert I.getVal(node) == [1]
  assert isinstance(I.getVal(node), np.ndarray)

def test_getChildFromName():
  yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZBCA ZoneBC_t:
      bc1 BC_t:
        Index_i IndexArray_t:
      bc2 BC_t:
        Index_ii IndexArray_t:
    ZBCB ZoneBC_t:
      bc3 BC_t:
        Index_iii IndexArray_t:
      bc4 BC_t:
      bc5 BC_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  assert IE.getChildFromName(tree, "ZoneI") == I.getNodeFromName(tree, "ZoneI")
  assert I.getNodeFromName(tree, "ZoneB") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromName(tree, "ZoneB")

  base = I.getBases(tree)[0]
  assert IE.getChildFromName1(base, "ZoneI") == I.getNodeFromName1(base, "ZoneI")
  assert I.getNodeFromName1(base, "ZoneB") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromName1(base, "ZoneB")

  assert IE.getChildFromName2(tree, "ZoneI") == I.getNodeFromName2(tree, "ZoneI")
  assert I.getNodeFromName2(tree, "ZoneB") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromName2(tree, "ZoneB")

  assert IE.getChildFromName3(tree, "ZBCA") == I.getNodeFromName3(tree, "ZBCA")
  assert I.getNodeFromName3(tree, "ZZZZZ") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromName3(tree, "ZZZZZ")

def test_getChildFromLabel():
  yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZBCA ZoneBC_t:
      bc1 BC_t:
        Index_i IndexArray_t:
      bc2 BC_t:
        Index_ii IndexArray_t:
    ZBCB ZoneBC_t:
      bc3 BC_t:
        Index_iii IndexArray_t:
      bc4 BC_t:
      bc5 BC_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  assert IE.getChildFromLabel(tree, "Zone_t") == I.getNodeFromType(tree, "Zone_t")
  assert I.getNodeFromType(tree, "Family_t") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromLabel(tree, "Family_t")

  base = I.getBases(tree)[0]
  assert IE.getChildFromLabel1(base, "Zone_t") == I.getNodeFromType1(base, "Zone_t")
  assert I.getNodeFromType1(base, "Family_t") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromLabel1(base, "Family_t")

  assert IE.getChildFromLabel2(tree, "Zone_t") == I.getNodeFromType2(tree, "Zone_t")
  assert I.getNodeFromType2(tree, "Family_t") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromLabel2(tree, "Family_t")

  assert IE.getChildFromLabel3(tree, "ZoneBC_t") == I.getNodeFromType3(tree, "ZoneBC_t")
  assert I.getNodeFromType3(tree, "ZoneGridConnectivity_t") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromLabel3(tree, "ZoneGridConnectivity_t")

def test_getChildFromNameAndLabel():
  yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZBCA ZoneBC_t:
      bc1 BC_t:
        Index_i IndexArray_t:
      bc2 BC_t:
        Index_ii IndexArray_t:
    ZBCB ZoneBC_t:
      bc3 BC_t:
        Index_iii IndexArray_t:
      bc4 BC_t:
      bc5 BC_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  assert I.getNodeFromNameAndType(tree, "ZoneI", "Zone_t") == IE.requestChildFromNameAndLabel(tree, "ZoneI", "Zone_t")
  assert IE.getChildFromNameAndLabel(tree, "ZoneI", "Zone_t") == I.getNodeFromNameAndType(tree, "ZoneI", "Zone_t")
  assert I.getNodeFromNameAndType(tree, "ZoneB", "Zone_t")   == None
  assert I.getNodeFromNameAndType(tree, "ZoneI", "Family_t") == None
  assert IE.requestChildFromNameAndLabel(tree, "ZoneB", "Zone_t")   == None
  assert IE.requestChildFromNameAndLabel(tree, "ZoneI", "Family_t") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromNameAndLabel(tree, "ZoneB", "Zone_t")
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromNameAndLabel(tree, "ZoneI", "Family_t")

  base = I.getBases(tree)[0]
  assert I.getNodeFromNameAndType(base, "ZoneI", "Zone_t") == IE.requestChildFromNameAndLabel1(base, "ZoneI", "Zone_t")
  assert IE.getChildFromNameAndLabel1(base, "ZoneI", "Zone_t") == IE.requestChildFromNameAndLabel1(base, "ZoneI", "Zone_t")
  assert I.getNodeFromNameAndType(base, "ZoneB", "Zone_t")   == None
  assert I.getNodeFromNameAndType(base, "ZoneI", "Family_t") == None
  assert IE.requestChildFromNameAndLabel1(base, "ZoneB", "Zone_t")   == None
  assert IE.requestChildFromNameAndLabel1(base, "ZoneI", "Family_t") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromNameAndLabel1(base, "ZoneB", "Zone_t")
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromNameAndLabel1(base, "ZoneI", "Family_t")

  assert I.getNodeFromNameAndType(tree, "ZoneI", "Zone_t") == IE.requestChildFromNameAndLabel2(tree, "ZoneI", "Zone_t")
  assert IE.getChildFromNameAndLabel2(tree, "ZoneI", "Zone_t") == IE.requestChildFromNameAndLabel2(tree, "ZoneI", "Zone_t")
  assert I.getNodeFromNameAndType(tree, "ZoneB", "Zone_t")   == None
  assert I.getNodeFromNameAndType(tree, "ZoneI", "Family_t") == None
  assert IE.requestChildFromNameAndLabel2(tree, "ZoneB", "Zone_t")   == None
  assert IE.requestChildFromNameAndLabel2(tree, "ZoneI", "Family_t") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromNameAndLabel2(tree, "ZoneB", "Zone_t")
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromNameAndLabel2(tree, "ZoneI", "Family_t")

  assert I.getNodeFromNameAndType(tree, "ZBCA", "ZoneBC_t") == IE.requestChildFromNameAndLabel3(tree, "ZBCA", "ZoneBC_t")
  assert IE.getChildFromNameAndLabel3(tree, "ZBCA", "ZoneBC_t") == IE.requestChildFromNameAndLabel3(tree, "ZBCA", "ZoneBC_t")
  assert I.getNodeFromNameAndType(tree, "ZZZZZ", "ZoneBC_t")              == None
  assert I.getNodeFromNameAndType(tree, "ZBCA", "ZoneGridConnectivity_t") == None
  assert IE.requestChildFromNameAndLabel3(tree, "ZZZZZ", "ZoneBC_t")              == None
  assert IE.requestChildFromNameAndLabel3(tree, "ZBCA", "ZoneGridConnectivity_t") == None
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromNameAndLabel3(tree, "ZZZZZ", "ZoneBC_t")
  with pytest.raises(IE.CGNSNodeFromPredicateNotFoundError):
    IE.getChildFromNameAndLabel3(tree, "ZBCA", "ZoneGridConnectivity_t")

def test_getChildFromPredicate():
  yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZBCA ZoneBC_t:
      bc1 BC_t:
        Index_i IndexArray_t:
      bc2 BC_t:
        Index_ii IndexArray_t:
    ZBCB ZoneBC_t:
      bc3 BC_t:
        Index_iii IndexArray_t:
      bc4 BC_t:
      bc5 BC_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""
  is_base    = lambda n: I.getName(n) == 'Base' and I.getType(n) == 'CGNSBase_t'
  is_zonei   = lambda n: I.getName(n) == 'ZoneI' and I.getType(n) == 'Zone_t'
  is_nface   = lambda n: I.getName(n) == 'NFace' and I.getType(n) == 'Elements_t'
  is_ngon    = lambda n: I.getName(n) == 'Ngon' and I.getType(n) == 'Elements_t'
  is_zbca    = lambda n: I.getName(n) == 'ZBCA' and I.getType(n) == 'ZoneBC_t'
  is_bc1     = lambda n: I.getName(n) == 'bc1' and I.getType(n) == 'BC_t'
  is_index_i = lambda n: I.getName(n) == 'Index_i' and I.getType(n) == 'IndexArray_t'

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  base = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", method="bfs")
  assert is_zonei(zone)
  zone = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", method="dfs")
  assert is_zonei(zone)

  zone = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", method="dfs", depth=1)
  assert zone is None
  zone = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", method="dfs", depth=2)
  assert is_zonei(zone)
  zone = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", method="dfs", depth=3)
  assert is_zonei(zone)

  node = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "Base", method="dfs", depth=1)
  assert is_base(node)
  node = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", method="dfs", depth=2)
  assert is_zonei(node)
  node = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "Ngon", method="dfs", depth=3)
  assert is_ngon(node)
  node = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "bc1", method="dfs", depth=4)
  assert is_bc1(node)
  node = IE.requestChildFromPredicate(tree, lambda n: I.getName(n) == "Index_i", method="dfs", depth=5)
  assert is_index_i(node)

  element = IE.requestChildFromName(tree, "NFace")
  assert is_nface(element)
  element = IE.requestChildFromValue(tree, np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.requestChildFromLabel(tree, "Elements_t")
  assert is_ngon(element)
  element = IE.requestChildFromNameAndValue(tree, "NFace", np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.requestChildFromNameAndLabel(tree, "NFace", "Elements_t")
  assert is_nface(element)
  element = IE.requestChildFromValueAndLabel(tree, np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)
  element = IE.requestChildFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)
  #
  element = IE.requestChildFromName(tree, "TOTO")
  assert element is None
  element = IE.requestChildFromValue(tree, np.array([1230,0], order='F'))
  assert element is None
  element = IE.requestChildFromLabel(tree, "ZoneSubRegion_t")
  assert element is None
  element = IE.requestChildFromNameAndValue(tree, "TOTO", np.array([23,0], order='F'))
  assert element is None
  element = IE.requestChildFromNameAndValue(tree, "NFace", np.array([1230,0], order='F'))
  assert element is None
  element = IE.requestChildFromNameAndLabel(tree, "TOTO", "Elements_t")
  assert element is None
  element = IE.requestChildFromNameAndLabel(tree, "NFace", "ZoneSubRegion_t")
  assert element is None
  element = IE.requestChildFromValueAndLabel(tree, np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = IE.requestChildFromValueAndLabel(tree, np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None
  element = IE.requestChildFromNameValueAndLabel(tree, "TOTO", np.array([23,0], order='F'), "Elements_t")
  assert element is None
  element = IE.requestChildFromNameValueAndLabel(tree, "NFace", np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = IE.requestChildFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None

  # requestChildFrom... with level and dfs
  zone = IE.requestChildFromPredicate1(tree, lambda n: I.getName(n) == "ZoneI", method="dfs")
  assert zone is None
  zone = IE.requestChildFromPredicate2(tree, lambda n: I.getName(n) == "ZoneI", method="dfs")
  assert is_zonei(zone)
  zone = IE.requestChildFromPredicate3(tree, lambda n: I.getName(n) == "ZoneI", method="dfs")
  assert is_zonei(zone)

  zone = IE.requestChildFromName(tree, "ZoneI", method="dfs", depth=1)
  assert zone is None
  zone = IE.requestChildFromName(tree, "ZoneI", method="dfs", depth=2)
  assert is_zonei(zone)
  zone = IE.requestChildFromName(tree, "ZoneI", method="dfs", depth=3)
  assert is_zonei(zone)

  zone = IE.requestChildFromName1(tree, "ZoneI")
  assert zone is None
  zone = IE.requestChildFromName2(tree, "ZoneI")
  assert is_zonei(zone)
  zone = IE.requestChildFromName3(tree, "ZoneI")
  assert is_zonei(zone)

  node = IE.getChildFromName1(tree, "Base")
  assert is_base(node)
  node = IE.getChildFromName2(tree, "ZoneI")
  assert is_zonei(node)
  node = IE.getChildFromName3(tree, "Ngon")
  assert is_ngon(node)
  node = IE.getChildFromName4(tree, "bc1")
  assert is_bc1(node)
  node = IE.getChildFromName5(tree, "Index_i")
  assert is_index_i(node)

  node = IE.getChildFromLabel1(tree, "CGNSBase_t")
  assert is_base(node)
  node = IE.getChildFromLabel2(tree, "Zone_t")
  assert is_zonei(node)
  node = IE.getChildFromLabel3(tree, "Elements_t")
  assert is_ngon(node)
  node = IE.getChildFromLabel4(tree, "BC_t")
  assert is_bc1(node)
  node = IE.getChildFromLabel5(tree, "IndexArray_t")
  assert is_index_i(node)

  # getChildFrom...
  base = IE.getChildFromPredicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = IE.getChildFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", method="dfs")
  assert is_zonei(zone)

  element = IE.getChildFromName(tree, "NFace")
  assert is_nface(element)
  element = IE.getChildFromValue(tree, np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.getChildFromLabel(tree, "Elements_t")
  assert is_ngon(element)
  element = IE.getChildFromNameAndValue(tree, "NFace", np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.getChildFromNameAndLabel(tree, "NFace", "Elements_t")
  assert is_nface(element)
  element = IE.getChildFromValueAndLabel(tree, np.array([23,0], dtype='int64',order='F'), "Elements_t")
  assert is_nface(element)
  element = IE.getChildFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)

def test_getChildrenFromPredicate():
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
  tree = parse_yaml_cgns.to_complete_pytree(yt)

  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), method='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), method='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), method='dfs', depth=1)] == [])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), method='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromName(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromName1(tree, 'Zone*')] == [])
  assert([I.getName(n) for n in IE.getChildrenFromName2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromName3(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromName4(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromName5(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])

  ngon = np.array([22,0], order='F')
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: np.array_equal(n[1], ngon))] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), method='bfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), method='dfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), method='dfs', depth=1)] == [])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), method='dfs', depth=2)] == [])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), method='dfs', depth=3)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getChildrenFromValue(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getChildrenFromValue1(tree, ngon)] == [])
  assert([I.getName(n) for n in IE.getChildrenFromValue2(tree, ngon)] == [])
  assert([I.getName(n) for n in IE.getChildrenFromValue3(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getChildrenFromValue4(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getChildrenFromValue5(tree, ngon)] == ['NgonI', 'NgonJ'])

  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: n[3] == 'Zone_t', method='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: n[3] == 'Zone_t', method='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: n[3] == 'Zone_t', method='dfs', depth=1)] == [])
  assert([I.getName(n) for n in IE.getChildrenFromPredicate(tree, lambda n: n[3] == 'Zone_t', method='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromLabel(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromLabel1(tree, 'Zone_t')] == [])
  assert([I.getName(n) for n in IE.getChildrenFromLabel2(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromLabel3(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromLabel4(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getChildrenFromLabel5(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])

def test_getChildrenFromPredicate1():
  yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZBCA ZoneBC_t:
      bc1 BC_t:
        Index_i IndexArray_t:
      bc2 BC_t:
        Index_ii IndexArray_t:
    ZBCB ZoneBC_t:
      bc3 BC_t:
        Index_iii IndexArray_t:
      bc4 BC_t:
      bc5 BC_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  # Test from Name
  base = I.getBases(tree)[0]
  nodes_from_name1 = I.getNodesFromName1(base, "ZoneI")
  assert IE.getChildrenFromPredicate1(base, lambda n: I.getName(n) == "ZoneI") == nodes_from_name1
  # Wildcard is not allowed in getNodesFromName1() from Cassiopee
  assert I.getNodesFromName1(base, "Zone*") == []
  assert IE.getChildrenFromPredicate1(base, lambda n: fnmatch.fnmatch(I.getName(n), "Zone*")) == nodes_from_name1
  assert IE.getChildrenFromName1(base, "Zone*") == nodes_from_name1
  #Exclude top level which is included by Cassiopée
  assert IE.getChildrenFromName1(base[0], "Base") == []

  # Test from Type
  nodes_from_type1 = I.getNodesFromType1(base, CGL.Zone_t.name)
  assert IE.getChildrenFromPredicate1(base, lambda n: I.getType(n) == CGL.Zone_t.name) == nodes_from_type1
  assert IE.getChildrenFromLabel1(base, CGL.Zone_t.name) == nodes_from_type1

  # Test from Value
  zone = I.getZones(tree)[0]
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [n for n in I.getNodesFromType1(zone, CGL.Elements_t.name) if np.array_equal(I.getVal(n), ngon_value)]
  assert IE.getChildrenFromPredicate1(zone, lambda n: I.getType(n) == CGL.Elements_t.name and np.array_equal(I.getVal(n), ngon_value)) == elements_from_type_value1
  assert IE.getChildrenFromValue1(zone, ngon_value) == elements_from_type_value1

  zonebcs_from_type_name1 = [n for n in I.getNodesFromType1(zone, CGL.ZoneBC_t.name) if I.getName(n) != "ZBCA"]
  assert IE.getChildrenFromPredicate1(zone, lambda n: I.getType(n) == CGL.ZoneBC_t.name and I.getName(n) != "ZBCA") == zonebcs_from_type_name1

def test_getNodesDispatch1():
  fs = I.newFlowSolution()
  data_a   = I.newDataArray('DataA', [1,2,3], parent=fs)
  data_b   = I.newDataArray('DataB', [4,6,8], parent=fs)
  grid_loc = I.newGridLocation('Vertex', fs)
  assert IE.getNodesDispatch1(fs, 'DataB') == [data_b]
  assert IE.getNodesDispatch1(fs, 'DataArray_t') == [data_a, data_b]
  assert IE.getNodesDispatch1(fs, CGL.GridLocation_t) == [grid_loc]
  assert IE.getNodesDispatch1(fs, np.array([4,6,8])) == [data_b]
  assert IE.getNodesDispatch1(fs, lambda n: isinstance(I.getValue(n), str) and I.getValue(n) == 'Vertex') == [grid_loc]
  with pytest.raises(TypeError):
    IE.getNodesDispatch1(fs, False)

def test_getNodesByMatching():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index1_i IndexArray_t:
    bc2 BC_t:
      Index2_i IndexArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index3_i IndexArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index4_i IndexArray_t:
      Index4_ii IndexArray_t:
      Index4_iii IndexArray_t:
"""
  zoneI = parse_yaml_cgns.to_node(yt)
  zbcB  = I.getNodeFromName(zoneI, 'ZBCB' )

  assert list(IE.getNodesByMatching(zoneI, '')) == []
  assert list(IE.getNodesByMatching(zoneI, 'BC_t')) == []
  assert list(IE.getNodesByMatching(zoneI, 'Index4_ii')) == []

  onelvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(node) for node in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = IE.getNodesByMatching(zbcB, 'bc5')
  assert [I.getName(node) for node in onelvl] == ['bc5']

  twolvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [I.getName(node) for node in twolvl] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  twolvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t/bc5')
  assert [I.getName(node) for node in twolvl] == ['bc5']
  twolvl = IE.getNodesByMatching(zbcB, 'BC_t/IndexArray_t')
  assert [I.getName(node) for node in twolvl] == ['Index3_i','Index4_i','Index4_ii','Index4_iii']

  results3 = ['Index1_i','Index2_i','Index3_i','Index4_i','Index4_ii','Index4_iii']
  threelvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  assert [I.getName(node) for node in threelvl] == results3
  assert len(list(IE.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t/Index*_i'))) == 4

  threelvl1 = IE.getNodesByMatching(zoneI, "ZoneBC_t/BC_t/lambda n: I.getType(n) == 'IndexArray_t'")
  # print(f"threelvl1 = {[I.getName(node) for node in threelvl1]}")
  assert [I.getName(node) for node in threelvl1] == results3

  threelvl2 = IE.getNodesByMatching(zoneI, ["ZoneBC_t", CGL.BC_t.name, lambda n: I.getType(n) == 'IndexArray_t'])
  # print(f"threelvl2 = {[I.getName(node) for node in threelvl2]}")
  assert [I.getName(node) for node in threelvl2] == results3

  threelvl3 = IE.getNodesByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: I.getType(n) == 'IndexArray_t'])
  # print(f"threelvl3 = {[I.getName(node) for node in threelvl3]}")
  assert [I.getName(node) for node in threelvl3] == results3

  with pytest.raises(TypeError):
    list(IE.getNodesByMatching(zoneI, 12))

def test_getNodesWithParentsByMatching():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index_i IndexArray_t:
      PL1 DataArray_t:
    bc2 BC_t:
      Index_ii IndexArray_t:
      PL2 DataArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index_iii IndexArray_t:
      PL3 DataArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index_iv IndexArray_t:
      Index_v IndexArray_t:
      Index_vi IndexArray_t:
      PL4 DataArray_t:
"""
  zoneI = parse_yaml_cgns.to_node(yt)
  zbcB  = I.getNodeFromName(zoneI, 'ZBCB' )

  assert list(IE.getNodesWithParentsByMatching(zoneI, '')) == []
  assert list(IE.getNodesWithParentsByMatching(zoneI, 'BC_t')) == []

  onelvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = IE.getNodesWithParentsByMatching(zbcB, 'BC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCA', 'bc1'), ('ZBCA', 'bc2'), ('ZBCB', 'bc3'), ('ZBCB', 'bc4'), ('ZBCB', 'bc5')]
  twolvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/bc3')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCB', 'bc3')]
  twolvl = IE.getNodesWithParentsByMatching(zbcB, 'BC_t/IndexArray_t')
  for bc in I.getNodesFromType1(zbcB, 'BC_t'):
    for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
      assert next(twolvl) == (bc, idx)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
        assert next(threelvl) == (zbc, bc, idx)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/PL*')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getChildrenFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, ['ZoneBC_t', 'BC_t', 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getChildrenFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, 'BC_t', 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getChildrenFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getChildrenFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  p = re.compile('PL[12]')
  threelvl = IE.getNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: p.match(I.getName(n))])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in [n for n in IE.getChildrenFromName1(bc, 'PL*') if p.match(I.getName(n))]:
        assert next(threelvl) == (zbc, bc, pl)

  with pytest.raises(TypeError):
    list(IE.getNodesWithParentsByMatching(zoneI, 12))

def test_getSubregionExtent():
  yt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
    BC2 BC_t:
  ZGC1 ZoneGridConnectivity_t:
    GCA GridConnectivity_t:
    GCB GridConnectivity_t:
  ZGC2 ZoneGridConnectivity_t:
    GC1to1A GridConnectivity1to1_t:
    GC1to1B GridConnectivity1to1_t:
  UnLinkedZSR ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[]]:
  BCLinkedZSR ZoneSubRegion_t:
    BCRegionName Descriptor_t "BC2":
  GCLinkedZSR ZoneSubRegion_t:
    GridConnectivityRegionName Descriptor_t "GC1to1B":
  OrphelanZSR ZoneSubRegion_t:
    BCRegionName Descriptor_t "BC9":
  WrongZSR WrongType_t:
    BCRegionName Descriptor_t "BC":
  """
  zone = parse_yaml_cgns.to_node(yt)

  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'UnLinkedZSR'), zone) == 'UnLinkedZSR'
  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'BCLinkedZSR'), zone) == 'ZBC/BC2'
  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'GCLinkedZSR'), zone) == 'ZGC2/GC1to1B'

  with pytest.raises(ValueError):
    IE.getSubregionExtent(I.getNodeFromName(zone, 'OrphelanZSR'), zone)
  with pytest.raises(AssertionError):
    IE.getSubregionExtent(I.getNodeFromName(zone, 'WrongZSR'), zone)

def test_newDistribution():
  distri = IE.newDistribution()
  assert I.getName(distri) == ':CGNS#Distribution'
  assert I.getType(distri) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  distri = IE.newDistribution(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#Distribution') is not None

  zone = I.newZone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = I.newZone('zone')
  distri_arrayA = {'Cell' : [0,15,30]}
  distri_arrayB = {'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrayA, parent=zone)
  distri = IE.newDistribution(distri_arrayB, parent=zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(I.getNodesFromName(zone, ':CGNS#Distribution')) == 1

def test_newGlobalNumbering():
  gnum = IE.newGlobalNumbering()
  assert I.getName(gnum) == ':CGNS#GlobalNumbering'
  assert I.getType(gnum) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  gnum = IE.newGlobalNumbering(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#GlobalNumbering') is not None

  zone = I.newZone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = IE.newGlobalNumbering(gnum_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None

def test_getDistribution():
  zone = I.newZone()
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrays, zone)
  assert IE.getDistribution(zone) is distri
  assert (I.getVal(IE.getDistribution(zone, 'Cell')) == [0,15,30]).all()
  assert (I.getVal(IE.getDistribution(zone, 'Vertex')) == [100,1000,1000]).all()

def test_getGlobalNumbering():
  zone = I.newZone()
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = IE.newGlobalNumbering(gnum_arrays, zone)
  assert IE.getGlobalNumbering(zone) is gnum_node
  assert (I.getVal(IE.getGlobalNumbering(zone, 'Cell')) == [4,21,1,2,8,12]).all()
  assert  I.getVal(IE.getGlobalNumbering(zone, 'Vertex')) == None

if __name__ == "__main__":
  # test_getChildFromPredicate()
  # test_requireNodeFromName()
  # test_requireNodeFromType()
  # test_getRequireNodeFromNameAndType()
  test_getChildrenFromPredicate()
