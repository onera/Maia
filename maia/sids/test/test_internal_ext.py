import pytest
import re
import fnmatch
import numpy as np
from Converter import Internal as CI
from maia.sids import Internal_ext as IE
from maia.sids import internal as I
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

  @I.check_is_label('Zone_t')
  def apply_zone(node):
    pass

  for zone in I.getZones(tree):
    apply_zone(zone)

  with pytest.raises(I.CGNSLabelNotEqualError):
    for zone in I.getBases(tree):
      apply_zone(zone)

def test_getValue():
  # Camel case
  # ==========
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

  # Snake case
  # ==========
  node = I.new_data_array('Test', [1,2])
  assert (I.get_val(node) == [1,2]).all()
  assert isinstance(I.get_val(node), np.ndarray)
  node = I.new_data_array('Test', np.array([1]))
  # Result from Cassiopee
  assert I.get_value(node) == 1
  assert isinstance(I.get_value(node), int)
  # Result expected
  assert I.get_val(node) == [1]
  assert isinstance(I.get_val(node), np.ndarray)

def test_getNodeFromName():
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

  # Camel case
  # ==========
  assert I.getNodeFromName(tree, "ZoneI") == CI.getNodeFromName(tree, "ZoneI")
  assert CI.getNodeFromName(tree, "ZoneB") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromName(tree, "ZoneB")

  base = I.getBases(tree)[0]
  assert I.getNodeFromName1(base, "ZoneI") == CI.getNodeFromName1(base, "ZoneI")
  assert CI.getNodeFromName1(base, "ZoneB") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromName1(base, "ZoneB")

  assert I.getNodeFromName2(tree, "ZoneI") == CI.getNodeFromName2(tree, "ZoneI")
  assert CI.getNodeFromName2(tree, "ZoneB") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromName2(tree, "ZoneB")

  assert I.getNodeFromName3(tree, "ZBCA") == CI.getNodeFromName3(tree, "ZBCA")
  assert CI.getNodeFromName3(tree, "ZZZZZ") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromName3(tree, "ZZZZZ")

  # Snake case
  # ==========
  node = I.get_node_from_name(tree, "ZoneI")
  assert I.get_name(node) == "ZoneI"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name(tree, "ZoneB")

  base = I.get_all_base(tree)[0]
  node = I.get_node_from_name1(base, "ZoneI")
  assert I.get_name(node) == "ZoneI"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name1(base, "ZoneB")

  node = I.get_node_from_name2(tree, "ZoneI")
  assert I.get_name(node) == "ZoneI"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name2(tree, "ZoneB")

  node = I.get_node_from_name3(tree, "ZBCA")
  assert I.get_name(node) == "ZBCA"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name3(tree, "ZZZZZ")

def test_getNodeFromLabel():
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

  # Camel case
  # ==========
  assert I.getNodeFromLabel(tree, "Zone_t") == CI.getNodeFromType(tree, "Zone_t")
  assert CI.getNodeFromType(tree, "Family_t") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromLabel(tree, "Family_t")

  base = I.getBases(tree)[0]
  assert I.getNodeFromLabel1(base, "Zone_t") == CI.getNodeFromType1(base, "Zone_t")
  assert CI.getNodeFromType1(base, "Family_t") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromLabel1(base, "Family_t")

  assert I.getNodeFromLabel2(tree, "Zone_t") == CI.getNodeFromType2(tree, "Zone_t")
  assert CI.getNodeFromType2(tree, "Family_t") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromLabel2(tree, "Family_t")

  assert I.getNodeFromLabel3(tree, "ZoneBC_t") == CI.getNodeFromType3(tree, "ZoneBC_t")
  assert CI.getNodeFromType3(tree, "ZoneGridConnectivity_t") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromLabel3(tree, "ZoneGridConnectivity_t")

  # Snake case
  # ==========
  node = I.get_node_from_label(tree, "Zone_t")
  assert I.get_label(node) == "Zone_t"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_label(tree, "Family_t")

  base = I.get_all_base(tree)[0]
  node = I.get_node_from_label1(base, "Zone_t")
  assert I.get_label(node) == "Zone_t"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_label1(base, "Family_t")

  node = I.get_node_from_label2(tree, "Zone_t")
  assert I.get_label(node) == "Zone_t"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_label2(tree, "Family_t")

  node = I.get_node_from_label3(tree, "ZoneBC_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_label3(tree, "ZoneGridConnectivity_t")

def test_getNodeFromNameAndLabel():
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

  # Camel case
  # ==========
  assert CI.getNodeFromNameAndType(tree, "ZoneI", "Zone_t") == I.requestNodeFromNameAndLabel(tree, "ZoneI", "Zone_t")
  assert I.getNodeFromNameAndLabel(tree, "ZoneI", "Zone_t") == CI.getNodeFromNameAndType(tree, "ZoneI", "Zone_t")
  assert CI.getNodeFromNameAndType(tree, "ZoneB", "Zone_t")   == None
  assert CI.getNodeFromNameAndType(tree, "ZoneI", "Family_t") == None
  assert I.requestNodeFromNameAndLabel(tree, "ZoneB", "Zone_t")   == None
  assert I.requestNodeFromNameAndLabel(tree, "ZoneI", "Family_t") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromNameAndLabel(tree, "ZoneB", "Zone_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromNameAndLabel(tree, "ZoneI", "Family_t")

  base = I.getBases(tree)[0]
  assert CI.getNodeFromNameAndType(base, "ZoneI", "Zone_t") == I.requestNodeFromNameAndLabel1(base, "ZoneI", "Zone_t")
  assert I.getNodeFromNameAndLabel1(base, "ZoneI", "Zone_t") == I.requestNodeFromNameAndLabel1(base, "ZoneI", "Zone_t")
  assert CI.getNodeFromNameAndType(base, "ZoneB", "Zone_t")   == None
  assert CI.getNodeFromNameAndType(base, "ZoneI", "Family_t") == None
  assert I.requestNodeFromNameAndLabel1(base, "ZoneB", "Zone_t")   == None
  assert I.requestNodeFromNameAndLabel1(base, "ZoneI", "Family_t") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromNameAndLabel1(base, "ZoneB", "Zone_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromNameAndLabel1(base, "ZoneI", "Family_t")

  assert CI.getNodeFromNameAndType(tree, "ZoneI", "Zone_t") == I.requestNodeFromNameAndLabel2(tree, "ZoneI", "Zone_t")
  assert I.getNodeFromNameAndLabel2(tree, "ZoneI", "Zone_t") == I.requestNodeFromNameAndLabel2(tree, "ZoneI", "Zone_t")
  assert CI.getNodeFromNameAndType(tree, "ZoneB", "Zone_t")   == None
  assert CI.getNodeFromNameAndType(tree, "ZoneI", "Family_t") == None
  assert I.requestNodeFromNameAndLabel2(tree, "ZoneB", "Zone_t")   == None
  assert I.requestNodeFromNameAndLabel2(tree, "ZoneI", "Family_t") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromNameAndLabel2(tree, "ZoneB", "Zone_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromNameAndLabel2(tree, "ZoneI", "Family_t")

  assert CI.getNodeFromNameAndType(tree, "ZBCA", "ZoneBC_t") == I.requestNodeFromNameAndLabel3(tree, "ZBCA", "ZoneBC_t")
  assert I.getNodeFromNameAndLabel3(tree, "ZBCA", "ZoneBC_t") == I.requestNodeFromNameAndLabel3(tree, "ZBCA", "ZoneBC_t")
  assert CI.getNodeFromNameAndType(tree, "ZZZZZ", "ZoneBC_t")              == None
  assert CI.getNodeFromNameAndType(tree, "ZBCA", "ZoneGridConnectivity_t") == None
  assert I.requestNodeFromNameAndLabel3(tree, "ZZZZZ", "ZoneBC_t")              == None
  assert I.requestNodeFromNameAndLabel3(tree, "ZBCA", "ZoneGridConnectivity_t") == None
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromNameAndLabel3(tree, "ZZZZZ", "ZoneBC_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.getNodeFromNameAndLabel3(tree, "ZBCA", "ZoneGridConnectivity_t")

  # Snake case
  # ==========
  node = I.request_node_from_name_and_label(tree, "ZoneI", "Zone_t")
  assert I.get_name(node) == "ZoneI" and I.get_label(node) == "Zone_t"
  assert I.request_node_from_name_and_label(tree, "ZoneB", "Zone_t")   == None
  assert I.request_node_from_name_and_label(tree, "ZoneI", "Family_t") == None
  node = I.get_node_from_name_and_label(tree, "ZoneI", "Zone_t")
  assert I.get_name(node) == "ZoneI" and I.get_label(node) == "Zone_t"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name_and_label(tree, "ZoneB", "Zone_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name_and_label(tree, "ZoneI", "Family_t")

  base = I.get_base(tree) # get the first base
  node = I.request_node_from_name_and_label1(base, "ZoneI", "Zone_t")
  assert I.get_name(node) == "ZoneI" and I.get_label(node) == "Zone_t"
  assert I.request_node_from_name_and_label1(base, "ZoneB", "Zone_t")   == None
  assert I.request_node_from_name_and_label1(base, "ZoneI", "Family_t") == None
  node = I.get_node_from_name_and_label1(base, "ZoneI", "Zone_t")
  assert I.get_name(node) == "ZoneI" and I.get_label(node) == "Zone_t"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name_and_label1(base, "ZoneB", "Zone_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name_and_label1(base, "ZoneI", "Family_t")

  node = I.request_node_from_name_and_label2(tree, "ZoneI", "Zone_t")
  assert I.get_name(node) == "ZoneI" and I.get_label(node) == "Zone_t"
  assert I.request_node_from_name_and_label2(tree, "ZoneB", "Zone_t")   == None
  assert I.request_node_from_name_and_label2(tree, "ZoneI", "Family_t") == None
  node = I.get_node_from_name_and_label2(tree, "ZoneI", "Zone_t")
  assert I.get_name(node) == "ZoneI" and I.get_label(node) == "Zone_t"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name_and_label2(tree, "ZoneB", "Zone_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name_and_label2(tree, "ZoneI", "Family_t")

  node = I.request_node_from_name_and_label3(tree, "ZBCA", "ZoneBC_t")
  assert I.get_name(node) == "ZBCA" and I.get_label(node) == "ZoneBC_t"
  assert I.request_node_from_name_and_label3(tree, "ZZZZZ", "ZoneBC_t")              == None
  assert I.request_node_from_name_and_label3(tree, "ZBCA", "ZoneGridConnectivity_t") == None
  node = I.get_node_from_name_and_label3(tree, "ZBCA", "ZoneBC_t")
  assert I.get_name(node) == "ZBCA" and I.get_label(node) == "ZoneBC_t"
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name_and_label3(tree, "ZZZZZ", "ZoneBC_t")
  with pytest.raises(I.CGNSNodeFromPredicateNotFoundError):
    I.get_node_from_name_and_label3(tree, "ZBCA", "ZoneGridConnectivity_t")

def test_getNodeFromPredicate():
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

  # requestNodeFrom...
  # ******************

  # requestNodeFromPredicate
  # ========================
  # Camel case
  # ----------
  base = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="bfs")
  assert is_zonei(zone)
  zone = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)
  base = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="bfs")
  assert is_zonei(zone)
  zone = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)

  # Snake case
  # ----------
  zone = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=1)
  assert zone is None
  zone = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=2)
  assert is_zonei(zone)
  zone = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=3)
  assert is_zonei(zone)

  node = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Base", search="dfs", depth=1)
  assert is_base(node)
  node = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=2)
  assert is_zonei(node)
  node = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Ngon", search="dfs", depth=3)
  assert is_ngon(node)
  node = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "bc1", search="dfs", depth=4)
  assert is_bc1(node)
  node = IE.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Index_i", search="dfs", depth=5)
  assert is_index_i(node)

  # requestNodeFrom{Name, Label, ...}
  # =================================
  # Camel case
  # ----------
  element = IE.requestNodeFromName(tree, "NFace")
  assert is_nface(element)
  element = IE.requestNodeFromValue(tree, np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.requestNodeFromLabel(tree, "Elements_t")
  assert is_ngon(element)
  element = IE.requestNodeFromNameAndValue(tree, "NFace", np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.requestNodeFromNameAndLabel(tree, "NFace", "Elements_t")
  assert is_nface(element)
  element = IE.requestNodeFromValueAndLabel(tree, np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)
  element = IE.requestNodeFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)
  element = IE.requestNodeFromName(tree, "TOTO")
  assert element is None
  element = IE.requestNodeFromValue(tree, np.array([1230,0], order='F'))
  assert element is None
  element = IE.requestNodeFromLabel(tree, "ZoneSubRegion_t")
  assert element is None
  element = IE.requestNodeFromNameAndValue(tree, "TOTO", np.array([23,0], order='F'))
  assert element is None
  element = IE.requestNodeFromNameAndValue(tree, "NFace", np.array([1230,0], order='F'))
  assert element is None
  element = IE.requestNodeFromNameAndLabel(tree, "TOTO", "Elements_t")
  assert element is None
  element = IE.requestNodeFromNameAndLabel(tree, "NFace", "ZoneSubRegion_t")
  assert element is None
  element = IE.requestNodeFromValueAndLabel(tree, np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = IE.requestNodeFromValueAndLabel(tree, np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None
  element = IE.requestNodeFromNameValueAndLabel(tree, "TOTO", np.array([23,0], order='F'), "Elements_t")
  assert element is None
  element = IE.requestNodeFromNameValueAndLabel(tree, "NFace", np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = IE.requestNodeFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None

  # Snake case
  # ----------
  element = IE.request_node_from_name(tree, "TOTO")
  assert element is None
  element = IE.request_node_from_value(tree, np.array([1230,0], order='F'))
  assert element is None
  element = IE.request_node_from_label(tree, "ZoneSubRegion_t")
  assert element is None
  element = IE.request_node_from_name_and_value(tree, "TOTO", np.array([23,0], order='F'))
  assert element is None
  element = IE.request_node_from_name_and_value(tree, "NFace", np.array([1230,0], order='F'))
  assert element is None
  element = IE.request_node_from_name_and_label(tree, "TOTO", "Elements_t")
  assert element is None
  element = IE.request_node_from_name_and_label(tree, "NFace", "ZoneSubRegion_t")
  assert element is None
  element = IE.request_node_from_value_and_label(tree, np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = IE.request_node_from_value_and_label(tree, np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None
  element = IE.request_node_from_name_value_and_label(tree, "TOTO", np.array([23,0], order='F'), "Elements_t")
  assert element is None
  element = IE.request_node_from_name_value_and_label(tree, "NFace", np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = IE.request_node_from_name_value_and_label(tree, "NFace", np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None

  # requestNodeFromPredicate{depth} and dfs
  # =======================================
  # Camel case
  zone = IE.requestNodeFromPredicate1(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert zone is None
  zone = IE.requestNodeFromPredicate2(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)
  zone = IE.requestNodeFromPredicate3(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)

  # Snake case
  zone = IE.request_node_from_predicate1(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert zone is None
  zone = IE.request_node_from_predicate2(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)
  zone = IE.request_node_from_predicate3(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)

  # requestNodeFrom{Name, Label, ...}{depth}
  # ========================================
  # Camel case
  # ----------
  zone = IE.requestNodeFromName(tree, "ZoneI", search="dfs", depth=1)
  assert zone is None
  zone = IE.requestNodeFromName(tree, "ZoneI", search="dfs", depth=2)
  assert is_zonei(zone)
  zone = IE.requestNodeFromName(tree, "ZoneI", search="dfs", depth=3)
  assert is_zonei(zone)

  zone = IE.requestNodeFromName1(tree, "ZoneI")
  assert zone is None
  zone = IE.requestNodeFromName2(tree, "ZoneI")
  assert is_zonei(zone)
  zone = IE.requestNodeFromName3(tree, "ZoneI")
  assert is_zonei(zone)

  node = IE.requestNodeFromName1(tree, "Base")
  assert is_base(node)
  node = IE.requestNodeFromName2(tree, "ZoneI")
  assert is_zonei(node)
  node = IE.requestNodeFromName3(tree, "Ngon")
  assert is_ngon(node)
  node = IE.requestNodeFromName4(tree, "bc1")
  assert is_bc1(node)
  node = IE.requestNodeFromName5(tree, "Index_i")
  assert is_index_i(node)

  node = IE.requestNodeFromLabel1(tree, "CGNSBase_t")
  assert is_base(node)
  node = IE.requestNodeFromLabel2(tree, "Zone_t")
  assert is_zonei(node)
  node = IE.requestNodeFromLabel3(tree, "Elements_t")
  assert is_ngon(node)
  node = IE.requestNodeFromLabel4(tree, "BC_t")
  assert is_bc1(node)

  # Snake case
  # ----------
  zone = IE.request_node_from_name(tree, "ZoneI", search="dfs", depth=1)
  assert zone is None
  zone = IE.request_node_from_name(tree, "ZoneI", search="dfs", depth=2)
  assert is_zonei(zone)
  zone = IE.request_node_from_name(tree, "ZoneI", search="dfs", depth=3)
  assert is_zonei(zone)

  zone = IE.request_node_from_name1(tree, "ZoneI")
  assert zone is None
  zone = IE.request_node_from_name2(tree, "ZoneI")
  assert is_zonei(zone)
  zone = IE.request_node_from_name3(tree, "ZoneI")
  assert is_zonei(zone)

  node = IE.request_node_from_name1(tree, "Base")
  assert is_base(node)
  node = IE.request_node_from_name2(tree, "ZoneI")
  assert is_zonei(node)
  node = IE.request_node_from_name3(tree, "Ngon")
  assert is_ngon(node)
  node = IE.request_node_from_name4(tree, "bc1")
  assert is_bc1(node)
  node = IE.request_node_from_name5(tree, "Index_i")
  assert is_index_i(node)

  node = IE.requestNodeFromLabel5(tree, "IndexArray_t")
  assert is_index_i(node)
  node = IE.request_node_from_label1(tree, "CGNSBase_t")
  assert is_base(node)
  node = IE.request_node_from_label2(tree, "Zone_t")
  assert is_zonei(node)
  node = IE.request_node_from_label3(tree, "Elements_t")
  assert is_ngon(node)
  node = IE.request_node_from_label4(tree, "BC_t")
  assert is_bc1(node)
  node = IE.request_node_from_label5(tree, "IndexArray_t")
  assert is_index_i(node)

  # getNodeFrom...
  # **************

  # getNodeFromPredicate
  # ====================
  # Camel case
  # ----------
  base = IE.getNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = IE.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)
  base = IE.get_node_from_predicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = IE.get_node_from_predicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)

  # Snake case
  # ----------
  element = IE.getNodeFromName(tree, "NFace")
  assert is_nface(element)
  element = IE.getNodeFromValue(tree, np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.getNodeFromLabel(tree, "Elements_t")
  assert is_ngon(element)
  element = IE.getNodeFromNameAndValue(tree, "NFace", np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.getNodeFromNameAndLabel(tree, "NFace", "Elements_t")
  assert is_nface(element)
  element = IE.getNodeFromValueAndLabel(tree, np.array([23,0], dtype='int64',order='F'), "Elements_t")
  assert is_nface(element)
  element = IE.getNodeFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)
  element = IE.get_node_from_name(tree, "NFace")
  assert is_nface(element)
  element = IE.get_node_from_value(tree, np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.get_node_from_label(tree, "Elements_t")
  assert is_ngon(element)
  element = IE.get_node_from_name_and_value(tree, "NFace", np.array([23,0], order='F'))
  assert is_nface(element)
  element = IE.get_node_from_name_and_label(tree, "NFace", "Elements_t")
  assert is_nface(element)
  element = IE.get_node_from_value_and_label(tree, np.array([23,0], dtype='int64',order='F'), "Elements_t")
  assert is_nface(element)
  element = IE.get_node_from_name_value_and_label(tree, "NFace", np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)

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
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  base = IE.getNodeFromLabel(tree, "CGNSBase_t")

  # walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  # walker.search = 'bfs'
  # for n in walker():
  #   print(f"n = {I.getName(n)}")
  # walker.search = 'dfs'
  # for n in walker():
  #   print(f"n = {I.getName(n)}")
  # walker.search = 'bfs'; walker.depth = 2
  # for n in walker():
  #   print(f"n = {I.getName(n)}")
  # walker.search = 'dfs'; walker.depth = 2
  # for n in walker():
  #   print(f"n = {I.getName(n)}")
  # walker.search = 'bfs'; explore='shallow'
  # for n in walker():
  #   print(f"n = {I.getName(n)}")
  # walker.search = 'dfs'; explore='shallow'
  # for n in walker():
  #   print(f"n = {I.getName(n)}")

  # walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  # for n in walker(search='bfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='dfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='bfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='dfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='dfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")

  # print(f"Begin caching.......")
  # walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  # for n in walker(search='bfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='bfs'):
  #   print(f"n = {I.getName(n)}")

  # for n in walker(search='dfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='dfs'):
  #   print(f"n = {I.getName(n)}")

  # for n in walker(search='bfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='bfs', depth=2):
  #   print(f"n = {I.getName(n)}")

  # for n in walker(search='dfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='dfs', depth=2):
  #   print(f"n = {I.getName(n)}")

  # for n in walker(search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")

  # for n in walker(search='dfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # for n in walker(search='dfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # print(f"End caching.......")

  # walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  # for n in walker(search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # walker.predicate = lambda n: n[3] == 'BC_t'
  # for n in walker(search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")

  walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  for n in walker():
    print(f"n = {I.getName(n)}")
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

  walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
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
  walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.parent = base; walker.depth = 1
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])

  # Test predicate
  walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  walker.explore='shallow'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  assert([I.getName(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # Test search
  walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.explore='shallow'; walker.search='bfs'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])
  assert(walker.search == 'bfs')
  walker.search = 'dfs'; walker.explore = 'shallow'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])

  # Test explore
  walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.explore='deep'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])
  assert(walker.explore == 'deep')
  walker.explore = 'shallow'
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])

  # Test depth
  walker = IE.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.depth = 2
  assert([I.getName(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in walker.cache] == [])
  assert(walker.depth == 2)
  walker.depth = 1
  assert([I.getName(n) for n in walker()] == [])
  assert([I.getName(n) for n in walker.cache] == [])

def test_iterNodesFromPredicate():
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

  # for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")

  # Camel case
  # ----------
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])

  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'ZoneBC_t')] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # alias for shallow exploring
  assert([I.getName(n) for n in IE.siterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.siterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.siterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in IE.siterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in IE.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in IE.iterNodesFromName(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromName1(tree, 'Zone*')] == [])
  assert([I.getName(n) for n in IE.iterNodesFromName2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iterNodesFromName3(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.siterNodesFromName(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])

  # Snake case
  # ----------
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])

  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'ZoneBC_t')] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # alias for shallow exploring
  assert([I.getName(n) for n in IE.siter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.siter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.siter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in IE.siter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in IE.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in IE.iter_nodes_from_name(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_name1(tree, 'Zone*')] == [])
  assert([I.getName(n) for n in IE.iter_nodes_from_name2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.iter_nodes_from_name3(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.siter_nodes_from_name(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])

  # Snake case
  # ----------

def test_getNodesFromPredicate():
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

  # for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")

  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromName(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromName1(tree, 'Zone*')] == [])
  assert([I.getName(n) for n in IE.getNodesFromName2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromName3(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromName4(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromName5(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_name(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_name1(tree, 'Zone*')] == [])
  assert([I.getName(n) for n in IE.get_nodes_from_name2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])

  ngon = np.array([22,0], order='F')
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon))] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='bfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=2)] == [])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=3)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getNodesFromValue(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getNodesFromValue1(tree, ngon)] == [])
  assert([I.getName(n) for n in IE.getNodesFromValue2(tree, ngon)] == [])
  assert([I.getName(n) for n in IE.getNodesFromValue3(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getNodesFromValue4(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getNodesFromValue5(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon))] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_value(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_value1(tree, ngon)] == [])
  assert([I.getName(n) for n in IE.get_nodes_from_value2(tree, ngon)] == [])
  assert([I.getName(n) for n in IE.get_nodes_from_value3(tree, ngon)] == ['NgonI', 'NgonJ'])

  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in IE.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromLabel(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromLabel1(tree, 'Zone_t')] == [])
  assert([I.getName(n) for n in IE.getNodesFromLabel2(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromLabel3(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromLabel4(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getNodesFromLabel5(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_label(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.get_nodes_from_label1(tree, 'Zone_t')] == [])
  assert([I.getName(n) for n in IE.get_nodes_from_label2(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])

def test_getNodesFromPredicate1():
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
  assert IE.getNodesFromPredicate1(base, lambda n: I.getName(n) == "ZoneI") == nodes_from_name1
  # Wildcard is not allowed in getNodesFromName1() from Cassiopee
  assert CI.getNodesFromName1(base, "Zone*") == []
  assert IE.getNodesFromPredicate1(base, lambda n: fnmatch.fnmatch(I.getName(n), "Zone*")) == nodes_from_name1
  assert IE.getNodesFromName1(base, "Zone*") == nodes_from_name1
  # Exclude top level which is included by Cassiopée
  assert IE.getNodesFromName1(base[0], "Base") == []

  # Test from Type
  nodes_from_type1 = I.getNodesFromType1(base, CGL.Zone_t.name)
  assert IE.getNodesFromPredicate1(base, lambda n: I.getType(n) == CGL.Zone_t.name) == nodes_from_type1
  assert IE.getNodesFromLabel1(base, CGL.Zone_t.name) == nodes_from_type1

  # Test from Value
  zone = I.getZones(tree)[0]
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [n for n in I.getNodesFromType1(zone, CGL.Elements_t.name) if np.array_equal(I.getVal(n), ngon_value)]
  assert IE.getNodesFromPredicate1(zone, lambda n: I.getType(n) == CGL.Elements_t.name and np.array_equal(I.getVal(n), ngon_value)) == elements_from_type_value1
  assert IE.getNodesFromValue1(zone, ngon_value) == elements_from_type_value1

  zonebcs_from_type_name1 = [n for n in I.getNodesFromType1(zone, CGL.ZoneBC_t.name) if I.getName(n) != "ZBCA"]
  assert IE.getNodesFromPredicate1(zone, lambda n: I.getType(n) == CGL.ZoneBC_t.name and I.getName(n) != "ZBCA") == zonebcs_from_type_name1

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
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  walker = IE.NodesWalker(tree, lambda n: I.getType(n) == "BC_t", search='bfs')
  assert([I.getName(n) for n in walker()] == ['bca1', 'bcd2', 'bcb3', 'bce4', 'bcc5'])
  walker.sort = lambda children:reversed(children)
  assert([I.getName(n) for n in walker()] == ['bcc5', 'bce4', 'bcb3', 'bcd2', 'bca1'])
  walker.sort = IE.NodesWalker.BACKWARD
  assert([I.getName(n) for n in walker()] == ['bcc5', 'bce4', 'bcb3', 'bcd2', 'bca1'])

  fsort = lambda children : sorted(children, key=lambda n : I.getName(n)[2])
  walker.sort = fsort
  # for n in walker(search='bfs', sort=fsort):
  #   print(f"n = {I.getName(n)}")
  assert([I.getName(n) for n in walker()] == ['bca1', 'bcd2', 'bcb3', 'bcc5', 'bce4'])

  walker = IE.NodesWalker(tree, lambda n: I.getType(n) == "FamilyName_t")
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
  walker.sort = IE.NodesWalker.BACKWARD
  assert([I.getValue(n) for n in walker()] == ['BCB5', 'BCE4', 'BCD3', 'ROW1', 'BCA2', 'BCC1'])
  # walker.search = 'bfs'
  # walker.sort = IE.NodesWalker.BACKWARD
  # for n in walker(search='bfs', sort=IE.NodesWalker.BACKWARD):
  #   print(f"n = {I.getValue(n)}")
  walker.search = 'bfs'
  walker.sort = IE.NodesWalker.BACKWARD
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
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  walker = IE.NodesWalker(tree, lambda n: I.getType(n) == "BC_t", search='bfs')
  walker.apply(lambda n : I.setName(n, I.getName(n).upper()))
  for n in walker():
    print(f"n = {I.getName(n)}")
  assert([I.getName(n) for n in walker()] == ['BCA1', 'BCD2', 'BCB3', 'BCE4', 'BCC5'])
  assert([I.getName(n) for n in walker.cache] == [])

  walker = IE.NodesWalker(tree, lambda n: I.getType(n) == "BC_t", search='dfs', caching=True)
  walker.apply(lambda n : I.setName(n, f"_{I.getName(n).upper()}"))
  for n in walker():
    print(f"n = {I.getName(n)}")
  assert([I.getName(n) for n in walker()] == ['_BCA1', '_BCD2', '_BCB3', '_BCE4', '_BCC5'])
  assert([I.getName(n) for n in walker.cache] == ['_BCA1', '_BCD2', '_BCB3', '_BCE4', '_BCC5'])

# def test_NodesWalkerPost_apply():
#   yt = """
# Base CGNSBase_t:
#   ZoneI Zone_t:
#     Ngon Elements_t [22,0]:
#     NFace Elements_t [23,0]:
#     ZBCA ZoneBC_t:
#       bca1 BC_t:
#         FamilyName FamilyName_t 'BCC1':
#         Index_i IndexArray_t:
#       bcd2 BC_t:
#         FamilyName FamilyName_t 'BCA2':
#         Index_ii IndexArray_t:
#     FamilyName FamilyName_t 'ROW1':
#     ZBCB ZoneBC_t:
#       bcb3 BC_t:
#         FamilyName FamilyName_t 'BCD3':
#         Index_iii IndexArray_t:
#       bce4 BC_t:
#         FamilyName FamilyName_t 'BCE4':
#       bcc5 BC_t:
#         FamilyName FamilyName_t 'BCB5':
#         Index_iv IndexArray_t:
#         Index_v IndexArray_t:
#         Index_vi IndexArray_t:
# """
#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t")
#   walker.apply(lambda n : I.setName(n, "Toto"))
#   # I.printTree(tree)
#   # print([I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["Toto"]*6
#   assert walker.cache == []

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", depth=3)
#   walker.apply(lambda n : I.setName(n, "Toto"))
#   # I.printTree(tree)
#   # print([I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*2+["Toto"]+["FamilyName"]*3
#   assert walker.cache == []

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", explore='shallow')
#   walker.apply(lambda n : I.setName(n, "Toto"))
#   # I.printTree(tree)
#   # print([I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["Toto"]*6
#   assert walker.cache == []

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", explore='shallow', depth=3)
#   walker.apply(lambda n : I.setName(n, "Toto"))
#   # I.printTree(tree)
#   # print([I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*2+["Toto"]+["FamilyName"]*3
#   assert walker.cache == []

#   # with cache
#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", caching=True)
#   walker.apply(lambda n : I.setName(n, "Toto"))
#   # I.printTree(tree)
#   # print([I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["Toto"]*6
#   assert [I.getName(n) for n in walker.cache] == ["Toto"]*6

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", depth=3, caching=True)
#   walker.apply(lambda n : I.setName(n, "Toto"))
#   # I.printTree(tree)
#   # print([I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*2+["Toto"]+["FamilyName"]*3
#   assert [I.getName(n) for n in walker.cache] == ["Toto"]

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", explore='shallow', caching=True)
#   walker.apply(lambda n : I.setName(n, "Toto"))
#   # I.printTree(tree)
#   # print([I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["Toto"]*6
#   assert [I.getName(n) for n in walker.cache] == ["Toto"]*6

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", explore='shallow', depth=3, caching=True)
#   walker.apply(lambda n : I.setName(n, "Toto"))
#   # I.printTree(tree)
#   # print([I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*2+["Toto"]+["FamilyName"]*3
#   assert [I.getName(n) for n in walker.cache] == ["Toto"]

#   # with delete
#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t")
#   walker.delete()
#   # I.printTree(tree)
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == []

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", depth=3)
#   walker.delete()
#   # I.printTree(tree)
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", explore='shallow')
#   walker.delete()
#   # I.printTree(tree)
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == []

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", explore='shallow', depth=3)
#   walker.delete()
#   # I.printTree(tree)
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

#   # delete with sort
#   sort = lambda children:reversed(children)
#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", explore='shallow', sort=sort)
#   walker.delete()
#   # I.printTree(tree)
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == []

#   tree = parse_yaml_cgns.to_complete_pytree(yt)
#   walker = IE.NodesWalkerPost(tree, lambda n: I.getType(n) == "FamilyName_t", explore='shallow', depth=3, sort=sort)
#   walker.delete()
#   # I.printTree(tree)
#   assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

def test_rmChildrenFromPredicate():
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
  # tree = parse_yaml_cgns.to_complete_pytree(yt)
  # for bc_node in IE.iterNodesFromLabel(tree, "BC_t"):
  #   IE.rmChildrenFromPredicate(bc_node, lambda n: I.getType(n) == "FamilyName_t" and int(I.getValue(n)[-1]) > 4)
  # # I.printTree(tree)
  # print([I.getValue(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])
  # assert [I.getValue(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ['BCC1', 'BCA2', 'ROW1', 'BCD3', 'BCE4']

  # tree = parse_yaml_cgns.to_complete_pytree(yt)
  # for bc_node in IE.iterNodesFromLabel(tree, "BC_t"):
  #   IE.keepChildrenFromPredicate(bc_node, lambda n: I.getType(n) == "FamilyName_t" and int(I.getValue(n)[-1]) > 4)
  # # I.printTree(tree)
  # print([I.getValue(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")])

  tree = parse_yaml_cgns.to_complete_pytree(yt)

  walker = IE.NodesWalker(tree, lambda n: I.getType(n) == "BC_t")
  for n in walker():
    print(f"n = {I.getName(n)}")
  print("--------------")

  walker = IE.NodesWalkers(tree, ["BC_t", "FamilyName_t"])
  for n in walker():
    print(f"n = {I.getValue(n)}")
  print("--------------")

  walker = IE.NodesWalkers(tree, ["BC_t", "FamilyName_t"], caching=True)
  for n in walker():
    print(f"n = {I.getValue(n)}")
  print("--------------")

  predicates = [{'predicate':"BC_t", 'explore':'shallow'}, {'predicate':"FamilyName_t", 'depth':1}]
  walker = IE.NodesWalkers(tree, predicates)
  for n in walker():
    print(f"n = {I.getValue(n)}")
  print("--------------")

  walker = IE.NodesWalkers(tree, predicates, caching=True)
  for n in walker():
    print(f"n = {I.getValue(n)}")
  print("--------------")

  # walker = IE.NodesWalkers(tree, ["BC_t", "FamilyName_t"], for_each=[{'explore':'shallow'}, {'depth':1}])
  # for n in walker():
  #   print(f"n = {I.getValue(n)}")
  # print("--------------")

  # walker = IE.NodesWalkers(tree, ["BC_t", "FamilyName_t"], for_each=[{'explore':'shallow'}, {'depth':1}], caching=True)
  # for n in walker():
  #   print(f"n = {I.getValue(n)}")
  # print("--------------")

  # tree = parse_yaml_cgns.to_complete_pytree(yt)
  # for n in IE.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"]):
  #   print(f"n = {I.getValue(n)}")
  # print("--------------")
  # for n in IE.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow'):
  #   print(f"n = {I.getValue(n)}")
  # print("--------------")
  # for n in IE.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], for_each=[{'explore':'shallow'}, {'depth':1}]):
  #   print(f"n = {I.getValue(n)}")
  # print("--------------")
  # assert [[I.getValue(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] for n in IE.getNodesFromLabel(tree, "BC_t")] == ['BCB5']

def test_rmNodesromPredicate():
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
  # camel_case
  # ==========
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rmNodesFromPredicate(tree, lambda n: I.getType(n) == "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rmNodesFromPredicate3(tree, lambda n: I.getType(n) == "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

  # Name
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rmNodesFromName(tree, "FamilyName")
  # I.printTree(tree)
  assert [I.getName(n) for n in IE.getNodesFromName(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rmNodesFromName3(tree, "FamilyName")
  # I.printTree(tree)
  assert [I.getName(n) for n in IE.getNodesFromName(tree, "FamilyName")] == ["FamilyName"]*5

  # Label
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rmNodesFromLabel(tree, "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in IE.getNodesFromName(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rmNodesFromLabel3(tree, "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in IE.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

  # snake_case
  # ==========
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rm_nodes_from_predicate(tree, lambda n: I.getType(n) == "FamilyName_t")
  # I.printTree(tree)
  assert [I.getValue(n) for n in IE.get_nodes_from_label(tree, "FamilyName_t")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rm_nodes_from_predicate3(tree, lambda n: I.getType(n) == "FamilyName_t")
  # I.printTree(tree)
  print([I.getValue(n) for n in IE.get_nodes_from_label(tree, "FamilyName_t")])
  assert [I.getValue(n) for n in IE.get_nodes_from_label(tree, "FamilyName_t")] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # Name
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rm_nodes_from_name(tree, "FamilyName")
  # I.printTree(tree)
  assert [I.getValue(n) for n in IE.get_nodes_from_name(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rm_nodes_from_name3(tree, "FamilyName")
  # I.printTree(tree)
  assert [I.getValue(n) for n in IE.get_nodes_from_name(tree, "FamilyName")] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # Label
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rm_nodes_from_label(tree, "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in IE.get_nodes_from_label(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  IE.rm_nodes_from_label3(tree, "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in IE.get_nodes_from_label(tree, "FamilyName_t")] == ["FamilyName"]*5

def test_getAllLabel():
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
  assert([I.getName(n) for n in IE.getAllBase(tree)] == ['Base'])
  assert([I.getName(n) for n in IE.getAllZone(tree)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getAllElements(tree)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.getAllZoneBC(tree)] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in IE.getAllBC(tree)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in IE.get_all_base(tree)] == ['Base'])
  assert([I.getName(n) for n in IE.get_all_zone(tree)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.get_all_elements(tree)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.get_all_zone_bc(tree)] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in IE.get_all_bc(tree)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in IE.getAllBase1(tree)] == ['Base'])
  assert([I.getName(n) for n in IE.getAllZone1(tree)] == [])
  assert([I.getName(n) for n in IE.getAllZone2(tree)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.getAllElements1(tree)] == [])
  assert([I.getName(n) for n in IE.getAllElements2(tree)] == [])
  assert([I.getName(n) for n in IE.getAllElements3(tree)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in IE.get_all_base1(tree)] == ['Base'])
  assert([I.getName(n) for n in IE.get_all_zone1(tree)] == [])
  assert([I.getName(n) for n in IE.get_all_zone2(tree)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in IE.get_all_elements1(tree)] == [])
  assert([I.getName(n) for n in IE.get_all_elements2(tree)] == [])
  assert([I.getName(n) for n in IE.get_all_elements3(tree)] == ['NgonI', 'NgonJ'])

def test_getCGNSName():
  yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Coordinates GridCoordinates_t:
      CoordinateX DataArray_t:
      CoordinateY DataArray_t:
      CoordinateZ DataArray_t:
"""
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  assert(I.getName(IE.getCoordinateX(tree)) == 'CoordinateX')
  assert(I.getName(IE.getCoordinateY(tree)) == 'CoordinateY')
  assert(I.getName(IE.getCoordinateZ(tree)) == 'CoordinateZ')
  assert(I.getName(IE.get_coordinate_x(tree)) == 'CoordinateX')
  assert(I.getName(IE.get_coordinate_y(tree)) == 'CoordinateY')
  assert(I.getName(IE.get_coordinate_z(tree)) == 'CoordinateZ')

  grid_coordinates_node = IE.getNodeFromLabel(tree,'GridCoordinates_t')
  assert(I.getName(IE.getCoordinateX1(grid_coordinates_node)) == 'CoordinateX')
  assert(I.getName(IE.getCoordinateY1(grid_coordinates_node)) == 'CoordinateY')
  assert(I.getName(IE.getCoordinateZ1(grid_coordinates_node)) == 'CoordinateZ')
  assert(I.getName(IE.get_coordinate_x1(grid_coordinates_node)) == 'CoordinateX')
  assert(I.getName(IE.get_coordinate_y1(grid_coordinates_node)) == 'CoordinateY')
  assert(I.getName(IE.get_coordinate_z1(grid_coordinates_node)) == 'CoordinateZ')

  zone_node = IE.getNodeFromLabel(tree,'Zone_t')
  assert(I.getName(IE.getCoordinateX2(zone_node)) == 'CoordinateX')
  assert(I.getName(IE.getCoordinateY2(zone_node)) == 'CoordinateY')
  assert(I.getName(IE.getCoordinateZ2(zone_node)) == 'CoordinateZ')
  assert(I.getName(IE.get_coordinate_x2(zone_node)) == 'CoordinateX')
  assert(I.getName(IE.get_coordinate_y2(zone_node)) == 'CoordinateY')
  assert(I.getName(IE.get_coordinate_z2(zone_node)) == 'CoordinateZ')

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

  assert list(IE.iterNodesByMatching(zoneI, '')) == []
  assert list(IE.iterNodesByMatching(zoneI, 'BC_t')) == []
  assert list(IE.iterNodesByMatching(zoneI, 'Index4_ii')) == []

  onelvl = IE.iterNodesByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(node) for node in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = IE.iterNodesByMatching(zbcB, 'bc5')
  assert [I.getName(node) for node in onelvl] == ['bc5']

  twolvl = IE.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [I.getName(node) for node in twolvl] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  twolvl = IE.iterNodesByMatching(zoneI, 'ZoneBC_t/bc5')
  assert [I.getName(node) for node in twolvl] == ['bc5']
  twolvl = IE.iterNodesByMatching(zbcB, 'BC_t/IndexArray_t')
  assert [I.getName(node) for node in twolvl] == ['Index3_i','Index4_i','Index4_ii','Index4_iii']

  results3 = ['Index1_i','Index2_i','Index3_i','Index4_i','Index4_ii','Index4_iii']
  threelvl = IE.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  assert [I.getName(node) for node in threelvl] == results3
  assert len(list(IE.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t/Index*_i'))) == 4

  threelvl1 = IE.iterNodesByMatching(zoneI, "ZoneBC_t/BC_t/lambda n: I.getType(n) == 'IndexArray_t'")
  # print(f"threelvl1 = {[I.getName(node) for node in threelvl1]}")
  assert [I.getName(node) for node in threelvl1] == results3

  threelvl2 = IE.iterNodesByMatching(zoneI, ["ZoneBC_t", CGL.BC_t.name, lambda n: I.getType(n) == 'IndexArray_t'])
  # print(f"threelvl2 = {[I.getName(node) for node in threelvl2]}")
  assert [I.getName(node) for node in threelvl2] == results3

  threelvl3 = IE.iterNodesByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: I.getType(n) == 'IndexArray_t'])
  # print(f"threelvl3 = {[I.getName(node) for node in threelvl3]}")
  assert [I.getName(node) for node in threelvl3] == results3

  with pytest.raises(TypeError):
    list(IE.iterNodesByMatching(zoneI, 12))

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

  assert list(IE.iterNodesWithParentsByMatching(zoneI, '')) == []
  assert list(IE.iterNodesWithParentsByMatching(zoneI, 'BC_t')) == []

  onelvl = IE.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = IE.iterNodesWithParentsByMatching(zbcB, 'BC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = IE.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCA', 'bc1'), ('ZBCA', 'bc2'), ('ZBCB', 'bc3'), ('ZBCB', 'bc4'), ('ZBCB', 'bc5')]
  twolvl = IE.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/bc3')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCB', 'bc3')]
  twolvl = IE.iterNodesWithParentsByMatching(zbcB, 'BC_t/IndexArray_t')
  for bc in I.getNodesFromType1(zbcB, 'BC_t'):
    for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
      assert next(twolvl) == (bc, idx)

  threelvl = IE.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
        assert next(threelvl) == (zbc, bc, idx)

  threelvl = IE.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/PL*')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.iterNodesWithParentsByMatching(zoneI, ['ZoneBC_t', 'BC_t', 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, 'BC_t', 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  p = re.compile('PL[12]')
  threelvl = IE.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: p.match(I.getName(n))])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in [n for n in IE.getNodesFromName1(bc, 'PL*') if p.match(I.getName(n))]:
        assert next(threelvl) == (zbc, bc, pl)

  with pytest.raises(TypeError):
    list(IE.iterNodesWithParentsByMatching(zoneI, 12))

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
  with pytest.raises(IE.CGNSLabelNotEqualError):
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
  # test_getNodeFromPredicate()
  # test_requireNodeFromName()
  # test_requireNodeFromType()
  # test_getRequireNodeFromNameAndType()
  # test_NodesWalker()
  # test_NodesWalker_sort()
  # test_NodesWalker_apply()
  # test_NodesWalkerPost_apply()
  # test_rmChildrenFromPredicate()
  # test_rmNodesromPredicate()
  test_iterNodesFromPredicate()
  # test_getNodesFromPredicate()
  # test_getAllLabel()
  # test_getCGNSName()
