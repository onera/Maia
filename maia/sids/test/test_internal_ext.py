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
  assert I.is_valid_label('') == True
  assert I.is_valid_label('BC') == False
  assert I.is_valid_label('BC_t') == True
  assert I.is_valid_label('BC_toto') == False
  assert I.is_valid_label('FakeLabel_t') == False

  assert I.is_valid_label(CGL.BC_t.name) == True

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
  is_base    = lambda n: I.getName(n) == 'Base' and I.getLabel(n) == 'CGNSBase_t'
  is_zonei   = lambda n: I.getName(n) == 'ZoneI' and I.getLabel(n) == 'Zone_t'
  is_nface   = lambda n: I.getName(n) == 'NFace' and I.getLabel(n) == 'Elements_t'
  is_ngon    = lambda n: I.getName(n) == 'Ngon' and I.getLabel(n) == 'Elements_t'
  is_zbca    = lambda n: I.getName(n) == 'ZBCA' and I.getLabel(n) == 'ZoneBC_t'
  is_bc1     = lambda n: I.getName(n) == 'bc1' and I.getLabel(n) == 'BC_t'
  is_index_i = lambda n: I.getName(n) == 'Index_i' and I.getLabel(n) == 'IndexArray_t'

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  # requestNodeFrom...
  # ******************

  # requestNodeFromPredicate
  # ========================
  # Camel case
  # ----------
  base = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="bfs")
  assert is_zonei(zone)
  zone = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)
  base = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="bfs")
  assert is_zonei(zone)
  zone = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)

  # Snake case
  # ----------
  zone = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=1)
  assert zone is None
  zone = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=2)
  assert is_zonei(zone)
  zone = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=3)
  assert is_zonei(zone)

  node = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Base", search="dfs", depth=1)
  assert is_base(node)
  node = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=2)
  assert is_zonei(node)
  node = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Ngon", search="dfs", depth=3)
  assert is_ngon(node)
  node = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "bc1", search="dfs", depth=4)
  assert is_bc1(node)
  node = I.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Index_i", search="dfs", depth=5)
  assert is_index_i(node)

  # requestNodeFrom{Name, Label, ...}
  # =================================
  # Camel case
  # ----------
  element = I.requestNodeFromName(tree, "NFace")
  assert is_nface(element)
  element = I.requestNodeFromValue(tree, np.array([23,0], order='F'))
  assert is_nface(element)
  element = I.requestNodeFromLabel(tree, "Elements_t")
  assert is_ngon(element)
  element = I.requestNodeFromNameAndValue(tree, "NFace", np.array([23,0], order='F'))
  assert is_nface(element)
  element = I.requestNodeFromNameAndLabel(tree, "NFace", "Elements_t")
  assert is_nface(element)
  element = I.requestNodeFromValueAndLabel(tree, np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)
  element = I.requestNodeFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)
  element = I.requestNodeFromName(tree, "TOTO")
  assert element is None
  element = I.requestNodeFromValue(tree, np.array([1230,0], order='F'))
  assert element is None
  element = I.requestNodeFromLabel(tree, "ZoneSubRegion_t")
  assert element is None
  element = I.requestNodeFromNameAndValue(tree, "TOTO", np.array([23,0], order='F'))
  assert element is None
  element = I.requestNodeFromNameAndValue(tree, "NFace", np.array([1230,0], order='F'))
  assert element is None
  element = I.requestNodeFromNameAndLabel(tree, "TOTO", "Elements_t")
  assert element is None
  element = I.requestNodeFromNameAndLabel(tree, "NFace", "ZoneSubRegion_t")
  assert element is None
  element = I.requestNodeFromValueAndLabel(tree, np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = I.requestNodeFromValueAndLabel(tree, np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None
  element = I.requestNodeFromNameValueAndLabel(tree, "TOTO", np.array([23,0], order='F'), "Elements_t")
  assert element is None
  element = I.requestNodeFromNameValueAndLabel(tree, "NFace", np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = I.requestNodeFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None

  # Snake case
  # ----------
  element = I.request_node_from_name(tree, "TOTO")
  assert element is None
  element = I.request_node_from_value(tree, np.array([1230,0], order='F'))
  assert element is None
  element = I.request_node_from_label(tree, "ZoneSubRegion_t")
  assert element is None
  element = I.request_node_from_name_and_value(tree, "TOTO", np.array([23,0], order='F'))
  assert element is None
  element = I.request_node_from_name_and_value(tree, "NFace", np.array([1230,0], order='F'))
  assert element is None
  element = I.request_node_from_name_and_label(tree, "TOTO", "Elements_t")
  assert element is None
  element = I.request_node_from_name_and_label(tree, "NFace", "ZoneSubRegion_t")
  assert element is None
  element = I.request_node_from_value_and_label(tree, np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = I.request_node_from_value_and_label(tree, np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None
  element = I.request_node_from_name_value_and_label(tree, "TOTO", np.array([23,0], order='F'), "Elements_t")
  assert element is None
  element = I.request_node_from_name_value_and_label(tree, "NFace", np.array([1230,0], order='F'), "Elements_t")
  assert element is None
  element = I.request_node_from_name_value_and_label(tree, "NFace", np.array([23,0], order='F'), "ZoneSubRegion_t")
  assert element is None

  # requestNodeFromPredicate{depth} and dfs
  # =======================================
  # Camel case
  zone = I.requestNodeFromPredicate1(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert zone is None
  zone = I.requestNodeFromPredicate2(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)
  zone = I.requestNodeFromPredicate3(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)

  # Snake case
  zone = I.request_node_from_predicate1(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert zone is None
  zone = I.request_node_from_predicate2(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)
  zone = I.request_node_from_predicate3(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)

  # requestNodeFrom{Name, Label, ...}{depth}
  # ========================================
  # Camel case
  # ----------
  zone = I.requestNodeFromName(tree, "ZoneI", search="dfs", depth=1)
  assert zone is None
  zone = I.requestNodeFromName(tree, "ZoneI", search="dfs", depth=2)
  assert is_zonei(zone)
  zone = I.requestNodeFromName(tree, "ZoneI", search="dfs", depth=3)
  assert is_zonei(zone)

  zone = I.requestNodeFromName1(tree, "ZoneI")
  assert zone is None
  zone = I.requestNodeFromName2(tree, "ZoneI")
  assert is_zonei(zone)
  zone = I.requestNodeFromName3(tree, "ZoneI")
  assert is_zonei(zone)

  node = I.requestNodeFromName1(tree, "Base")
  assert is_base(node)
  node = I.requestNodeFromName2(tree, "ZoneI")
  assert is_zonei(node)
  node = I.requestNodeFromName3(tree, "Ngon")
  assert is_ngon(node)
  node = I.requestNodeFromName4(tree, "bc1")
  assert is_bc1(node)
  node = I.requestNodeFromName5(tree, "Index_i")
  assert is_index_i(node)

  node = I.requestNodeFromLabel1(tree, "CGNSBase_t")
  assert is_base(node)
  node = I.requestNodeFromLabel2(tree, "Zone_t")
  assert is_zonei(node)
  node = I.requestNodeFromLabel3(tree, "Elements_t")
  assert is_ngon(node)
  node = I.requestNodeFromLabel4(tree, "BC_t")
  assert is_bc1(node)

  # Snake case
  # ----------
  zone = I.request_node_from_name(tree, "ZoneI", search="dfs", depth=1)
  assert zone is None
  zone = I.request_node_from_name(tree, "ZoneI", search="dfs", depth=2)
  assert is_zonei(zone)
  zone = I.request_node_from_name(tree, "ZoneI", search="dfs", depth=3)
  assert is_zonei(zone)

  zone = I.request_node_from_name1(tree, "ZoneI")
  assert zone is None
  zone = I.request_node_from_name2(tree, "ZoneI")
  assert is_zonei(zone)
  zone = I.request_node_from_name3(tree, "ZoneI")
  assert is_zonei(zone)

  node = I.request_node_from_name1(tree, "Base")
  assert is_base(node)
  node = I.request_node_from_name2(tree, "ZoneI")
  assert is_zonei(node)
  node = I.request_node_from_name3(tree, "Ngon")
  assert is_ngon(node)
  node = I.request_node_from_name4(tree, "bc1")
  assert is_bc1(node)
  node = I.request_node_from_name5(tree, "Index_i")
  assert is_index_i(node)

  node = I.requestNodeFromLabel5(tree, "IndexArray_t")
  assert is_index_i(node)
  node = I.request_node_from_label1(tree, "CGNSBase_t")
  assert is_base(node)
  node = I.request_node_from_label2(tree, "Zone_t")
  assert is_zonei(node)
  node = I.request_node_from_label3(tree, "Elements_t")
  assert is_ngon(node)
  node = I.request_node_from_label4(tree, "BC_t")
  assert is_bc1(node)
  node = I.request_node_from_label5(tree, "IndexArray_t")
  assert is_index_i(node)

  # getNodeFrom...
  # **************

  # getNodeFromPredicate
  # ====================
  # Camel case
  # ----------
  base = I.getNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = I.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)
  base = I.get_node_from_predicate(tree, lambda n: I.getName(n) == "Base")
  assert is_base(base)
  zone = I.get_node_from_predicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")
  assert is_zonei(zone)

  # Snake case
  # ----------
  element = I.getNodeFromName(tree, "NFace")
  assert is_nface(element)
  element = I.getNodeFromValue(tree, np.array([23,0], order='F'))
  assert is_nface(element)
  element = I.getNodeFromLabel(tree, "Elements_t")
  assert is_ngon(element)
  element = I.getNodeFromNameAndValue(tree, "NFace", np.array([23,0], order='F'))
  assert is_nface(element)
  element = I.getNodeFromNameAndLabel(tree, "NFace", "Elements_t")
  assert is_nface(element)
  element = I.getNodeFromValueAndLabel(tree, np.array([23,0], dtype='int64',order='F'), "Elements_t")
  assert is_nface(element)
  element = I.getNodeFromNameValueAndLabel(tree, "NFace", np.array([23,0], order='F'), "Elements_t")
  assert is_nface(element)
  element = I.get_node_from_name(tree, "NFace")
  assert is_nface(element)
  element = I.get_node_from_value(tree, np.array([23,0], order='F'))
  assert is_nface(element)
  element = I.get_node_from_label(tree, "Elements_t")
  assert is_ngon(element)
  element = I.get_node_from_name_and_value(tree, "NFace", np.array([23,0], order='F'))
  assert is_nface(element)
  element = I.get_node_from_name_and_label(tree, "NFace", "Elements_t")
  assert is_nface(element)
  element = I.get_node_from_value_and_label(tree, np.array([23,0], dtype='int64',order='F'), "Elements_t")
  assert is_nface(element)
  element = I.get_node_from_name_value_and_label(tree, "NFace", np.array([23,0], order='F'), "Elements_t")
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
  base = I.get_base(tree)

  # walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
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

  # walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
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
  # walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
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

  # walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  # for n in walker(search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # walker.predicate = lambda n: n[3] == 'BC_t'
  # for n in walker(search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")

  walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  for n in walker():
    print(f"n = {I.get_name(n)}")
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='bfs'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='dfs'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='bfs'; walker.depth=2
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='dfs'; walker.depth=2
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='bfs'; walker.explore='shallow'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])
  walker.search='dfs'; walker.explore='shallow'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ']); assert(walker.cache == [])

  walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  walker.search='bfs'
  assert([I.get_name(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.get_name(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  walker.predicate = lambda n: fnmatch.fnmatch(n[0], 'Zone*')
  walker.search='dfs'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  walker.search='bfs'; walker.depth=4
  assert([I.get_name(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.get_name(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  walker.predicate = lambda n: fnmatch.fnmatch(n[0], 'Zone*')
  walker.search='dfs'; walker.depth=4
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  walker.search='bfs'; walker.depth=4
  assert([I.get_name(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.get_name(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  walker.predicate = lambda n: fnmatch.fnmatch(n[0], 'Zone*')
  walker.search='dfs'; walker.depth=4
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])

  # Test parent
  walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.root = base; walker.depth = 1
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])

  # Test predicate
  walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), caching=True)
  walker.explore='shallow'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == ['ZoneI', 'ZoneJ'])
  walker.predicate = lambda n: n[3] == 'BC_t'
  assert([I.get_name(n) for n in walker()] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.get_name(n) for n in walker.cache] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # Test search
  walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.explore='shallow'; walker.search='bfs'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == [])
  assert(walker.search == 'bfs')
  walker.search = 'dfs'; walker.explore = 'shallow'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == [])

  # Test explore
  walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.explore='deep'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == [])
  assert(walker.explore == 'deep')
  walker.explore = 'shallow'
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == [])

  # Test depth
  walker = I.NodesWalker(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))
  walker.depth = 2
  assert([I.get_name(n) for n in walker()] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in walker.cache] == [])
  assert(walker.depth == 2)
  walker.depth = 1
  assert([I.get_name(n) for n in walker()] == [])
  assert([I.get_name(n) for n in walker.cache] == [])

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

  # for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")

  # Camel case
  # ----------
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])

  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'ZoneBC_t')] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # alias for shallow exploring
  assert([I.getName(n) for n in I.siterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.siterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.siterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in I.siterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='bfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in I.iterNodesFromPredicate(tree, lambda n: n[3] == 'BC_t', search='dfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in I.iterNodesFromName(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromName1(tree, 'Zone*')] == [])
  assert([I.getName(n) for n in I.iterNodesFromName2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iterNodesFromName3(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.siterNodesFromName(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])

  # Snake case
  # ----------
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])

  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'ZoneBC_t')] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', explore='shallow')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs', explore='shallow')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  # alias for shallow exploring
  assert([I.getName(n) for n in I.siter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.siter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.siter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in I.siter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs')] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='bfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])
  assert([I.getName(n) for n in I.iter_nodes_from_predicate(tree, lambda n: n[3] == 'BC_t', search='dfs', depth=4)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in I.iter_nodes_from_name(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_name1(tree, 'Zone*')] == [])
  assert([I.getName(n) for n in I.iter_nodes_from_name2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.iter_nodes_from_name3(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.siter_nodes_from_name(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])

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

  # for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs'):
  #   print(f"n = {I.getName(n)}")
  # for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2):
  #   print(f"n = {I.getName(n)}")
  # for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")
  # for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', explore='shallow'):
  #   print(f"n = {I.getName(n)}")

  # Camel case
  # ----------
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromName(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromName1(tree, 'Zone*')] == [])
  assert([I.getName(n) for n in I.getNodesFromName2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromName3(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromName4(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromName5(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])

  ngon = np.array([22,0], order='F')
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon))] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='bfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs')] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=2)] == [])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=3)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in I.getNodesFromValue(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in I.getNodesFromValue1(tree, ngon)] == [])
  assert([I.getName(n) for n in I.getNodesFromValue2(tree, ngon)] == [])
  assert([I.getName(n) for n in I.getNodesFromValue3(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in I.getNodesFromValue4(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in I.getNodesFromValue5(tree, ngon)] == ['NgonI', 'NgonJ'])

  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=1)] == [])
  assert([I.getName(n) for n in I.getNodesFromPredicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromLabel(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromLabel1(tree, 'Zone_t')] == [])
  assert([I.getName(n) for n in I.getNodesFromLabel2(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromLabel3(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromLabel4(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getNodesFromLabel5(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])

  # Snake case
  # ----------
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'))] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=1)] == [])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: fnmatch.fnmatch(n[0], 'Zone*'), search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_name(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_name1(tree, 'Zone*')] == [])
  assert([I.get_name(n) for n in I.get_nodes_from_name2(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_name3(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_name4(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_name5(tree, 'Zone*')] == ['ZoneI', 'ZoneJ'])

  ngon = np.array([22,0], order='F')
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon))] == ['NgonI', 'NgonJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='bfs')] == ['NgonI', 'NgonJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs')] == ['NgonI', 'NgonJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=1)] == [])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=2)] == [])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: np.array_equal(n[1], ngon), search='dfs', depth=3)] == ['NgonI', 'NgonJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_value(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_value1(tree, ngon)] == [])
  assert([I.get_name(n) for n in I.get_nodes_from_value2(tree, ngon)] == [])
  assert([I.get_name(n) for n in I.get_nodes_from_value3(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_value4(tree, ngon)] == ['NgonI', 'NgonJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_value5(tree, ngon)] == ['NgonI', 'NgonJ'])

  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='bfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=1)] == [])
  assert([I.get_name(n) for n in I.get_nodes_from_predicate(tree, lambda n: n[3] == 'Zone_t', search='dfs', depth=2)] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_label(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_label1(tree, 'Zone_t')] == [])
  assert([I.get_name(n) for n in I.get_nodes_from_label2(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_label3(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_label4(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_nodes_from_label5(tree, 'Zone_t')] == ['ZoneI', 'ZoneJ'])

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

  # Camel case
  # ----------
  # Test from Name
  base = I.getBase(tree)
  nodes_from_name1 = ["ZoneI"]
  assert [I.getName(n) for n in I.getNodesFromPredicate1(base, lambda n: I.getName(n) == "ZoneI")] == nodes_from_name1
  # Wildcard is not allowed in getNodesFromName1() from Cassiopee
  assert CI.getNodesFromName1(base, "Zone*") == []
  assert [I.getName(n) for n in I.getNodesFromPredicate1(base, lambda n: fnmatch.fnmatch(I.getName(n), "Zone*"))] == nodes_from_name1
  assert [I.getName(n) for n in I.getNodesFromName1(base, "Zone*")] == nodes_from_name1
  assert [I.getName(n) for n in I.getNodesFromName1(base, "Base")] == ["Base"]

  # Test from Type
  nodes_from_type1 = ["Zone_t"]
  assert [I.getLabel(n) for n in I.getNodesFromPredicate1(base, lambda n: I.getLabel(n) == CGL.Zone_t.name)] == nodes_from_type1
  assert [I.getLabel(n) for n in I.getNodesFromLabel1(base, CGL.Zone_t.name)] == nodes_from_type1

  # Test from Value
  zone = I.getZone(tree)
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [ngon_value[0]]
  assert [I.getVal(n)[0] for n in I.getNodesFromPredicate1(zone, lambda n: I.getLabel(n) == CGL.Elements_t.name and np.array_equal(I.getVal(n), ngon_value))] == elements_from_type_value1
  assert [I.getVal(n)[0] for n in I.getNodesFromValue1(zone, ngon_value)] == elements_from_type_value1

  zonebcs_from_type_name1 = ['ZBCB']
  assert [I.getName(n) for n in I.getNodesFromPredicate1(zone, lambda n: I.getLabel(n) == CGL.ZoneBC_t.name and I.getName(n) != "ZBCA")] == zonebcs_from_type_name1

  # Snake case
  # ----------
  # Test from Name
  base = I.get_base(tree)
  nodes_from_name1 = ["ZoneI"]
  assert [I.get_name(n) for n in I.get_nodes_from_predicate1(base, lambda n: I.get_name(n) == "ZoneI")] == nodes_from_name1
  assert [I.get_name(n) for n in I.get_nodes_from_predicate1(base, lambda n: fnmatch.fnmatch(I.get_name(n), "Zone*"))] == nodes_from_name1
  assert [I.get_name(n) for n in I.get_nodes_from_name1(base, "Zone*")] == nodes_from_name1
  assert [I.get_name(n) for n in I.get_nodes_from_name1(base, "Base")] == ["Base"]

  # Test from Type
  nodes_from_type1 = ["Zone_t"]
  assert [I.get_label(n) for n in I.get_nodes_from_predicate1(base, lambda n: I.get_label(n) == CGL.Zone_t.name)] == nodes_from_type1
  assert [I.get_label(n) for n in I.get_nodes_from_label1(base, CGL.Zone_t.name)] == nodes_from_type1

  # Test from Value
  zone = I.get_zone(tree)
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [ngon_value[0]]
  assert [I.get_val(n)[0] for n in I.get_nodes_from_predicate1(zone, lambda n: I.get_label(n) == CGL.Elements_t.name and np.array_equal(I.get_val(n), ngon_value))] == elements_from_type_value1
  assert [I.get_val(n)[0] for n in I.get_nodes_from_value1(zone, ngon_value)] == elements_from_type_value1

  zonebcs_from_type_name1 = ['ZBCB']
  assert [I.get_name(n) for n in I.get_nodes_from_predicate1(zone, lambda n: I.get_label(n) == CGL.ZoneBC_t.name and I.get_name(n) != "ZBCA")] == zonebcs_from_type_name1

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

  # Snake case
  # ==========
  walker = I.NodesWalker(tree, lambda n: I.getLabel(n) == "BC_t", search='bfs')
  assert([I.get_name(n) for n in walker()] == ['bca1', 'bcd2', 'bcb3', 'bce4', 'bcc5'])
  walker.sort = lambda children:reversed(children)
  assert([I.get_name(n) for n in walker()] == ['bcc5', 'bce4', 'bcb3', 'bcd2', 'bca1'])
  walker.sort = I.NodesWalker.BACKWARD
  assert([I.get_name(n) for n in walker()] == ['bcc5', 'bce4', 'bcb3', 'bcd2', 'bca1'])

  fsort = lambda children : sorted(children, key=lambda n : I.get_name(n)[2])
  walker.sort = fsort
  # for n in walker(search='bfs', sort=fsort):
  #   print(f"n = {I.get_name(n)}")
  assert([I.get_name(n) for n in walker()] == ['bca1', 'bcd2', 'bcb3', 'bcc5', 'bce4'])

  walker = I.NodesWalker(tree, lambda n: I.getLabel(n) == "FamilyName_t")
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
  walker.sort = I.NodesWalker.BACKWARD
  assert([I.getValue(n) for n in walker()] == ['BCB5', 'BCE4', 'BCD3', 'ROW1', 'BCA2', 'BCC1'])
  # walker.search = 'bfs'
  # walker.sort = I.NodesWalker.BACKWARD
  # for n in walker(search='bfs', sort=I.NodesWalker.BACKWARD):
  #   print(f"n = {I.getValue(n)}")
  walker.search = 'bfs'
  walker.sort = I.NodesWalker.BACKWARD
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

  walker = I.NodesWalker(tree, lambda n: I.getLabel(n) == "BC_t", search='bfs')
  walker.apply(lambda n : I.setName(n, I.get_name(n).upper()))
  for n in walker():
    print(f"n = {I.get_name(n)}")
  assert([I.get_name(n) for n in walker()] == ['BCA1', 'BCD2', 'BCB3', 'BCE4', 'BCC5'])
  assert([I.get_name(n) for n in walker.cache] == [])

  walker = I.NodesWalker(tree, lambda n: I.getLabel(n) == "BC_t", search='dfs', caching=True)
  walker.apply(lambda n : I.setName(n, f"_{I.get_name(n).upper()}"))
  for n in walker():
    print(f"n = {I.get_name(n)}")
  assert([I.get_name(n) for n in walker()] == ['_BCA1', '_BCD2', '_BCB3', '_BCE4', '_BCC5'])
  assert([I.get_name(n) for n in walker.cache] == ['_BCA1', '_BCD2', '_BCB3', '_BCE4', '_BCC5'])

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
  tree = parse_yaml_cgns.to_complete_pytree(yt)

  walker = I.NodesWalker(tree, lambda n: I.getLabel(n) == "FamilyName_t")
  assert [I.getValue(n) for n in walker()] == ['ROW1', 'BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # All predicates have the same options
  # ------------------------------------
  # Avoid Node : FamilyName FamilyName_t 'ROW1'
  walker = I.NodesWalkers(tree, ["BC_t", "FamilyName_t"])
  print(f"nodes = {[I.getValue(n) for n in walker()]}")
  assert [I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in walker.cache] == []

  # Avoid Node : FamilyName FamilyName_t 'ROW1', with caching
  walker = I.NodesWalkers(tree, ["BC_t", "FamilyName_t"], caching=True)
  print(f"nodes = {[I.getValue(n) for n in walker()]}")
  assert [I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in walker.cache] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  fpathv = lambda nodes: '/'.join([I.get_name(n) for n in nodes[:-1]]+[I.get_value(nodes[-1])])
  # ... with ancestors
  walker = I.NodesWalkers(tree, ["BC_t", "FamilyName_t"], ancestors = True)
  # for nodes in walker():
  #   print(f"nodes = {nodes}")
  print(f"nodes = {[[I.get_name(n) for n in nodes] for nodes in walker()]}")
  assert [fpathv(nodes) for nodes in walker()] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in walker.cache] == []

  # ... with ancestors and caching
  walker = I.NodesWalkers(tree, ["BC_t", "FamilyName_t"], caching=True, ancestors=True)
  # for nodes in walker():
  #   print(f"nodes = {nodes}")
  print(f"nodes = {[[I.get_name(n) for n in nodes] for nodes in walker()]}")
  assert [fpathv(nodes) for nodes in walker()] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in walker.cache] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # Each predicate has theirs owns options
  # --------------------------------------
  predicates = [{'predicate':"BC_t", 'explore':'shallow'}, {'predicate':"FamilyName_t", 'depth':1}]

  # Avoid Node : FamilyName FamilyName_t 'ROW1'
  walker = I.NodesWalkers(tree, predicates)
  print(f"nodes = {[I.getValue(n) for n in walker()]}")
  assert [I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in walker.cache] == []

  # Avoid Node : FamilyName FamilyName_t 'ROW1', with caching
  walker = I.NodesWalkers(tree, predicates, caching=True)
  print(f"nodes = {[I.getValue(n) for n in walker()]}")
  assert [I.getValue(n) for n in walker()] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in walker.cache] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  fpathv = lambda nodes: '/'.join([I.get_name(n) for n in nodes[:-1]]+[I.get_value(nodes[-1])])
  # ... with ancestors
  walker = I.NodesWalkers(tree, predicates, ancestors = True)
  # for nodes in walker():
  #   print(f"nodes = {nodes}")
  print(f"nodes = {[[I.get_name(n) for n in nodes] for nodes in walker()]}")
  assert [fpathv(nodes) for nodes in walker()] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in walker.cache] == []

  # ... with ancestors and caching
  walker = I.NodesWalkers(tree, predicates, caching=True, ancestors=True)
  # for nodes in walker():
  #   print(f"nodes = {nodes}")
  print(f"nodes = {[[I.get_name(n) for n in nodes] for nodes in walker()]}")
  assert [fpathv(nodes) for nodes in walker()] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in walker.cache] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

def test_iterNodesFromPredicates():
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
  base = I.get_base(tree)

  assert [I.getValue(n) for n in I.iterNodesFromPredicates(tree, [lambda n: I.getLabel(n) == "FamilyName_t"])] == ['ROW1', 'BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # All predicates have the same options
  # ------------------------------------
  # Avoid Node : FamilyName FamilyName_t 'ROW1', traversal=deep
  assert [I.getValue(n) for n in I.iterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"])] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  fpathv = lambda nodes: '/'.join([I.get_name(n) for n in nodes[:-1]]+[I.get_value(nodes[-1])])
  assert [fpathv(nodes) for nodes in I.iterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], ancestors=True)] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # Avoid Node : FamilyName FamilyName_t 'ROW1', traversal=shallow
  assert [I.getValue(n) for n in I.iterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow')] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in I.siterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"])] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  assert [fpathv(nodes) for nodes in I.iterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow', ancestors=True)] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in I.siterNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], ancestors=True)] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # With search='dfs', depth=1 -> iterNodesByMatching
  assert [I.getValue(n) for n in I.iterNodesFromPredicates(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', depth=1)] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in I.iterNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs')] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  assert [I.getValue(n) for n in I.iterNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"])] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  assert [fpathv(nodes) for nodes in I.iterNodesFromPredicates(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', depth=1, ancestors=True)] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in I.iterNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', ancestors=True)] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]
  assert [fpathv(nodes) for nodes in I.iterNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], ancestors=True)] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]

  # Each predicate has theirs owns options
  # --------------------------------------
  predicates = [{'predicate':"BC_t", 'explore':'shallow'}, {'predicate':"FamilyName_t", 'depth':1}]

  # Avoid Node : FamilyName FamilyName_t 'ROW1'
  assert [I.getValue(n) for n in I.iterNodesFromPredicates(tree, predicates)] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  assert [fpathv(nodes) for nodes in I.iterNodesFromPredicates(tree, predicates, ancestors=True)] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

def test_getNodesFromPredicates():
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
  base = I.get_base(tree)

  results = I.getNodesFromPredicates(tree, [lambda n: I.getLabel(n) == "FamilyName_t"])
  assert [I.getValue(n) for n in results] == ['ROW1', 'BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # All predicates have the same options
  # ------------------------------------
  # Avoid Node : FamilyName FamilyName_t 'ROW1', traversal=deep
  results = I.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"])
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  fpathv = lambda nodes: '/'.join([I.get_name(n) for n in nodes[:-1]]+[I.get_value(nodes[-1])])
  results = I.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # Avoid Node : FamilyName FamilyName_t 'ROW1', traversal=shallow
  results = I.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow')
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  results = I.sgetNodesFromPredicates(tree, ["BC_t", "FamilyName_t"])
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  results = I.getNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], explore='shallow', ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]
  results = I.sgetNodesFromPredicates(tree, ["BC_t", "FamilyName_t"], ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

  # With search='dfs', depth=1 -> iterNodesByMatching
  results = I.getNodesFromPredicates(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', depth=1)
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  results = I.getNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs')
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  results = I.getNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"])
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  results = I.getNodesFromPredicates(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', depth=1, ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]
  results = I.getNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], search='dfs', ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]
  results = I.getNodesFromPredicates1(base, ["Zone_t", "ZoneBC_t", "BC_t", "FamilyName_t"], ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["ZoneI/ZBCA/bca1/BCC1", "ZoneI/ZBCA/bcd2/BCA2", "ZoneI/ZBCB/bcb3/BCD3", "ZoneI/ZBCB/bce4/BCE4", "ZoneI/ZBCB/bcc5/BCB5"]

  # Each predicate has theirs owns options
  # --------------------------------------
  predicates = [{'predicate':"BC_t", 'explore':'shallow'}, {'predicate':"FamilyName_t", 'depth':1}]

  # Avoid Node : FamilyName FamilyName_t 'ROW1'
  results = I.getNodesFromPredicates(tree, predicates)
  assert [I.getValue(n) for n in results] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']
  # ... with ancestors
  results = I.getNodesFromPredicates(tree, predicates, ancestors=True)
  assert [fpathv(nodes) for nodes in results] == ["bca1/BCC1", "bcd2/BCA2", "bcb3/BCD3", "bce4/BCE4", "bcc5/BCB5"]

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
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  for bc_node in I.iterNodesFromLabel(tree, "BC_t"):
    I.rmChildrenFromPredicate(bc_node, lambda n: I.getLabel(n) == "FamilyName_t" and int(I.get_value(n)[-1]) > 4)
  # I.printTree(tree)
  # print([I.get_value(n) for n in I.getNodesFromLabel(tree, "FamilyName_t")])
  assert [I.get_value(n) for n in I.getNodesFromLabel(tree, "FamilyName_t")] == ['BCC1', 'BCA2', 'ROW1', 'BCD3', 'BCE4']

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  for bc_node in I.iterNodesFromLabel(tree, "BC_t"):
    I.keepChildrenFromPredicate(bc_node, lambda n: I.getLabel(n) == "FamilyName_t" and int(I.get_value(n)[-1]) > 4)
  # I.printTree(tree)
  # print([I.get_value(n) for n in I.getNodesFromLabel(tree, "FamilyName_t")])
  assert [I.get_value(n) for n in I.getNodesFromLabel(tree, "FamilyName_t")] == ['ROW1', 'BCB5']

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
  # Camel case
  # ==========
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rmNodesFromPredicate(tree, lambda n: I.getLabel(n) == "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in I.getNodesFromLabel(tree, "FamilyName_t")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rmNodesFromPredicate3(tree, lambda n: I.getLabel(n) == "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in I.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

  # Name
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rmNodesFromName(tree, "FamilyName")
  # I.printTree(tree)
  assert [I.getName(n) for n in I.getNodesFromName(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rmNodesFromName3(tree, "FamilyName")
  # I.printTree(tree)
  assert [I.getName(n) for n in I.getNodesFromName(tree, "FamilyName")] == ["FamilyName"]*5

  # Label
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rmNodesFromLabel(tree, "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in I.getNodesFromName(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rmNodesFromLabel3(tree, "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in I.getNodesFromLabel(tree, "FamilyName_t")] == ["FamilyName"]*5

  # Snake case
  # ==========
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rm_nodes_from_predicate(tree, lambda n: I.getLabel(n) == "FamilyName_t")
  # I.printTree(tree)
  assert [I.get_value(n) for n in I.get_nodes_from_label(tree, "FamilyName_t")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rm_nodes_from_predicate3(tree, lambda n: I.getLabel(n) == "FamilyName_t")
  # I.printTree(tree)
  print([I.get_value(n) for n in I.get_nodes_from_label(tree, "FamilyName_t")])
  assert [I.get_value(n) for n in I.get_nodes_from_label(tree, "FamilyName_t")] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # Name
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rm_nodes_from_name(tree, "FamilyName")
  # I.printTree(tree)
  assert [I.get_value(n) for n in I.get_nodes_from_name(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rm_nodes_from_name3(tree, "FamilyName")
  # I.printTree(tree)
  assert [I.get_value(n) for n in I.get_nodes_from_name(tree, "FamilyName")] == ['BCC1', 'BCA2', 'BCD3', 'BCE4', 'BCB5']

  # Label
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rm_nodes_from_label(tree, "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in I.get_nodes_from_label(tree, "FamilyName")] == []

  tree = parse_yaml_cgns.to_complete_pytree(yt)
  I.rm_nodes_from_label3(tree, "FamilyName_t")
  # I.printTree(tree)
  assert [I.getName(n) for n in I.get_nodes_from_label(tree, "FamilyName_t")] == ["FamilyName"]*5

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
  # Camel case
  # ==========
  assert([I.getName(n) for n in I.getAllBase(tree)] == ['Base'])
  assert([I.getName(n) for n in I.getAllZone(tree)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getAllElements(tree)] == ['NgonI', 'NgonJ'])
  assert([I.getName(n) for n in I.getAllZoneBC(tree)] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.getName(n) for n in I.getAllBC(tree)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.getName(n) for n in I.getAllBase1(tree)] == ['Base'])
  assert([I.getName(n) for n in I.getAllZone1(tree)] == [])
  assert([I.getName(n) for n in I.getAllZone2(tree)] == ['ZoneI', 'ZoneJ'])
  assert([I.getName(n) for n in I.getAllElements1(tree)] == [])
  assert([I.getName(n) for n in I.getAllElements2(tree)] == [])
  assert([I.getName(n) for n in I.getAllElements3(tree)] == ['NgonI', 'NgonJ'])

  # Snake case
  # ==========
  assert([I.get_name(n) for n in I.get_all_base(tree)] == ['Base'])
  assert([I.get_name(n) for n in I.get_all_zone(tree)] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_all_elements(tree)] == ['NgonI', 'NgonJ'])
  assert([I.get_name(n) for n in I.get_all_zone_bc(tree)] == ['ZBCAI', 'ZBCBI', 'ZBCAJ', 'ZBCBJ'])
  assert([I.get_name(n) for n in I.get_all_bc(tree)] == ['bc1I', 'bc2', 'bc3I', 'bc4', 'bc5', 'bc1J', 'bc3J'])

  assert([I.get_name(n) for n in I.get_all_base1(tree)] == ['Base'])
  assert([I.get_name(n) for n in I.get_all_zone1(tree)] == [])
  assert([I.get_name(n) for n in I.get_all_zone2(tree)] == ['ZoneI', 'ZoneJ'])
  assert([I.get_name(n) for n in I.get_all_elements1(tree)] == [])
  assert([I.get_name(n) for n in I.get_all_elements2(tree)] == [])
  assert([I.get_name(n) for n in I.get_all_elements3(tree)] == ['NgonI', 'NgonJ'])

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

  # Camel case
  # ==========
  assert(I.getName(I.getCoordinateX(tree)) == 'CoordinateX')
  assert(I.getName(I.getCoordinateY(tree)) == 'CoordinateY')
  assert(I.getName(I.getCoordinateZ(tree)) == 'CoordinateZ')

  grid_coordinates_node = I.getNodeFromLabel(tree,'GridCoordinates_t')
  assert(I.getName(I.getCoordinateX1(grid_coordinates_node)) == 'CoordinateX')
  assert(I.getName(I.getCoordinateY1(grid_coordinates_node)) == 'CoordinateY')
  assert(I.getName(I.getCoordinateZ1(grid_coordinates_node)) == 'CoordinateZ')

  zone_node = I.getNodeFromLabel(tree,'Zone_t')
  assert(I.getName(I.getCoordinateX2(zone_node)) == 'CoordinateX')
  assert(I.getName(I.getCoordinateY2(zone_node)) == 'CoordinateY')
  assert(I.getName(I.getCoordinateZ2(zone_node)) == 'CoordinateZ')

  # Snake case
  # ==========
  assert(I.get_name(I.get_coordinate_x(tree)) == 'CoordinateX')
  assert(I.get_name(I.get_coordinate_y(tree)) == 'CoordinateY')
  assert(I.get_name(I.get_coordinate_z(tree)) == 'CoordinateZ')

  grid_coordinates_node = I.get_node_from_label(tree,'GridCoordinates_t')
  assert(I.get_name(I.get_coordinate_x1(grid_coordinates_node)) == 'CoordinateX')
  assert(I.get_name(I.get_coordinate_y1(grid_coordinates_node)) == 'CoordinateY')
  assert(I.get_name(I.get_coordinate_z1(grid_coordinates_node)) == 'CoordinateZ')

  zone_node = I.get_node_from_label(tree,'Zone_t')
  assert(I.get_name(I.get_coordinate_x2(zone_node)) == 'CoordinateX')
  assert(I.get_name(I.get_coordinate_y2(zone_node)) == 'CoordinateY')
  assert(I.get_name(I.get_coordinate_z2(zone_node)) == 'CoordinateZ')

def test_getNodesDispatch1():
  fs = I.newFlowSolution()
  data_a   = I.newDataArray('DataA', [1,2,3], parent=fs)
  data_b   = I.newDataArray('DataB', [4,6,8], parent=fs)
  grid_loc = I.newGridLocation('Vertex', fs)
  assert I.getNodesDispatch1(fs, 'DataB') == [data_b]
  assert I.getNodesDispatch1(fs, 'DataArray_t') == [data_a, data_b]
  assert I.getNodesDispatch1(fs, CGL.GridLocation_t) == [grid_loc]
  assert I.getNodesDispatch1(fs, np.array([4,6,8])) == [data_b]
  assert I.getNodesDispatch1(fs, lambda n: isinstance(I.getValue(n), str) and I.getValue(n) == 'Vertex') == [grid_loc]
  with pytest.raises(TypeError):
    I.getNodesDispatch1(fs, False)

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

  # Equivalence with NodesWalkers
  # -----------------------------
  walker = I.NodesWalkers(zoneI, [''], search='dfs', depth=1)
  assert list(walker()) == []
  walker.predicates = ['BC_t']
  assert list(walker()) == []
  walker.predicates = ['Index4_ii']
  assert list(walker()) == []

  walker.predicates = ['ZoneBC_t']
  assert [I.getName(node) for node in walker()] == ['ZBCA', 'ZBCB']
  walker.root       = zbcB
  walker.predicates = ['bc5']
  assert [I.getName(node) for node in walker()] == ['bc5']

  walker.root       = zoneI
  walker.predicates = ['ZoneBC_t', 'BC_t']
  assert [I.getName(node) for node in walker()] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  walker.predicates = ['ZoneBC_t', 'bc5']
  assert [I.getName(node) for node in walker()] == ['bc5']

  results3 = ['Index1_i','Index2_i','Index3_i','Index4_i','Index4_ii','Index4_iii']

  walker.predicates = ['ZoneBC_t', 'BC_t','IndexArray_t']
  assert [I.getName(node) for node in walker()] == results3
  walker.predicates = ['ZoneBC_t', 'BC_t','Index*_i']
  assert([I.getName(node) for node in walker()] == ['Index1_i','Index2_i','Index3_i','Index4_i'])

  walker.predicates = ["ZoneBC_t", CGL.BC_t.name, lambda n: I.getLabel(n) == 'IndexArray_t']
  # print(f"threelvl2 = {[I.getName(node) for node in walker()]}")
  assert [I.getName(node) for node in walker()] == results3

  walker.predicates = [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: I.getLabel(n) == 'IndexArray_t']
  # print(f"threelvl3 = {[I.getName(node) for node in walker()]}")
  assert [I.getName(node) for node in walker()] == results3

  with pytest.raises(TypeError):
    walker.predicates = 12

  # iterNodesByMatching
  # -------------------
  assert list(I.iterNodesByMatching(zoneI, '')) == []
  assert list(I.iterNodesByMatching(zoneI, 'BC_t')) == []
  assert list(I.iterNodesByMatching(zoneI, 'Index4_ii')) == []

  onelvl = I.iterNodesByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(node) for node in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = I.iterNodesByMatching(zbcB, 'bc5')
  assert [I.getName(node) for node in onelvl] == ['bc5']

  twolvl = I.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [I.getName(node) for node in twolvl] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  twolvl = I.iterNodesByMatching(zoneI, 'ZoneBC_t/bc5')
  assert [I.getName(node) for node in twolvl] == ['bc5']
  twolvl = I.iterNodesByMatching(zbcB, 'BC_t/IndexArray_t')
  assert [I.getName(node) for node in twolvl] == ['Index3_i','Index4_i','Index4_ii','Index4_iii']

  results3 = ['Index1_i','Index2_i','Index3_i','Index4_i','Index4_ii','Index4_iii']
  threelvl = I.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  assert [I.getName(node) for node in threelvl] == results3
  assert len(list(I.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t/Index*_i'))) == 4

  # threelvl1 = I.iterNodesByMatching(zoneI, "ZoneBC_t/BC_t/lambda n: I.getType(n) == 'IndexArray_t'")
  # # print(f"threelvl1 = {[I.getName(node) for node in threelvl1]}")
  # assert [I.getName(node) for node in threelvl1] == results3

  threelvl2 = I.iterNodesByMatching(zoneI, ["ZoneBC_t", CGL.BC_t.name, lambda n: I.getLabel(n) == 'IndexArray_t'])
  # print(f"threelvl2 = {[I.getName(node) for node in threelvl2]}")
  assert [I.getName(node) for node in threelvl2] == results3

  threelvl3 = I.iterNodesByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: I.getLabel(n) == 'IndexArray_t'])
  # print(f"threelvl3 = {[I.getName(node) for node in threelvl3]}")
  assert [I.getName(node) for node in threelvl3] == results3

  with pytest.raises(TypeError):
    list(I.iterNodesByMatching(zoneI, 12))

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

  # Equivalence with NodesWalkers
  # -----------------------------
  walker = I.NodesWalkers(zoneI, [''], search='dfs', depth=1, ancestors=True)
  assert list(walker()) == []
  walker.predicates = ['BC_t']
  assert list(walker()) == []
  walker.predicates = ['Index4_ii']
  assert list(walker()) == []

  walker.predicates = ['ZoneBC_t']
  assert [[I.getName(n) for n in nodes] for nodes in walker()] == [['ZBCA'], ['ZBCB']]
  walker.root       = zbcB
  walker.predicates = ['BC_t']
  assert [[I.getName(n) for n in nodes] for nodes in walker()] == [['bc3'], ['bc4'], ['bc5']]

  fpath = lambda nodes: '/'.join([I.get_name(n) for n in nodes])
  walker.root       = zoneI
  walker.predicates = ['ZoneBC_t', 'BC_t']
  assert [fpath(nodes) for nodes in walker()] == ['ZBCA/bc1', 'ZBCA/bc2', 'ZBCB/bc3', 'ZBCB/bc4', 'ZBCB/bc5']
  walker.predicates = ['ZoneBC_t', 'bc3']
  assert [fpath(nodes) for nodes in walker()] == ['ZBCB/bc3']

  walker.root       = zbcB
  walker.predicates = ['BC_t','IndexArray_t']
  iterator = walker()
  for bc in I.getNodesFromLabel1(zbcB, 'BC_t'):
    for idx in I.getNodesFromLabel1(bc, 'IndexArray_t'):
      assert next(iterator) == (bc, idx)

  walker.root       = zoneI
  walker.predicates = ['ZoneBC_t', 'BC_t','IndexArray_t']
  iterator = walker()
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for idx in I.getNodesFromLabel1(bc, 'IndexArray_t'):
        assert next(iterator) == (zbc, bc, idx)

  walker.predicates = ['ZoneBC_t', 'BC_t', 'PL*']
  iterator = walker()
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(iterator) == (zbc, bc, pl)

  p = re.compile('PL[12]')
  walker.predicates = [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: p.match(I.getName(n))]
  iterator = walker()
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in [n for n in I.getNodesFromName1(bc, 'PL*') if p.match(I.getName(n))]:
        assert next(iterator) == (zbc, bc, pl)

  with pytest.raises(TypeError):
    walker.predicates = 12

  # iterNodesWithParentsByMatching
  # ------------------------------
  assert list(I.iterNodesWithParentsByMatching(zoneI, '')) == []
  assert list(I.iterNodesWithParentsByMatching(zoneI, 'BC_t')) == []

  onelvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = I.iterNodesWithParentsByMatching(zbcB, 'BC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCA', 'bc1'), ('ZBCA', 'bc2'), ('ZBCB', 'bc3'), ('ZBCB', 'bc4'), ('ZBCB', 'bc5')]
  twolvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/bc3')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCB', 'bc3')]
  twolvl = I.iterNodesWithParentsByMatching(zbcB, 'BC_t/IndexArray_t')
  for bc in I.getNodesFromLabel1(zbcB, 'BC_t'):
    for idx in I.getNodesFromLabel1(bc, 'IndexArray_t'):
      assert next(twolvl) == (bc, idx)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for idx in I.getNodesFromLabel1(bc, 'IndexArray_t'):
        assert next(threelvl) == (zbc, bc, idx)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/PL*')
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, ['ZoneBC_t', 'BC_t', 'PL*'])
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, 'BC_t', 'PL*'])
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, 'PL*'])
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  p = re.compile('PL[12]')
  threelvl = I.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: p.match(I.getName(n))])
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in [n for n in I.getNodesFromName1(bc, 'PL*') if p.match(I.getName(n))]:
        assert next(threelvl) == (zbc, bc, pl)

  with pytest.raises(TypeError):
    list(I.iterNodesWithParentsByMatching(zoneI, 12))

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

  assert I.getSubregionExtent(I.getNodeFromName(zone, 'UnLinkedZSR'), zone) == 'UnLinkedZSR'
  assert I.getSubregionExtent(I.getNodeFromName(zone, 'BCLinkedZSR'), zone) == 'ZBC/BC2'
  assert I.getSubregionExtent(I.getNodeFromName(zone, 'GCLinkedZSR'), zone) == 'ZGC2/GC1to1B'

  with pytest.raises(ValueError):
    I.getSubregionExtent(I.getNodeFromName(zone, 'OrphelanZSR'), zone)
  with pytest.raises(I.CGNSLabelNotEqualError):
    I.getSubregionExtent(I.getNodeFromName(zone, 'WrongZSR'), zone)

def test_newDistribution():
  distri = I.newDistribution()
  assert I.getName(distri) == ':CGNS#Distribution'
  assert I.getLabel(distri) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  distri = I.newDistribution(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#Distribution') is not None

  zone = I.newZone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = I.newDistribution(distri_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = I.newZone('zone')
  distri_arrayA = {'Cell' : [0,15,30]}
  distri_arrayB = {'Vertex' : [100,1000,1000]}
  distri = I.newDistribution(distri_arrayA, parent=zone)
  distri = I.newDistribution(distri_arrayB, parent=zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(I.getNodesFromName(zone, ':CGNS#Distribution')) == 1

def test_newGlobalNumbering():
  gnum = I.newGlobalNumbering()
  assert I.getName(gnum) == ':CGNS#GlobalNumbering'
  assert I.getLabel(gnum) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  gnum = I.newGlobalNumbering(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#GlobalNumbering') is not None

  zone = I.newZone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = I.newGlobalNumbering(gnum_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None

def test_getDistribution():
  zone = I.newZone()
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = I.newDistribution(distri_arrays, zone)
  assert I.getDistribution(zone) is distri
  assert (I.getVal(I.getDistribution(zone, 'Cell')) == [0,15,30]).all()
  assert (I.getVal(I.getDistribution(zone, 'Vertex')) == [100,1000,1000]).all()

def test_getGlobalNumbering():
  zone = I.newZone()
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = I.newGlobalNumbering(gnum_arrays, zone)
  assert I.getGlobalNumbering(zone) is gnum_node
  assert (I.getVal(I.getGlobalNumbering(zone, 'Cell')) == [4,21,1,2,8,12]).all()
  assert  I.getVal(I.getGlobalNumbering(zone, 'Vertex')) == None

if __name__ == "__main__":
  # test_getNodeFromPredicate()
  # test_getNodesFromPredicate1()
  # test_requireNodeFromName()
  # test_requireNodeFromType()
  # test_getRequireNodeFromNameAndType()
  # test_NodesWalker()
  # test_NodesWalker_sort()
  # test_NodesWalker_apply()
  # test_NodesWalkers()
  # test_iterNodesFromPredicates()
  test_getNodesFromPredicates()
  # test_getNodesByMatching()
  # test_getNodesWithParentsByMatching()
  # test_getNodesWithParentsByMatching()
  # test_rmChildrenFromPredicate()
  # test_rmNodesromPredicate()
  # test_iterNodesFromPredicate()
  # test_getNodesFromPredicate()
  # test_getAllLabel()
  # test_getCGNSName()
