import pytest
import numpy as np
import fnmatch

import Converter.Internal as I
from maia.pytree.cgns_keywords import Label as CGL

import maia.pytree as PT

from maia.utils.yaml   import parse_yaml_cgns

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

def names(nodes):
  return [PT.get_name(n) for n in nodes]

def test_generated_walkers():          
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  assert PT.get_node_from_name(tree, "ZoneI") == PT.get_node_from_predicate(tree, lambda n: PT.match_name(n, "ZoneI"))
  assert PT.get_node_from_value(tree, np.array([22,0])) == \
         PT.get_node_from_predicate(tree, lambda n: PT.match_value(n, np.array([22,0])))
  assert list(PT.iter_nodes_from_name(tree, "IndexArray_t")) == \
         list(PT.iter_nodes_from_predicate(tree, lambda n: PT.match_name(n, "IndexArray_t")))
  assert PT.get_nodes_from_name_and_label(tree, "Index_iii", "IndexArray_t") == \
         PT.get_nodes_from_predicate(tree, lambda n: PT.get_label(n) == "IndexArray_t" and PT.get_name(n) == "Index_iii")

  with pytest.raises(PT.CGNSNodeFromPredicateNotFoundError):
    PT.request_node_from_name(tree, "Zzz")

  assert PT.get_child_from_name(tree, "ZoneI") is None
  root = PT.get_node_from_name(tree, 'ZBCA')
  assert PT.get_children_from_label(root, CGL.BC_t.name) == PT.get_nodes_from_label(root, CGL.BC_t.name)

def test_generated_walkers_leg():
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  assert PT.getNodesFromType(tree, "IndexArray_t") == list(PT.iter_nodes_from_label(tree, "IndexArray_t"))
  assert PT.getNodesFromType2(tree, "IndexArray_t") == list(PT.iter_nodes_from_label(tree, "IndexArray_t", depth=2))
  assert PT.getNodesFromName1(tree, "Base") == list(PT.iter_children_from_name(tree, "Base"))


def test_generated_remove():
  treeA = parse_yaml_cgns.to_cgns_tree(yt)
  treeB = parse_yaml_cgns.to_cgns_tree(yt)

  PT.rm_nodes_from_predicate(treeA, lambda n: PT.match_name(n, "bc*"))
  PT.rm_nodes_from_name(treeB, "bc*")
  assert PT.is_same_tree(treeA, treeB)

def test_get_all_label():
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  assert names(PT.get_all_CGNSBase_t(tree)) == ['Base']
  assert names(PT.get_all_Zone_t(tree)) == ['ZoneI']
  assert names(PT.iter_all_BC_t(tree)) == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']




# Move in functionnal test ?
def test_getNodeFromPredicate():
  is_base    = lambda n: I.getName(n) == 'Base'    and I.getType(n) == 'CGNSBase_t'
  is_zonei   = lambda n: I.getName(n) == 'ZoneI'   and I.getType(n) == 'Zone_t'
  is_nface   = lambda n: I.getName(n) == 'NFace'   and I.getType(n) == 'Elements_t'
  is_ngon    = lambda n: I.getName(n) == 'Ngon'    and I.getType(n) == 'Elements_t'
  is_zbca    = lambda n: I.getName(n) == 'ZBCA'    and I.getType(n) == 'ZoneBC_t'
  is_bc1     = lambda n: I.getName(n) == 'bc1'     and I.getType(n) == 'BC_t'
  is_index_i = lambda n: I.getName(n) == 'Index_i' and I.getType(n) == 'IndexArray_t'

  tree = parse_yaml_cgns.to_cgns_tree(yt)

  # getNodeFrom...
  # ******************

  # getNodeFromPredicate
  # ========================
  # Camel case
  # ----------
  assert is_base  ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")                )
  assert is_zonei ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="bfs") )
  assert is_zonei ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs") )
  assert is_base  ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")                )
  assert is_zonei ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="bfs") )
  assert is_zonei ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs") )

  # Snake case
  # ----------
  assert PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=1) is None
  assert is_zonei(PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=2))
  assert is_zonei(PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=3))

  assert is_base    ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "Base", search="dfs", depth=1)    )
  assert is_zonei   ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs", depth=2)   )
  assert is_ngon    ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "Ngon", search="dfs", depth=3)    )
  assert is_bc1     ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "bc1", search="dfs", depth=4)     )
  assert is_index_i ( PT.getNodeFromPredicate(tree, lambda n: I.getName(n) == "Index_i", search="dfs", depth=5) )

  # getNodeFrom{Name, Label, ...}
  # =================================
  # Camel case
  # ----------
  assert is_nface( PT.getNodeFromName(tree, "NFace")                                                         )
  assert is_nface( PT.getNodeFromValue(tree, np.array([23,0], order='F'))                                    )
  assert is_ngon(  PT.getNodeFromLabel(tree, "Elements_t")                                                   )
  assert is_nface( PT.getNodeFromNameAndLabel(tree, "NFace", "Elements_t")                                   )
  assert PT.getNodeFromName(tree, "TOTO")                                                               is None
  assert PT.getNodeFromValue(tree, np.array([1230,0], order='F'))                                       is None
  assert PT.getNodeFromLabel(tree, "ZoneSubRegion_t")                                                   is None
  assert PT.getNodeFromNameAndLabel(tree, "TOTO", "Elements_t")                                         is None
  assert PT.getNodeFromNameAndLabel(tree, "NFace", "ZoneSubRegion_t")                                   is None

  # Snake case
  # ----------
  assert PT.get_node_from_name(tree, "TOTO")                                                                  is None
  assert PT.get_node_from_value(tree, np.array([1230,0], order='F'))                                          is None
  assert PT.get_node_from_label(tree, "ZoneSubRegion_t")                                                      is None
  assert PT.get_node_from_name_and_label(tree, "TOTO", "Elements_t")                                          is None
  assert PT.get_node_from_name_and_label(tree, "NFace", "ZoneSubRegion_t")                                    is None

  # getNodeFromPredicate{depth} and dfs
  # =======================================
  # Camel case
  assert PT.getNodeFromPredicate1(tree, lambda n: I.getName(n) == "ZoneI", search="dfs") is None
  assert is_zonei(PT.getNodeFromPredicate2(tree, lambda n: I.getName(n) == "ZoneI", search="dfs"))
  assert is_zonei(PT.getNodeFromPredicate3(tree, lambda n: I.getName(n) == "ZoneI", search="dfs"))

  # Snake case
  assert PT.get_child_from_predicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs") is None

  # getNodeFrom{Name, Label, ...}{depth}
  # ========================================
  # Camel case
  # ----------
  assert PT.getNodeFromName(tree, "ZoneI", search="dfs", depth=1) is None
  assert is_zonei(PT.getNodeFromName(tree, "ZoneI", search="dfs", depth=2))
  assert is_zonei(PT.getNodeFromName(tree, "ZoneI", search="dfs", depth=3))

  assert PT.getNodeFromName1(tree, "ZoneI") is None
  assert is_zonei(PT.getNodeFromName2(tree, "ZoneI"))
  assert is_zonei(PT.getNodeFromName3(tree, "ZoneI"))

  assert is_base    (PT.getNodeFromName(tree, "Base"   , depth=1))
  assert is_zonei   (PT.getNodeFromName(tree, "ZoneI"  , depth=2))
  assert is_ngon    (PT.getNodeFromName(tree, "Ngon"   , depth=3))
  assert is_bc1     (PT.getNodeFromName(tree, "bc1"    , depth=4))
  assert is_index_i (PT.getNodeFromName(tree, "Index_i", depth=5))

  assert is_base  (PT.getNodeFromLabel(tree, "CGNSBase_t", depth=1))
  assert is_zonei (PT.getNodeFromLabel(tree, "Zone_t"    , depth=2))
  assert is_ngon  (PT.getNodeFromLabel(tree, "Elements_t", depth=3))
  assert is_bc1   (PT.getNodeFromLabel(tree, "BC_t"      , depth=4))

  # Snake case
  # ----------
  assert PT.get_node_from_name(tree, "ZoneI", search="dfs", depth=1) is None
  assert is_zonei( PT.get_node_from_name(tree, "ZoneI", search="dfs", depth=2) )
  assert is_zonei( PT.get_node_from_name(tree, "ZoneI", search="dfs", depth=3) )

  assert PT.get_child_from_name(tree, "ZoneI") is None
  assert is_zonei( PT.get_node_from_name(tree, "ZoneI", depth=2) )
  assert is_zonei( PT.get_node_from_name(tree, "ZoneI", depth=3) )

  assert is_base    (PT.get_child_from_name(tree, "Base"   ,        ))
  assert is_zonei   (PT.get_node_from_name (tree, "ZoneI"  , depth=2))
  assert is_ngon    (PT.get_node_from_name (tree, "Ngon"   , depth=3))
  assert is_bc1     (PT.get_node_from_name (tree, "bc1"    , depth=4))
  assert is_index_i (PT.get_node_from_name (tree, "Index_i", depth=5))

  assert is_base    ( PT.get_node_from_label(tree, "CGNSBase_t"  , depth=1))
  assert is_zonei   ( PT.get_node_from_label(tree, "Zone_t"      , depth=2))
  assert is_ngon    ( PT.get_node_from_label(tree, "Elements_t"  , depth=3))
  assert is_bc1     ( PT.get_node_from_label(tree, "BC_t"        , depth=4))
  assert is_index_i ( PT.get_node_from_label(tree, "IndexArray_t", depth=5))

  # requestNodeFrom...
  # **************

  # requestNodeFromPredicate
  # ====================
  # Camel case
  # ----------
  assert is_base  ( PT.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "Base")                   )
  assert is_zonei ( PT.requestNodeFromPredicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs")    )
  assert is_base  ( PT.request_node_from_predicate(tree, lambda n: I.getName(n) == "Base")                )
  assert is_zonei ( PT.request_node_from_predicate(tree, lambda n: I.getName(n) == "ZoneI", search="dfs") )

  # Snake case
  # ----------
  assert is_nface ( PT.requestNodeFromName(tree, "NFace")                                                               )
  assert is_nface ( PT.requestNodeFromValue(tree, np.array([23,0], order='F'))                                          )
  assert is_ngon  ( PT.requestNodeFromLabel(tree, "Elements_t")                                                         )
  assert is_nface ( PT.requestNodeFromNameAndLabel(tree, "NFace", "Elements_t")                                         )
  predicate = lambda n: PT.predicate.match_value_label(n, np.array([23,0], dtype='int64',order='F'), "Elements_t")
  assert is_nface ( PT.requestNodeFromPredicate(tree, predicate)                                                        )
  assert is_nface ( PT.request_node_from_name(tree, "NFace")                                                            )
  assert is_nface ( PT.request_node_from_value(tree, np.array([23,0], order='F'))                                       )
  assert is_ngon  ( PT.request_node_from_label(tree, "Elements_t")                                                      )
  assert is_nface ( PT.request_node_from_name_and_label(tree, "NFace", "Elements_t")                                    )

  tree = parse_yaml_cgns.to_cgns_tree(yt)

  # Camel case
  # ----------
  # Test from Name
  base = I.getBases(tree)[0]
  nodes_from_name1 = ["ZoneI"]
  assert [I.getName(n) for n in PT.getNodesFromPredicate1(base, lambda n: I.getName(n) == "ZoneI")] == nodes_from_name1
  # Wildcard is not allowed in getNodesFromName1() from Cassiopee
  assert I.getNodesFromName1(base, "Zone*") == []
  assert [I.getName(n) for n in PT.getNodesFromPredicate1(base, lambda n: fnmatch.fnmatch(I.getName(n), "Zone*"))] == nodes_from_name1
  assert [I.getName(n) for n in PT.getNodesFromName1(base, "Zone*")] == nodes_from_name1
  assert [I.getName(n) for n in PT.getNodesFromName1(base, "Base")] == ["Base"]

  # Test from Type
  nodes_from_type1 = ["Zone_t"]
  assert [I.getType(n) for n in PT.getNodesFromPredicate1(base, lambda n: I.getType(n) == CGL.Zone_t.name)] == nodes_from_type1
  assert [I.getType(n) for n in PT.getNodesFromLabel1(base, CGL.Zone_t.name)] == nodes_from_type1

  # Test from Value
  zone = I.getNodeFromType(tree, 'Zone_t')
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [ngon_value[0]]
  assert [I.getVal(n)[0] for n in PT.getNodesFromPredicate1(zone, lambda n: I.getType(n) == CGL.Elements_t.name and np.array_equal(I.getVal(n), ngon_value))] == elements_from_type_value1
  assert [I.getVal(n)[0] for n in PT.getNodesFromValue1(zone, ngon_value)] == elements_from_type_value1

  zonebcs_from_type_name1 = ['ZBCB']
  assert [I.getName(n) for n in PT.getNodesFromPredicate1(zone, lambda n: I.getType(n) == CGL.ZoneBC_t.name and I.getName(n) != "ZBCA")] == zonebcs_from_type_name1

  # Snake case
  # ----------
  # Test from Name
  base = PT.requestNodeFromLabel(tree, 'CGNSBase_t') # get the first base
  nodes_from_name1 = ["ZoneI"]
  assert [I.getName(n) for n in PT.get_children_from_predicate(base, lambda n: I.getName(n) == "ZoneI")] == nodes_from_name1
  assert [I.getName(n) for n in PT.get_children_from_predicate(base, lambda n: fnmatch.fnmatch(I.getName(n), "Zone*"))] == nodes_from_name1
  assert [I.getName(n) for n in PT.get_children_from_name(base, "Zone*")] == nodes_from_name1
  assert [I.getName(n) for n in PT.get_children_from_name(base, "Base")] == []

  # Test from Type
  nodes_from_type1 = ["Zone_t"]
  assert [I.getType(n) for n in PT.get_children_from_predicate(base, lambda n: I.getType(n) == CGL.Zone_t.name)] == nodes_from_type1
  assert [I.getType(n) for n in PT.get_children_from_label(base, CGL.Zone_t.name)] == nodes_from_type1

  # Test from Value
  zone = I.getNodeFromType(tree, 'Zone_t')
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [ngon_value[0]]
  assert [I.getVal(n)[0] for n in PT.get_children_from_predicate(zone, lambda n: I.getType(n) == CGL.Elements_t.name and np.array_equal(I.getVal(n), ngon_value))] == elements_from_type_value1
  assert [I.getVal(n)[0] for n in PT.get_children_from_value(zone, ngon_value)] == elements_from_type_value1

  zonebcs_from_type_name1 = ['ZBCB']
  assert [I.getName(n) for n in PT.get_children_from_predicate(zone, lambda n: I.getType(n) == CGL.ZoneBC_t.name and I.getName(n) != "ZBCA")] == zonebcs_from_type_name1


