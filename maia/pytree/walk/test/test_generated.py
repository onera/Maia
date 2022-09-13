import pytest
import numpy as np
import fnmatch

from maia.pytree.cgns_keywords import Label as CGL

import maia.pytree as PT

from maia.utils.yaml   import parse_yaml_cgns

yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZGCA ZoneGridConnectivity_t:
      gc1 GridConnectivity_t:
        Index_i IndexArray_t:
      gc2 GridConnectivity_t:
        Index_ii IndexArray_t:
    ZGCB ZoneGridConnectivity_t:
      gc3 GridConnectivity_t:
        Index_iii IndexArray_t:
      gc4 GridConnectivity_t:
      gc5 GridConnectivity_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""

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
  root = PT.get_node_from_name(tree, 'ZGCA')
  assert PT.get_children_from_label(root, CGL.GridConnectivity_t.name) == PT.get_nodes_from_label(root, CGL.GridConnectivity_t.name)

def test_generated_walkers_leg():
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  assert PT.getNodesFromType(tree, "IndexArray_t") == list(PT.iter_nodes_from_label(tree, "IndexArray_t"))
  assert PT.getNodesFromType2(tree, "IndexArray_t") == list(PT.iter_nodes_from_label(tree, "IndexArray_t", depth=2))
  assert PT.getNodesFromName1(tree, "Base") == list(PT.iter_children_from_name(tree, "Base"))


def test_generated_remove():
  treeA = parse_yaml_cgns.to_cgns_tree(yt)
  treeB = parse_yaml_cgns.to_cgns_tree(yt)

  PT.rm_nodes_from_predicate(treeA, lambda n: PT.match_name(n, "gc*"))
  PT.rm_nodes_from_name(treeB, "gc*")
  assert PT.is_same_tree(treeA, treeB)

def test_get_all_label():
  tree = parse_yaml_cgns.to_cgns_tree(yt)

  assert PT.get_names(PT.get_all_CGNSBase_t(tree)) == ['Base']
  assert PT.get_names(PT.get_all_Zone_t(tree)) == ['ZoneI']
  assert PT.get_names(PT.iter_all_BC_t(tree)) == []


def test_from_path():
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  assert PT.get_node_from_path(tree, 'Base/ZoneI/ZGCB/gc3') == PT.get_node_from_name(tree, 'gc3')
  assert PT.get_node_from_path(tree, 'Base/Zone/ZGCB/gc3') is None
  #With ancestors
  nodes = PT.get_node_from_path(tree, 'Base/ZoneI/ZGCB/gc3', ancestors=True)
  assert [PT.get_label(n) for n in nodes] == ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', 'GridConnectivity_t']
  assert PT.get_node_from_path(tree, 'Base/Zone/ZGCB/gc3', ancestors=True) == []

# Move in functionnal test ?
def test_getNodeFromPredicate():
  is_base    = lambda n: PT.get_name(n) == 'Base'    and PT.get_label(n) == 'CGNSBase_t'
  is_zonei   = lambda n: PT.get_name(n) == 'ZoneI'   and PT.get_label(n) == 'Zone_t'
  is_nface   = lambda n: PT.get_name(n) == 'NFace'   and PT.get_label(n) == 'Elements_t'
  is_ngon    = lambda n: PT.get_name(n) == 'Ngon'    and PT.get_label(n) == 'Elements_t'
  is_zgca    = lambda n: PT.get_name(n) == 'ZGCA'    and PT.get_label(n) == 'ZoneGridConnectivity_t'
  is_gc1     = lambda n: PT.get_name(n) == 'gc1'     and PT.get_label(n) == 'GridConnectivity_t'
  is_index_i = lambda n: PT.get_name(n) == 'Index_i' and PT.get_label(n) == 'IndexArray_t'

  tree = parse_yaml_cgns.to_cgns_tree(yt)

  # getNodeFrom...
  # ******************

  # getNodeFromPredicate
  # ========================
  # Camel case
  # ----------
  assert is_base  ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "Base")                )
  assert is_zonei ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="bfs") )
  assert is_zonei ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs") )
  assert is_base  ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "Base")                )
  assert is_zonei ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="bfs") )
  assert is_zonei ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs") )

  # Snake case
  # ----------
  assert PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs", depth=1) is None
  assert is_zonei(PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs", depth=2))
  assert is_zonei(PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs", depth=3))

  assert is_base    ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "Base", search="dfs", depth=1)    )
  assert is_zonei   ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs", depth=2)   )
  assert is_ngon    ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "Ngon", search="dfs", depth=3)    )
  assert is_gc1     ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "gc1", search="dfs", depth=4)     )
  assert is_index_i ( PT.getNodeFromPredicate(tree, lambda n: PT.get_name(n) == "Index_i", search="dfs", depth=5) )

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
  assert PT.getNodeFromPredicate1(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs") is None
  assert is_zonei(PT.getNodeFromPredicate2(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs"))
  assert is_zonei(PT.getNodeFromPredicate3(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs"))

  # Snake case
  assert PT.get_child_from_predicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs") is None

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
  assert is_gc1     (PT.getNodeFromName(tree, "gc1"    , depth=4))
  assert is_index_i (PT.getNodeFromName(tree, "Index_i", depth=5))

  assert is_base  (PT.getNodeFromLabel(tree, "CGNSBase_t", depth=1))
  assert is_zonei (PT.getNodeFromLabel(tree, "Zone_t"    , depth=2))
  assert is_ngon  (PT.getNodeFromLabel(tree, "Elements_t", depth=3))
  assert is_gc1   (PT.getNodeFromLabel(tree, "GridConnectivity_t"      , depth=4))

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
  assert is_gc1     (PT.get_node_from_name (tree, "gc1"    , depth=4))
  assert is_index_i (PT.get_node_from_name (tree, "Index_i", depth=5))

  assert is_base    ( PT.get_node_from_label(tree, "CGNSBase_t"  , depth=1))
  assert is_zonei   ( PT.get_node_from_label(tree, "Zone_t"      , depth=2))
  assert is_ngon    ( PT.get_node_from_label(tree, "Elements_t"  , depth=3))
  assert is_gc1     ( PT.get_node_from_label(tree, "GridConnectivity_t"        , depth=4))
  assert is_index_i ( PT.get_node_from_label(tree, "IndexArray_t", depth=5))

  # requestNodeFrom...
  # **************

  # requestNodeFromPredicate
  # ====================
  # Camel case
  # ----------
  assert is_base  ( PT.requestNodeFromPredicate(tree, lambda n: PT.get_name(n) == "Base")                   )
  assert is_zonei ( PT.requestNodeFromPredicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs")    )
  assert is_base  ( PT.request_node_from_predicate(tree, lambda n: PT.get_name(n) == "Base")                )
  assert is_zonei ( PT.request_node_from_predicate(tree, lambda n: PT.get_name(n) == "ZoneI", search="dfs") )

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
  base = PT.get_child_from_label(tree, 'CGNSBase_t')
  nodes_from_name1 = ["ZoneI"]
  assert [PT.get_name(n) for n in PT.getNodesFromPredicate1(base, lambda n: PT.get_name(n) == "ZoneI")] == nodes_from_name1
  assert [PT.get_name(n) for n in PT.getNodesFromPredicate1(base, lambda n: fnmatch.fnmatch(PT.get_name(n), "Zone*"))] == nodes_from_name1
  assert [PT.get_name(n) for n in PT.getNodesFromName1(base, "Zone*")] == nodes_from_name1
  assert [PT.get_name(n) for n in PT.getNodesFromName1(base, "Base")] == ["Base"]

  # Test from Type
  nodes_from_type1 = ["Zone_t"]
  assert [PT.get_label(n) for n in PT.getNodesFromPredicate1(base, lambda n: PT.get_label(n) == CGL.Zone_t.name)] == nodes_from_type1
  assert [PT.get_label(n) for n in PT.getNodesFromLabel1(base, CGL.Zone_t.name)] == nodes_from_type1

  # Test from Value
  zone = PT.get_node_from_label(tree, 'Zone_t')
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [ngon_value[0]]
  assert [PT.get_value(n)[0] for n in PT.getNodesFromPredicate1(zone, lambda n: PT.get_label(n) == CGL.Elements_t.name and np.array_equal(PT.get_value(n), ngon_value))] == elements_from_type_value1
  assert [PT.get_value(n)[0] for n in PT.getNodesFromValue1(zone, ngon_value)] == elements_from_type_value1

  zonegcs_from_type_name1 = ['ZGCB']
  assert [PT.get_name(n) for n in PT.getNodesFromPredicate1(zone, lambda n: PT.get_label(n) == CGL.ZoneGridConnectivity_t.name and PT.get_name(n) != "ZGCA")] == zonegcs_from_type_name1

  # Snake case
  # ----------
  # Test from Name
  base = PT.requestNodeFromLabel(tree, 'CGNSBase_t') # get the first base
  nodes_from_name1 = ["ZoneI"]
  assert [PT.get_name(n) for n in PT.get_children_from_predicate(base, lambda n: PT.get_name(n) == "ZoneI")] == nodes_from_name1
  assert [PT.get_name(n) for n in PT.get_children_from_predicate(base, lambda n: fnmatch.fnmatch(PT.get_name(n), "Zone*"))] == nodes_from_name1
  assert [PT.get_name(n) for n in PT.get_children_from_name(base, "Zone*")] == nodes_from_name1
  assert [PT.get_name(n) for n in PT.get_children_from_name(base, "Base")] == []

  # Test from Type
  nodes_from_type1 = ["Zone_t"]
  assert [PT.get_label(n) for n in PT.get_children_from_predicate(base, lambda n: PT.get_label(n) == CGL.Zone_t.name)] == nodes_from_type1
  assert [PT.get_label(n) for n in PT.get_children_from_label(base, CGL.Zone_t.name)] == nodes_from_type1

  # Test from Value
  zone = PT.get_node_from_label(tree, 'Zone_t')
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [ngon_value[0]]
  assert [PT.get_value(n)[0] for n in PT.get_children_from_predicate(zone, lambda n: PT.get_label(n) == CGL.Elements_t.name and np.array_equal(PT.get_value(n), ngon_value))] == elements_from_type_value1
  assert [PT.get_value(n)[0] for n in PT.get_children_from_value(zone, ngon_value)] == elements_from_type_value1

  zonegcs_from_type_name1 = ['ZGCB']
  assert [PT.get_name(n) for n in PT.get_children_from_predicate(zone, lambda n: PT.get_label(n) == CGL.ZoneGridConnectivity_t.name and PT.get_name(n) != "ZGCA")] == zonegcs_from_type_name1


