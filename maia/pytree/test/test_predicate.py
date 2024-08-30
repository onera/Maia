import pytest
import numpy as np

from maia.pytree.cgns_keywords import Label as CGL
from maia.pytree      import walk      as W
from maia.pytree      import predicate as P

from maia.pytree.yaml   import parse_yaml_cgns

def partial_funcs_equal(f1, f2):
  return all([getattr(f1, attr) == getattr(f2, attr) for attr in ['func', 'args', 'keywords']])

def test_matches():
  nface = ['NFace', np.array([23, 0], np.int32), [], 'Elements_t']

  assert P.match_name(nface, 'NFace')
  assert P.match_name(nface, 'NFac*')
  assert not P.match_name(nface, 'NFacE')
  assert P.match_value(nface, np.array([23,0]))
  assert P.match_str_label(nface, 'Elements_t')
  assert P.match_cgk_label(nface, CGL.Elements_t)
  assert P.match_label(nface, 'Elements_t')
  assert P.match_label(nface, 'Elemen*')
  assert P.match_label(nface, CGL.Elements_t)

  assert P.match_name_value(nface, 'NFace', np.array([23,0]))
  assert not P.match_name_value(nface, 'NFAce', np.array([23,0]))
  assert P.match_name_label(nface, 'NFace', 'Elements_t')
  assert P.match_name_label(nface, 'NFace', CGL.Elements_t)
  assert not P.match_name_label(nface, 'NFAce', 'Elements')
  assert not P.match_name_label(nface, 'NFace', 'Elements')
  assert not P.match_name_label(nface, 'NFace', CGL.Zone_t)
  assert not P.match_value_label(nface, np.array([22,0]), 'Elements_t')
  assert not P.match_value_label(nface, np.array([23,0]), 'Elements')
  assert not P.match_value_label(nface, np.array([23,0]), CGL.Zone_t)
  assert P.match_name_value_label(nface, 'NFace', np.array([23,0]), 'Elements_t')
  assert P.match_name_value_label(nface, 'NFace', np.array([23,0]), CGL.Elements_t)

  node = ['FamilyName', np.array([b'F', b'A', b'M', b'I', b'L', b'Y']), [], 'FamilyName_t']
  assert P.match_value(node, 'FAMILY')

def test_belongs_to_family():
  yt = """
ZoneBC ZoneBC_t:
  BC1 BC_t:
    FamilyName FamilyName_t "SecondFamily":
  BC3 BC_t:
    SubBC BC_t:
      FamilyName FamilyName_t "FirstFamily":
  BC4 BC_t:
    FamilyName FamilyName_t "SecondFamily":
    AdditionalFamilyName1 AdditionalFamilyName_t "ThirdFamily":
    AdditionalFamilyName2 AdditionalFamilyName_t "FirstFamily":
"""
  node = parse_yaml_cgns.to_node(yt)
  assert P.belongs_to_family(W.get_node_from_name(node, 'BC1'), 'SecondFamily')  == True
  assert P.belongs_to_family(W.get_node_from_name(node, 'BC3'), 'FirstFamily') == False
  assert P.belongs_to_family(W.get_node_from_name(node, 'BC4'), 'FirstFamily') == False
  assert P.belongs_to_family(W.get_node_from_name(node, 'BC4'), 'FirstFamily', allow_additional=True) == True
  assert P.belongs_to_family(W.get_node_from_path(node, 'BC1/FamilyName'), 'SecondFamily') == False

def test_is_bc_of_loc():
  yt = """
  ZoneBC ZoneBC_t:
    BC1 BC_t:
      GridLocation GridLocation_t "Vertex":
    BC2 BC_t:
    BC3 BC_t:
      GridLocation GridLocation_t "EdgeCenter":
  """
  node = parse_yaml_cgns.to_node(yt)
  assert P.is_bc_of_loc(W.get_node_from_name(node, 'BC1'), 'Vertex'    )  == True
  assert P.is_bc_of_loc(W.get_node_from_name(node, 'BC2'), 'Vertex'    )  == True
  assert P.is_bc_of_loc(W.get_node_from_name(node, 'BC2'), 'EdgeCenter')  == False
  assert P.is_bc_of_loc(W.get_node_from_name(node, 'BC3'), 'FaceCenter')  == False

def test_is_elmt_of_type():
  yt = """
  Zone Zone_t:
    NGON   Elements_t I4 [22, 0]:
    TRI    Elements_t I4 [ 5, 0]:
    NFACE  Elements_t I4 [23, 0]:
    NODE   Elements_t I4 [ 2, 0]:
    TETRA  Elements_t I4 [11, 0]:
    BAR    Elements_t I4 [ 4, 0]:
    ZoneBC ZoneBC_t:
  """
  node = parse_yaml_cgns.to_node(yt)
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'ZoneBC'),                           )  == False
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'TRI'   ),                           )  == True
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'NODE'  ),                      dim=0)  == True
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'NODE'  ),                      dim=2)  == False
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'NGON'  ), cgns_name='NGON_n'        )  == True
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'TETRA' ), cgns_name='TRI_3'         )  == False
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'NFACE' ), cgns_name='NFACE_n', dim=3)  == True
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'BAR'   ), cgns_name='NGON_n' , dim=1)  == False
  assert P.is_elmt_of_type(W.get_node_from_name(node, 'BAR'   ), cgns_name='BAR_2'  , dim=2)  == False

def test_auto_predicate():
  nface = ['NFace', np.array([23, 0], np.int32), [], 'Elements_t']

  assert P.auto_predicate('NFace')(nface)
  assert P.auto_predicate('Elements_t')(nface)
  assert P.auto_predicate(CGL.Elements_t)(nface)
  assert not P.auto_predicate('Element_t')(nface)
  assert P.auto_predicate(np.array([23,0]))(nface)
  assert P.auto_predicate(lambda n : True)(nface)
  assert P.auto_predicate(lambda n : len(n[0]) == 5)(nface)
  assert not P.auto_predicate(lambda n : False)(nface)

  with pytest.raises(TypeError):
    P.auto_predicate(123)

def test_auto_predicates():
  auto_predicates = P.auto_predicates(['Base', 'Zone'])
  assert partial_funcs_equal(auto_predicates[0], P.auto_predicate('Base'))
  assert partial_funcs_equal(auto_predicates[1], P.auto_predicate('Zone'))
  auto_predicates = P.auto_predicates('ZoneBC_t/BC_t')
  assert partial_funcs_equal(auto_predicates[0], P.auto_predicate('ZoneBC_t'))
  assert partial_funcs_equal(auto_predicates[1], P.auto_predicate('BC_t'))

  with pytest.raises(TypeError):
    P.auto_predicates(123)
