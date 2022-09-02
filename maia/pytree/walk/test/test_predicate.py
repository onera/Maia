import pytest
import numpy as np

from maia.pytree.cgns_keywords import Label as CGL
from maia.pytree      import walk      as W
from maia.pytree.walk import predicate as P

from maia.utils.yaml   import parse_yaml_cgns

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

def test_belongs_to_family():
  yt = """
ZBC ZoneBC_t:
  BC1 BC_t:
    FamilyName FamilyName_t "SecondFamily":
  BC3 BC_t:
    SubBC BC_t:
      FamilyName FamilyName_t "FirstFamily":
  BC4 BC_t:
    FamilyName FamilyName_t "SecondFamily":
    AdditionalFamilyName AdditionalFamilyName_t "ThirdFamily":
    AdditionalFamilyName AdditionalFamilyName_t "FirstFamily":
"""
  node = parse_yaml_cgns.to_node(yt)
  assert P.belongs_to_family(W.get_node_from_name(node, 'BC1'), 'SecondFamily')  == True
  assert P.belongs_to_family(W.get_node_from_name(node, 'BC3'), 'FirstFamily') == False
  assert P.belongs_to_family(W.get_node_from_name(node, 'BC4'), 'FirstFamily') == False
  assert P.belongs_to_family(W.get_node_from_name(node, 'BC4'), 'FirstFamily', allow_additional=True) == True

 
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
