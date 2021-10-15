import pytest
import os
import numpy as np

import Converter.Internal as I

from maia.sids.cgns_keywords import Label as CGL
from maia.sids.pytree import predicate as P

from maia.utils        import parse_yaml_cgns

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_matches():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  assert P.match_name(I.getNodeFromName(tree, 'Index_iii'), 'Index_iii')
  assert P.match_name(I.getNodeFromName(tree, 'Index_iii'), 'Index_i*')
  assert not P.match_name(I.getNodeFromName(tree, 'Index_iv'), 'Index_iii')

  nface = I.getNodeFromName(tree, 'NFace')
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
  assert P.belongs_to_family(I.getNodeFromName(node, 'BC1'), 'SecondFamily')  == True
  assert P.belongs_to_family(I.getNodeFromName(node, 'BC3'), 'FirstFamily') == False
  assert P.belongs_to_family(I.getNodeFromName(node, 'BC4'), 'FirstFamily') == False
  assert P.belongs_to_family(I.getNodeFromName(node, 'BC4'), 'FirstFamily', allow_additional=True) == True

 
def test_auto_predicate():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)
  nface = I.getNodeFromName(tree, 'NFace')

  assert P.auto_predicate('NFace')(nface)
  assert P.auto_predicate('Elements_t')(nface)
  assert P.auto_predicate(CGL.Elements_t)(nface)
  assert not P.auto_predicate('Element_t')(nface)
  assert P.auto_predicate(np.array([23,0]))(nface)
  assert P.auto_predicate(lambda n : True)(nface)
  assert P.auto_predicate(lambda n : len(I.getName(n)) == 5)(nface)
  assert not P.auto_predicate(lambda n : False)(nface)

  with pytest.raises(TypeError):
    P.auto_predicate(123)

if __name__ == "__main__":
  test_matches()