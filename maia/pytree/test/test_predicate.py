import pytest
import numpy as np

from maia.pytree.cgns_keywords import Label as CGL
from maia.pytree      import walk      as W
from maia.pytree      import predicate as P

from maia.pytree.yaml   import parse_yaml_cgns


def test_matches():
  nface = ['NFace', np.array([23, 0], np.int32), [], 'Elements_t']

  assert P.name_matches('NFace')(nface)
  assert P.name_matches('NFac*')(nface)
  assert not P.name_matches('NFacE')(nface)
  assert P.value_is(np.array([23,0]))(nface)
  assert P.label_matches('Elements_t')(nface)
  assert P.label_matches('Elemen*')(nface)
  assert P.label_matches(CGL.Elements_t)(nface)

  # Try composition
  assert (P.name_matches('NFace') & P.label_matches('Elements_t'))(nface)
  assert (P.name_matches('NFace') & P.label_matches(CGL.Elements_t))(nface)
  assert not (P.name_matches('NFAce') & P.label_matches('Elements'))(nface)
  assert not (P.name_matches('NFace') & P.label_matches('Elements'))(nface)
  assert not (P.name_matches('NFace') & P.label_matches(CGL.Zone_t))(nface)

  node = ['FamilyName', np.array([b'F', b'A', b'M', b'I', b'L', b'Y']), [], 'FamilyName_t']
  assert P.value_is('FAMILY')(node)

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
  assert P.__belongs_to_family(W.get_node_from_name(node, 'BC1'), 'SecondFamily')  == True
  assert P.__belongs_to_family(W.get_node_from_name(node, 'BC1'), '*Fam*')  == True
  assert P.__belongs_to_family(W.get_node_from_name(node, 'BC3'), 'FirstFamily') == False
  assert P.__belongs_to_family(W.get_node_from_name(node, 'BC4'), 'FirstFamily') == False
  assert P.__belongs_to_family(W.get_node_from_name(node, 'BC4'), 'FirstFamily', allow_additional=True) == True
  assert P.__belongs_to_family(W.get_node_from_path(node, 'BC1/FamilyName'), 'SecondFamily') == False

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
  assert [n[0] for n in W.get_nodes_from_predicate(node, P.is_bc_of_loc('Vertex'))]==['BC1','BC2']
  assert W.get_node_from_predicate(node, P.is_bc_of_loc('EdgeCenter'))[0] == 'BC3'
  assert W.get_node_from_predicate(node, P.is_bc_of_loc('FaceCenter')) is None

def test_is_elmt_of_type():
  yt = """
  Zone Zone_t:
    NGON   Elements_t I4 [22, 0]:
    TRI1   Elements_t I4 [ 5, 0]:
    NODE   Elements_t I4 [ 2, 0]:
    TRI2   Elements_t I4 [ 5, 0]:
    BAR    Elements_t I4 [ 4, 0]:
    ZoneBC ZoneBC_t:
  """
  node = parse_yaml_cgns.to_node(yt)
  assert W.get_node_from_predicate (node, P.is_elmt_of_type('TETRA_4')) is None
  assert W.get_node_from_predicate (node, P.is_elmt_of_type('ZoneBC' )) is None
  assert [n[0] for n in W.get_nodes_from_predicate(node, P.is_elmt_of_type('TRI_3'))]==['TRI1','TRI2']
  assert W.get_node_from_predicate(node, P.is_elmt_of_type('NODE'))[0]=='NODE'
