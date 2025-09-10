import pytest
import numpy as np

import maia.pytree as PT
from maia.pytree.cgns_keywords import Label as CGL
from maia.pytree      import pred      as P

from maia.pytree.yaml   import parse_yaml_cgns

def test_escape_str():
  assert P._escape_str('flux') == 'flux'
  assert P._escape_str('flux*') == 'flux*'
  assert P._escape_str('flux[RO]') == 'flux[[]RO[]]'
  assert P._escape_str('flux[[RO]]') == 'flux[[][[]RO[]][]]'
  assert P._escape_str('flux[R*]') == 'flux[[]R*[]]'

def test_tt():
  import maia
  import maia.pytree as PT
  from mpi4py import MPI
  tree = maia.factory.generate_dist_block(21, 'Poly', MPI.COMM_SELF)
  z = PT.get_all_Zone_t(tree)[0]
  PT.set_name(z, 'zone[12]')
  assert PT.get_node_from_name(tree, 'zone[12]') is not None
  assert PT.get_node_from_name(tree, 'zone[1*]') is not None

def test_matches():
  nface = ['NFace', np.array([23, 0], np.int32), [], 'Elements_t']

  assert P.name_matches('NFace')(nface)
  assert P.name_matches('NFac*')(nface)
  assert not P.name_matches('NFacE')(nface)
  assert P.value_is(np.array([23,0]))(nface)
  assert P.label_matches('Elements_t')(nface)
  assert P.label_matches('Elemen*')(nface)
  assert P.label_is(CGL.Elements_t)(nface)

  # Try composition
  assert (P.name_matches('NFace') & P.label_matches('Elements_t'))(nface)
  assert (P.name_matches('NFace') & P.label_is(CGL.Elements_t))(nface)
  assert not (P.name_matches('NFAce') & P.label_matches('Elements'))(nface)
  assert not (P.name_matches('NFace') & P.label_matches('Elements'))(nface)
  assert not (P.name_matches('NFace') & P.label_is(CGL.Zone_t))(nface)

  node = ['FamilyName', np.array([b'F', b'A', b'M', b'I', b'L', b'Y']), [], 'FamilyName_t']
  assert P.value_is('FAMILY')(node)

def test_combination():

  # Check lazy evaluation of predicates with a counter
  def _name_is(node, name):
    nonlocal cnt
    cnt += 1
    return PT.get_name(node) == name
  def _label_is(node, label):
    nonlocal cnt
    cnt +=1
    return PT.get_label(node) == label
  def name_is(name):
    return P.NodePredicate(lambda X: _name_is(X, name))
  def label_is(label):
    return P.NodePredicate(lambda X: _label_is(X, label))

  root = PT.new_node('Zone', 'Zone_t')
  tri  = PT.new_Elements('TRI', 'TRI_3', parent=root)
  quad = PT.new_Elements('QUAD', 'QUAD_4', parent=root)
  hexa = PT.new_Elements('HEXA', 'HEXA_8', parent=root)

  cnt = 0
  assert name_is('QUAD')(hexa) == False

  cnt = 0
  assert (name_is('QUAD') | label_is('Elements_t'))(hexa) == True
  assert cnt == 2 # Both funcs are evaluated

  cnt = 0
  assert (label_is('Elements_t') | name_is('QUAD'))(hexa) == True
  assert cnt == 1 # First funcs return True -> only one eval

  cnt = 0
  assert P.all([label_is('Elements_t'), name_is('QUAD'), name_is('HEXA')])(hexa) == False
  assert cnt == 2 
  assert P.all([label_is('Elements_t'), name_is('QUAD') | name_is('HEXA')])(hexa) == True
  cnt = 0
  assert P.any([label_is('Elements_t'), name_is('QUAD'), name_is('HEXA')])(hexa) == True
  assert cnt == 1

  assert P.all([])(hexa) == True
  assert P.any([])(hexa) == False

  assert len(PT.get_nodes_from_predicate(root, 
      P.all([label_is('Elements_t'), name_is('QUAD') | name_is('HEXA')]))) == 2

def test_has_child():
  yt = """
  ZoneBC ZoneBC_t:
    BC1 BC_t:
    BC2 BC_t:
      GridLocation GridLocation_t "IEdgeCenter":
  """
  root = parse_yaml_cgns.to_node(yt)
  assert [PT.get_name(node) for node in 
          PT.iter_nodes_from_predicate(root, P.has_child_of_name('GridLocation'))] == ['BC2']
  assert [PT.get_name(node) for node in 
          PT.iter_nodes_from_predicate(root, P.has_child_of_name('*GridLocation'))] == []
  assert [PT.get_name(node) for node in 
          PT.iter_nodes_from_predicate(root, P.has_child_of_label('BC_t'))] == ['ZoneBC']


def test_has_loc():
  yt = """
  ZoneBC ZoneBC_t:
    BC1 BC_t:
    BC2 BC_t:
      GridLocation GridLocation_t "IEdgeCenter":
    BC3 BC_t:
      GridLocation GridLocation_t "JEdgeCenter":
    Descriptor Descriptor_t:
      GridLocation GridLocation_t "CellCenter":
    
  """
  root = parse_yaml_cgns.to_node(yt)
  assert [PT.get_name(node) for node in 
          PT.iter_nodes_from_predicate(root, P.has_location('JEdgeCenter'))] == ['BC3']
  assert [PT.get_name(node) for node in 
          PT.iter_nodes_from_predicate(root, P.has_location('*EdgeCenter'))] == ['BC2', 'BC3']
  assert [PT.get_name(node) for node in 
          PT.iter_nodes_from_predicate(root, P.has_location('Vertex'))] == ['BC1']
  # Descriptor node is not returned, because its label does not allow GridLocation
  assert [PT.get_name(node) for node in 
          PT.iter_nodes_from_predicate(root, P.has_location('CellCenter'))] == []


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
  assert P.__belongs_to_family(PT.find_node_from_name(node, 'BC1'), 'SecondFamily')  == True
  assert P.__belongs_to_family(PT.find_node_from_name(node, 'BC1'), '*Fam*')  == True
  assert P.__belongs_to_family(PT.find_node_from_name(node, 'BC3'), 'FirstFamily') == False
  assert P.__belongs_to_family(PT.find_node_from_name(node, 'BC4'), 'FirstFamily') == True
  assert P.__belongs_to_family(PT.find_node_from_name(node, 'BC4'), 'FirstFamily', allow_additional=False) == False
  assert P.__belongs_to_family(PT.find_node_from_path(node, 'BC1/FamilyName'), 'SecondFamily') == False

def test_is_bc_of_location():
  yt = """
  ZoneBC ZoneBC_t:
    BC1 BC_t:
      GridLocation GridLocation_t "Vertex":
    BC2 BC_t:
    BC3 BC_t:
      GridLocation GridLocation_t "EdgeCenter":
  """
  node = parse_yaml_cgns.to_node(yt)
  assert [n[0] for n in PT.get_nodes_from_predicate(node, P.is_bc_of_location('Vertex'))]==['BC1','BC2']
  assert PT.find_node_from_predicate(node, P.is_bc_of_location('EdgeCenter'))[0] == 'BC3'
  assert PT.get_node_from_predicate(node, P.is_bc_of_location('FaceCenter')) is None


def test_is_element_of_type():
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
  assert PT.get_node_from_predicate (node, P.is_element_of_type('TETRA_4')) is None
  assert PT.get_node_from_predicate (node, P.is_element_of_type('ZoneBC' )) is None
  assert [n[0] for n in PT.get_nodes_from_predicate(node, P.is_element_of_type('TRI_3'))]==['TRI1','TRI2']
  assert PT.find_node_from_predicate(node, P.is_element_of_type('NODE'))[0]=='NODE'

def test_is_zone_of_kind():
  struct1d = PT.new_Zone('Struct1D', type='Structured', size=[[11,10,0]])
  struct2d = PT.new_Zone('Struct2D', type='Structured', size=[[11,10,0], [6,5,0]])
  struct3d = PT.new_Zone('Struct3D', type='Structured', size=[[11,10,0], [6,5,0], [2,1,0]])
  elt1d = PT.new_Zone('Elt1D', type='Unstructured')
  PT.new_Elements('BAR', 'BAR_2', erange=[1,10], parent=elt1d)
  elt2d = PT.new_Zone('Elt2D', type='Unstructured')
  PT.new_Elements('BAR', 'BAR_2', erange=[1,10], parent=elt2d)
  PT.new_Elements('TRI', 'TRI_3', erange=[11,15], parent=elt2d)
  elt3d = PT.new_Zone('Elt3D', type='Unstructured')
  PT.new_Elements('TRI', 'TRI_3', erange=[1,15], parent=elt3d)
  PT.new_Elements('TETRA', 'TETRA_4', erange=[16,20], parent=elt3d)
  poly2d = PT.yaml.to_node("""
  Poly2D Zone_t:
    ZoneType ZoneType_t "Unstructured":
    Edges Elements_t [3,0]:
    Faces Elements_t [22,0]:
  """)
  poly2d2 = PT.yaml.to_node("""
  Poly2DPE Zone_t:
    ZoneType ZoneType_t "Unstructured":
    Edges Elements_t [3,0]:
      ParentElements DataArray_t:
  """)
  poly3d = PT.yaml.to_node("""
  Poly3D Zone_t:
    ZoneType ZoneType_t "Unstructured":
    NG Elements_t [22,0]:
    NF Elements_t [23,0]:
  """)
  poly3d2 = PT.yaml.to_node("""
  Poly3DPE Zone_t:
    ZoneType ZoneType_t "Unstructured":
    NG Elements_t [22,0]:
      ParentElements DataArray_t:
  """)

  base = PT.new_node('Base', 'CGNSBase_t', children=[struct1d, struct2d, struct3d,
      elt1d, elt2d, elt3d, poly2d, poly2d2, poly3d, poly3d2])

  assert PT.find_node_from_predicate(base, P.is_zone_of_kind('S', 2))[0] == 'Struct2D'
  assert PT.find_node_from_predicate(base, P.is_zone_of_kind('Elt', 3))[0] == 'Elt3D'
  assert [PT.get_name(n) for n in PT.get_nodes_from_predicate(base, P.is_zone_of_kind('S'))] \
    == ['Struct1D', 'Struct2D', 'Struct3D']
  assert [PT.get_name(n) for n in PT.get_nodes_from_predicate(base, P.is_zone_of_kind('Poly', 2))] \
    == ['Poly2D', 'Poly2DPE']
  assert [PT.get_name(n) for n in PT.get_nodes_from_predicate(base, P.is_zone_of_kind('Poly'))] \
    == ['Poly2D', 'Poly2DPE', 'Poly3D', 'Poly3DPE']
  assert [PT.get_name(n) for n in PT.get_nodes_from_predicate(base, P.is_zone_of_kind(None, 2))] \
    == ['Struct2D', 'Elt2D', 'Poly2D', 'Poly2DPE']
  assert [PT.get_name(n) for n in PT.get_nodes_from_predicate(base, P.is_zone_of_kind('U', 3))] \
    == ['Elt3D', 'Poly3D', 'Poly3DPE']
  assert len(PT.get_nodes_from_predicate(base, P.is_zone_of_kind(None, None))) == 10


def test_is_gc_of_kind():
  bc = PT.new_BC('BC')
  gc_s = PT.new_GridConnectivity1to1('SMatch')
  gc_s_per = PT.new_GridConnectivity1to1('SMatchPerio')
  PT.new_GridConnectivityProperty(periodic={'translation' : [1,0,0]}, parent=gc_s_per)

  gc_abb = PT.new_GridConnectivity('UNoMatch', type='Abutting') 
  gc_abb_per = PT.new_GridConnectivity('UNoMatchPerio', type='Abutting') 
  PT.new_GridConnectivityProperty(periodic={'translation' : [1,0,0]}, parent=gc_abb_per)

  gc_u = PT.new_GridConnectivity('UMatch', type='Abutting1to1') 
  gc_u_per = PT.new_GridConnectivity('UMatchPerio', type='Abutting1to1') 
  PT.new_GridConnectivityProperty(periodic={'translation' : [1,0,0]}, parent=gc_u_per)
  
  zgc = PT.new_node('ZoneGridConnectivity', 'ZoneGridConnectivity_t', children=
                    [bc, gc_s, gc_s_per, gc_abb, gc_abb_per, gc_u, gc_u_per])

  assert PT.find_node_from_predicate(zgc, P.is_gc_of_kind(False, True))[0] == 'UNoMatchPerio'
  assert [PT.get_name(n) for n in PT.get_nodes_from_predicate(zgc, P.is_gc_of_kind(is_perio=True))] \
    == ['SMatchPerio', 'UNoMatchPerio', 'UMatchPerio']
  assert [PT.get_name(n) for n in PT.get_nodes_from_predicate(zgc, P.is_gc_of_kind(True, True))] \
    == ['SMatchPerio', 'UMatchPerio']
  assert len(PT.get_nodes_from_predicate(zgc, P.is_gc_of_kind())) == 6
