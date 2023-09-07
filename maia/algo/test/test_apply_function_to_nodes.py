import pytest

import maia.pytree as PT

from maia.pytree.yaml                  import parse_yaml_cgns
from maia.algo.apply_function_to_nodes import apply_to_zones, zones_iterator

def test_apply_to_zones():

  def add_vtx(zone):
    zone[1][0][0] += 1

  yt = """
  BaseA CGNSBase_t:
    zoneI Zone_t I4 [[8,5,0]]:
      ZoneType ZoneType_t "Unstructured":
  BaseB CGNSBase_t:
    zoneII Zone_t I4 [[27,40,0]]:
      ZoneType ZoneType_t "Unstructured":
      ZoneBC ZoneBC_t:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  zoneI = PT.get_node_from_name(tree, 'zoneI')
  zoneII = PT.get_node_from_name(tree, 'zoneII')
  other_base = PT.get_node_from_name(tree, 'BaseB')

  # Single zone
  apply_to_zones(add_vtx, zoneI)
  assert PT.Zone.n_vtx(zoneI) == 9
  #Base
  apply_to_zones(add_vtx, other_base)
  assert PT.Zone.n_vtx(zoneII) == 28
  #Tree
  apply_to_zones(add_vtx, tree)
  assert PT.Zone.n_vtx(zoneI)  == 10
  assert PT.Zone.n_vtx(zoneII) == 29

  #BC (err)
  with pytest.raises(Exception):
    apply_to_zones(add_vtx, PT.get_node_from_label(tree, 'ZoneBC_t'))

  
def test_zones_iterator():
  yt = """
  BaseA CGNSBase_t:
    zoneI Zone_t:
  BaseB CGNSBase_t:
    zoneII Zone_t:
      ZoneBC ZoneBC_t:
  """

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  zone = PT.get_node_from_name(tree, 'zoneI')
  other_base = PT.get_node_from_name(tree, 'BaseB')

  # Single zone
  zone = PT.get_all_Zone_t(tree)[0]
  assert [z[0] for z in zones_iterator(zone)] == ['zoneI']
  assert [z[0] for z in zones_iterator(other_base)] == ['zoneII']
  assert [z[0] for z in zones_iterator(tree)] == ['zoneI', 'zoneII']

  #Other node (err)
  with pytest.raises(ValueError):
    for z in zones_iterator(PT.get_node_from_label(tree, 'ZoneBC_t')):
      pass

  
