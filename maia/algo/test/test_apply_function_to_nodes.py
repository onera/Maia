import pytest

import Converter.Internal as I
import maia
import maia.pytree as PT

from maia.utils.yaml                   import parse_yaml_cgns
from maia.algo.apply_function_to_nodes import apply_to_zones, zones_iterator

def test_apply_to_zones():

  def add_vtx(zone):
    zone[1][0][0] += 1

  yt = """
  BaseA CGNSBase_t:
    zoneI Zone_t I4 [[8,5,0]]:
  BaseB CGNSBase_t:
    zoneII Zone_t I4 [[27,40,0]]:
      ZoneBC ZoneBC_t:
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  zone = I.getNodeFromPath(tree, 'BaseA/zoneI')
  other_base = I.getNodeFromPath(tree, 'BaseB')

  # Single zone
  apply_to_zones(add_vtx, zone)
  assert I.getNodeFromName(tree, 'zoneI')[1][0][0] == 9
  #Base
  apply_to_zones(add_vtx, other_base)
  assert I.getNodeFromName(tree, 'zoneII')[1][0][0] == 28
  #Tree
  apply_to_zones(add_vtx, tree)
  assert I.getNodeFromName(tree, 'zoneI')[1][0][0] == 10
  assert I.getNodeFromName(tree, 'zoneII')[1][0][0] == 29

  #BC (err)
  with pytest.raises(Exception):
    apply_to_zones(add_vtx, I.getNodeFromType(tree, 'ZoneBC_t'))

  
def test_zones_iterator():
  yt = """
  BaseA CGNSBase_t:
    zoneI Zone_t:
  BaseB CGNSBase_t:
    zoneII Zone_t:
      ZoneBC ZoneBC_t:
  """

  tree = parse_yaml_cgns.to_cgns_tree(yt)
  zone = I.getNodeFromPath(tree, 'BaseA/zoneI')
  other_base = I.getNodeFromPath(tree, 'BaseB')

  # Single zone
  zone = PT.get_all_Zone_t(tree)[0]
  assert [z[0] for z in zones_iterator(zone)] == ['zoneI']
  assert [z[0] for z in zones_iterator(other_base)] == ['zoneII']
  assert [z[0] for z in zones_iterator(tree)] == ['zoneI', 'zoneII']

  #Other node (err)
  with pytest.raises(ValueError):
    for z in zones_iterator(I.getNodeFromType(tree, 'ZoneBC_t')):
      pass

  
