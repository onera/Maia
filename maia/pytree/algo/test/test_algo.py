import pytest

import maia.pytree as PT
from maia.pytree.yaml import parse_yaml_cgns

from maia.pytree.algo.graph import depth_first_search, zip_depth_first_search, pytree_zip_adaptor

t0 = parse_yaml_cgns.to_node("""
Base CGNSBase_t:
  ZoneI0 Zone_t:
    NGon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZGCA ZoneGridConnectivity_t:
      gc1 GridConnectivity_t:
        Index_i IndexArray_t:
""")

# same as t0 but NGon/NFace swapped
t1 = parse_yaml_cgns.to_node("""
Base CGNSBase_t:
  ZoneI1 Zone_t:
    NFace Elements_t [23,0]:
    NGon Elements_t [22,0]:
    ZGCB ZoneGridConnectivity_t:
      gc1 GridConnectivity_t:
        Index_i IndexArray_t:
        Index_j IndexArray_t:
""")

class node_name_recorder:
  def __init__(self):
    self.s = ''

  def pre(self, n):
    self.s += PT.get_name(n) + '\n'

def test_tree_algo():
  v = node_name_recorder()
  depth_first_search(t0, v)

  expected_s = \
    'Base\n' \
    'ZoneI0\n' \
    'NGon\n' \
    'NFace\n' \
    'ZGCA\n' \
    'gc1\n' \
    'Index_i\n'
  assert v.s == expected_s


class two_node_names_recorder:
  def __init__(self):
    self.s = ''

  def pre(self, ns):
    def get_name(n):
      if n is None:
        return '[None]'
      else:
        return PT.get_name(n)

    self.s += get_name(ns[0])
    self.s += ' | '
    self.s += get_name(ns[1])
    self.s += '\n'

def test_zip_tree_algo():
  v = two_node_names_recorder()
  zip_depth_first_search([t0,t1], v)

  expected_s = \
    'Base | Base\n' \
    'ZoneI0 | ZoneI1\n' \
    'NFace | NFace\n' \
    'NGon | NGon\n' \
    'ZGCA | ZGCB\n' \
    'gc1 | gc1\n' \
    'Index_i | Index_i\n' \
    '[None] | Index_j\n'
  assert v.s == expected_s
