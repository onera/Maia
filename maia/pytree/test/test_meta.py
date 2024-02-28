import pytest
import os

import maia.pytree as PT
from maia.pytree.yaml   import parse_yaml_cgns

from maia.pytree import meta

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_check_is_label():
  with open(os.path.join(dir_path, "minimal_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  @meta.check_is_label('Zone_t')
  def apply_zone(node):
    pass

  for zone in PT.get_all_Zone_t(tree):
    apply_zone(zone)

  with pytest.raises(meta.CGNSLabelNotEqualError):
    for zone in PT.get_all_CGNSBase_t(tree):
      apply_zone(zone)


def test_check_in_labels():
  with open(os.path.join(dir_path, "minimal_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  @meta.check_in_labels(['Zone_t', 'CGNSBase_t'])
  def foo(node):
    pass

  for zone in PT.get_all_Zone_t(tree):
    foo(zone)
  for zone in PT.get_all_CGNSBase_t(tree):
    foo(zone)
  with pytest.raises(meta.CGNSLabelNotEqualError):
    foo(tree)
