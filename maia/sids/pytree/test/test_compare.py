import pytest
import os
import numpy as np

import Converter.Internal as I

from maia.sids.cgns_keywords import Label as CGL
from maia.sids import pytree as PT

from maia.utils        import parse_yaml_cgns

dir_path = os.path.dirname(os.path.realpath(__file__))

def test_is_valid_name():
  assert PT.is_valid_name('MyName') == True
  assert PT.is_valid_name(123) == False

def test_check_name():
  assert PT.check_name('MyName') == 'MyName'
  with pytest.raises(TypeError):
    PT.check_name(123)

def test_is_valid_value():
  assert PT.is_valid_value(123) == False
  assert PT.is_valid_value('Coucou') == False
  assert PT.is_valid_value(np.array(tuple('Coucou'), '|S1')) == True
  assert PT.is_valid_value(np.array([123])) == True
  assert PT.is_valid_value(np.ones((3,3), order='F')) == True
  assert PT.is_valid_value(np.ones((3,3), order='C')) == False

def test_check_value():
  assert PT.check_value(np.array([123])) == np.array([123])
  with pytest.raises(TypeError):
    PT.check_value(123)

def test_is_valid_children():
  assert PT.is_valid_children(['BC', 'BCWall', [], 'BC_t']) == True
  assert PT.is_valid_children(123) == False

def test_check_children():
  assert PT.check_children(['BC', 'BCWall', [], 'BC_t']) == ['BC', 'BCWall', [], 'BC_t']
  with pytest.raises(TypeError):
    PT.check_children(123)

def test_is_valid_label():
  assert PT.is_valid_label('') == True
  assert PT.is_valid_label('BC') == False
  assert PT.is_valid_label('BC_t') == True
  assert PT.is_valid_label('BC_toto') == False
  assert PT.is_valid_label('FakeLabel_t') == False

  assert PT.is_valid_label(CGL.BC_t.name) == True

def test_check_label():
  assert PT.check_label('BC_t') == 'BC_t'
  with pytest.raises(TypeError):
    PT.check_label('BC')

def test_is_valid_node():
  assert PT.is_valid_node(['BC', np.array(tuple('BCWall'), '|S1'), [], 'BC_t']) == True
  assert PT.is_valid_node(123) == False
  assert PT.is_valid_node(['BC', np.array(tuple('BCWall'), '|S1'), [], 'BC_t', 'Coucou']) == False


def test_check_is_label():
  with open(os.path.join(dir_path, "minimal_bc_tree.yaml"), 'r') as yt:
    tree = parse_yaml_cgns.to_cgns_tree(yt)

  @PT.check_is_label('Zone_t')
  def apply_zone(node):
    pass

  for zone in I.getZones(tree):
    apply_zone(zone)

  with pytest.raises(PT.CGNSLabelNotEqualError):
    for zone in I.getBases(tree):
      apply_zone(zone)
