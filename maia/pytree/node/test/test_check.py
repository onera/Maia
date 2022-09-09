import numpy as np

from maia.pytree.node import check

def test_is_valid_name():
  assert check.is_valid_name('MyName') == True
  assert check.is_valid_name(123) == False

def test_is_valid_value():
  assert check.is_valid_value(123) == False
  assert check.is_valid_value('Coucou') == False
  assert check.is_valid_value(np.array(tuple('Coucou'), '|S1')) == True
  assert check.is_valid_value(np.array([123])) == True
  assert check.is_valid_value(np.ones((3,3), order='F')) == True
  assert check.is_valid_value(np.ones((3,3), order='C')) == False

def test_is_valid_children():
  assert check.is_valid_children(['BC', 'BCWall', [], 'BC_t']) == True
  assert check.is_valid_children(123) == False

def test_is_valid_label():
  assert check.is_valid_label('') == False
  assert check.is_valid_label('BC') == False
  assert check.is_valid_label('BC_t') == True
  assert check.is_valid_label('BC_toto') == False
  assert check.is_valid_label('FakeLabel_t', only_sids=False) == True
  assert check.is_valid_label('FakeLabel_t', only_sids=True) == False

def test_is_valid_node():
  assert check.is_valid_node(['BC', np.array(tuple('BCWall'), '|S1'), [], 'BC_t']) == True
  assert check.is_valid_node(123) == False
  assert check.is_valid_node(['BC', np.array(tuple('BCWall'), '|S1'), [], 'BC_t', 'Coucou']) == False

