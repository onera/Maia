import pytest

from maia.sids.pytree import path_utils as PU

def test_path_head():
  assert PU.path_head('some/path/to/node', 2) == 'some/path'
  assert PU.path_head('some/path/to/node', 4) == 'some/path/to/node'
  assert PU.path_head('some/path/to/node', 0) == ''
  assert PU.path_head('some/path/to/node', -2) == 'some/path'
  assert PU.path_head('some/path/to/node') == 'some/path/to'

def test_path_tail():
  assert PU.path_tail('some/path/to/node', 2) == 'to/node'
  assert PU.path_tail('some/path/to/node', 0) == 'some/path/to/node'
  assert PU.path_tail('some/path/to/node', -1) == 'node'

def test_update_path_elt():
  path = 'some/path/to/node'
  assert PU.update_path_elt(path, 3, lambda n : 'something') == 'some/path/to/something'
  assert PU.update_path_elt(path, -1, lambda n : n.upper()) == 'some/path/to/NODE'
  assert PU.update_path_elt(path, 1, lambda n : 'crazy' + n) == 'some/crazypath/to/node'
