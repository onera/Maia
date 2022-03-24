import pytest

from maia.sids.pytree import path_utils as PU

def test_path_head():
  assert PU.path_head('some/path/to/node', 2) == 'some/path'
  assert PU.path_head('some/path/to/node', 4) == 'some/path/to/node'
  assert PU.path_head('some/path/to/node', 0) == ''
  assert PU.path_head('some/path/to/node', -2) == 'some/path'
  assert PU.path_head('some/path/to/node') == 'some/path/to'

