import pytest

from  maia import utils
from cmaia import cpp20_enabled

def test_require_cpp20():
  # Test it on a random func
  fct = utils.py_utils.to_nested_list
  decorated = utils.require_cpp20(fct)

  assert decorated.__name__ == fct.__name__

  if cpp20_enabled:
    assert fct(['a', 'b', 'c', 'd', 'e'], [3,2]) == decorated(['a', 'b', 'c', 'd', 'e'], [3,2])
  else:
    with pytest.raises(Exception):
      decorated(['a', 'b', 'c', 'd', 'e'], [3,2])
