from maia.pytree.algo_utils import find, find_not, mismatch, begins_with

def test_find():
  def is_7(x): return x==7
  assert find([7,7,8,7,9], is_7) == 0
  assert find([8,7,9], is_7) == 1
  assert find([8,9], is_7) == 2

def test_find_not():
  def is_7(x): return x==7
  assert find_not([7,7,8,7,9], is_7) == 2
  assert find_not([8,7,9], is_7) == 0
  assert find_not([8,9], is_7) == 0
  assert find_not([7,7], is_7) == 2


def test_mismatch():
  assert mismatch(['a','b','c'], []               ) == 0
  assert mismatch(['a','b','c'], ['a']            ) == 1
  assert mismatch(['a','b','c'], ['a','b']        ) == 2
  assert mismatch(['a','b','c'], ['a','b','c']    ) == 3
  assert mismatch(['a','b','c'], ['a','b','c','d']) == 3

  assert mismatch(['a','b','c'], ['b']) == 0
  assert mismatch(['a','b','c'], ['a','c','c']) == 1

def test_begins_with():
  assert begins_with([0,1,2], [])
  assert begins_with([0,1,2], [0])
  assert begins_with([0,1,2], [0,1,2])

  assert not begins_with([0,1,2], [2])
  assert not begins_with([0,1,2], [0,1,2,3])
