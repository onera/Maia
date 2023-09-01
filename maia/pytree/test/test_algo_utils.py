from maia.pytree.algo_utils import find, find_not, mismatch, begins_with, partition_copy, set_intersection_difference

def test_find():
  assert find([7,7,8,7,9], 7) == 0
  assert find([8,7,9], 7) == 1
  assert find([8,9], 7) == 2

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


def test_partition_copy():
  def pred(x):
    return x < 5
  xs = [3,7,6,2,0,5]
  assert partition_copy(xs, pred) == ([3,2,0],[7,6,5])


def test_set_intersection_difference():
  xs = [2,4,5,7]
  ys = [0,2,3,5,8]

  x_inter_y_in_x, x_diff_y, x_inter_y_in_y, y_diff_x = set_intersection_difference(xs, ys)

  assert x_inter_y_in_x == [2,5]
  assert x_diff_y       == [4,7]
  assert x_inter_y_in_y == [2,5] # same as `x_inter_y_in_x` because we are comparing through a total order
  assert y_diff_x       == [0,3,8]

def test_set_intersection_difference_comp_on_projection():
  xs = [(2,'Alice'), (4,'Bob'), (5,'Charlie'), (7,'David')]
  ys = [(0,'Eve'), (2,'Frank'), (3,'Grace'), (5,'Heidi'), (8,'Ivan')]
  def comp(x, y):
    return x[0] < y[0]

  x_inter_y_in_x, x_diff_y, x_inter_y_in_y, y_diff_x = set_intersection_difference(xs, ys, comp)

  assert x_inter_y_in_x == [(2,'Alice'), (5,'Charlie')]
  assert x_diff_y       == [(4,'Bob'), (7,'David')]
  assert x_inter_y_in_y == [(2,'Frank'), (5,'Heidi')] # not the same as `x_inter_y_in_x` because we are comparing through a projection
  assert y_diff_x       == [(0,'Eve'), (3,'Grace'),(8,'Ivan')]
