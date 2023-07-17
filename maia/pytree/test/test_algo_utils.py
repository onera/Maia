def test_find_not():
  def is_7(x): return x==7
  assert find_not([7,7,8,7,9], is_7) == 2
  assert find_not([8,7,9], is_7) == 0
  assert find_not([8,9], is_7) == 0
  assert find_not([7,7], is_7) == 2


