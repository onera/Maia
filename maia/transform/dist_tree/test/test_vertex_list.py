from maia.transform.dist_tree import vertex_list as VL

def test_is_subset_l():
  L = [2,8,10,3,3]
  assert VL._is_subset_l([2],        L) == True
  assert VL._is_subset_l([10,3],     L) == True
  assert VL._is_subset_l([10,3,3],   L) == True
  assert VL._is_subset_l([3,2,8],    L) == True
  assert VL._is_subset_l([1],        L) == False
  assert VL._is_subset_l([3,8,2],    L) == False
  assert VL._is_subset_l([10,3,3,1], L) == False

def test_is_before():
  L = [2,8,10,3,3]
  assert VL._is_before(L, 8, 10) == True
  assert VL._is_before(L, 8, 9 ) == True
  assert VL._is_before(L, 8, 2 ) == False
  assert VL._is_before(L, 7, 2 ) == False
  assert VL._is_before(L, 7, 14) == False
