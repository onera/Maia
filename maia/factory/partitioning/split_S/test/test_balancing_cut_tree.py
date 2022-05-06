import pytest
from   maia.factory.partitioning.split_S import balancing_cut_tree as BCT

def test_init_cut_tree():
  assert BCT.init_cut_tree(2) == [[1]]
  assert BCT.init_cut_tree(3) == [[[1]]]

def test_reset_sub_tree():
  tree = [[[2],[3]], [[1],[5],[3]], [[8]]]
  BCT.reset_sub_tree(tree[1][2])
  assert tree == [[[2],[3]], [[1],[5],[1]], [[8]]]
  BCT.reset_sub_tree(tree[0])
  assert tree == [[[1],[1]], [[1],[5],[1]], [[8]]]
  BCT.reset_sub_tree(tree)
  assert tree == [[[1],[1]], [[1],[1],[1]], [[1]]]

def test_depth():
  assert BCT.depth(2) == 0
  assert BCT.depth("oups") == 0
  assert BCT.depth([2,4,6,8]) == 1
  assert BCT.depth([2,4,[6],8]) == 2
  assert BCT.depth([2,4,[[6]],8]) == 3
  assert BCT.depth([2,[4,[6]],8]) == 3

def test_sum_leaves():
  tree = [[[2],[3]], [[1],[5],[3]], [[1]]]
  assert BCT.sum_leaves([[], []]) == 0
  assert BCT.sum_leaves(tree) == 15
  assert BCT.sum_leaves(tree[1]) == 9
  assert BCT.sum_leaves(tree[1][2]) == 3

def test_child_with_least_leaves():
  tree = [[[2],[3]], [[5],[1],[3]], [[1]]]
  assert BCT.child_with_least_leaves(tree) is tree[2]
  assert BCT.child_with_least_leaves(tree[0]) is tree[0][0]
  assert BCT.child_with_least_leaves(tree[1]) is tree[1][1]
  assert BCT.child_with_least_leaves(tree[2]) is tree[2][0]

def test_weight():
  tree = [[[2],[3]], [[5],[1],[3]], [[1]]]
  dims = [10, 15, 20]
  assert BCT.weight(tree, dims) == 3./dims[0]
  assert BCT.weight(tree[0], dims) == 2./dims[1]
  assert BCT.weight(tree[1], dims) == 3./dims[1]
  assert BCT.weight(tree[2], dims) == 1./dims[1]
  assert BCT.weight(tree[1][0], dims) == 5./dims[2]
  assert BCT.weight(tree[1][1], dims) == 1./dims[2]
  assert BCT.weight(tree[1][2], dims) == 3./dims[2]
  assert BCT.weight(tree[2][0], dims) == 1./dims[2]
  with pytest.raises(ValueError):
    BCT.weight(tree[0][1][0], dims)

def test_select_insertion():
  tree = [[[2],[3]], [[5],[1],[3]], [[1]]]
  assert BCT.select_insertion(tree, tree, [10,15,20]) is tree[2][0]
  assert BCT.select_insertion(tree, tree, [100,150,20]) is tree[2]
  assert BCT.select_insertion(tree[1], tree[1], [10,15,20]) is tree[1][1]
  assert BCT.select_insertion(tree[1], tree[1], [10,150,20]) is tree[1]

def test_insert_child_at():
  tree = [[[2],[3]], [[5],[1],[3]], [[1]]]
  BCT.insert_child_at(tree[1][2], [10,15,20])
  assert tree == [[[2],[3]], [[5],[1],[4]], [[1]]]
  BCT.insert_child_at(tree[0], [10,15,20])  #Subnode is rebalanced
  assert tree == [[[2],[2], [2]], [[5],[1],[4]], [[1]]]
  BCT.insert_child_at(tree, [10,15,20])  #Whole tree is rebalanced
  assert tree == [[[2], [2], [1]], [[2], [2], [1]], [[2], [2]], [[2], [2]]]
  with pytest.raises(ValueError):
    BCT.insert_child_at(tree[3][0][0], [10,15,20]) 

def test_refine_cut_tree():
  tree = [[[2],[3]], [[5],[1],[3]], [[1]]]
  BCT.refine_cut_tree(tree, [10,15,20])
  assert tree == [[[2], [3]], [[5], [1], [3]], [[2]]]
  BCT.refine_cut_tree(tree, [10,15,20])
  assert tree == [[[2], [3]], [[5], [1], [3]], [[2], [1]]]
  n_leaves_current = BCT.sum_leaves(tree)
  for i in range(5):
    BCT.refine_cut_tree(tree, [10,15,20])
  assert BCT.sum_leaves(tree) == n_leaves_current + 5

