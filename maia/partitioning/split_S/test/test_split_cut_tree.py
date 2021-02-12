from   maia.partitioning.split_S import split_cut_tree     as SCT

def test_part_volume():
  assert SCT.get_part_volume([[0, 7], [0, 5]]) == 5*7
  assert SCT.get_part_volume([[2, 7], [1, 5]]) == 5*4
  assert SCT.get_part_volume([[2, 7], [1, 1]]) == 0
  assert SCT.get_part_volume([[5, 10], [2, 10], [0,4]]) == 5*8*4

def test_apply_weights_to_cut_tree():
  assert SCT.apply_weights_to_cut_tree([[2], [1], [1]], [.1, .2, .3, .4]) \
      == [[.1, .2], [.3], [.4]]
  assert SCT.apply_weights_to_cut_tree([[[1], [1]], [[2]]], [.2, .1, .5, .2]) \
      == [[[.2], [.1]], [[.5,.2]]]

class Test_bct_to_partitions_bounds():
  def test_2d_unweighted(self):
    parts = SCT.bct_to_partitions_bounds([[2], [1]], [10,10])
    assert len(parts) == 3
    assert parts[0] == [[0, 7], [0, 5]]
    assert parts[1] == [[0, 7], [5, 10]]
    assert parts[2] == [[7, 10], [0, 10]]

    parts = SCT.bct_to_partitions_bounds([[1], [1], [1]], [30,10])
    assert len(parts) == 3
    assert parts[0] == [[0, 10], [0, 10]]
    assert parts[1] == [[10, 20], [0, 10]]
    assert parts[2] == [[20, 30], [0, 10]]

  def test_2d_weighted(self):
    parts = SCT.bct_to_partitions_bounds([[1], [2]], [10,10], [[.5],[.25,.25]])
    assert len(parts) == 3
    assert parts[0] == [[0, 5], [0, 10]]
    assert parts[1] == [[5, 10], [0, 5]]
    assert parts[2] == [[5, 10], [5, 10]]

  def test_3d_unweighted(self):
    tree = [[[1], [1]], [[2]]]
    dims = [10, 10, 10]
    parts = SCT.bct_to_partitions_bounds(tree, dims)
    assert len(parts) == 4
    assert parts[0] == [[0, 5], [0, 5], [0,10]]
    assert parts[1] == [[0, 5], [5, 10], [0,10]]
    assert parts[2] == [[5, 10], [0, 10], [0,5]]
    assert parts[3] == [[5, 10], [0, 10], [5,10]]

    tree = [[[1]], [[1]], [[1]], [[1]]]
    dims = [30, 10, 5]
    parts = SCT.bct_to_partitions_bounds(tree, dims)
    assert len(parts) == 4
    assert parts[0] == [[0, 8], [0, 10], [0,5]]
    assert parts[1] == [[8, 15], [0, 10], [0,5]]
    assert parts[2] == [[15, 23], [0, 10], [0,5]]
    assert parts[3] == [[23, 30], [0, 10], [0,5]]

  def test_3d_unweighted(self):
    tree  = [[[1], [1]], [[2]]]
    wtree = [[[.2], [.1]], [[.5,.2]]]
    dims = [10, 10, 10]
    parts = SCT.bct_to_partitions_bounds(tree, dims, wtree)
    assert len(parts) == 4
    assert parts[0] == [[0, 3], [0, 7], [0,10]]
    assert parts[1] == [[0, 3], [7, 10], [0,10]]
    assert parts[2] == [[3, 10], [0, 10], [0,7]]
    assert parts[3] == [[3, 10], [0, 10], [7,10]]

class Test_part_block_from_dims():
  def test_unweighted_2d(self):
    parts = SCT.split_S_block([10,10], 4)
    assert parts[0] == [[0, 5], [0, 5]]
    assert parts[1] == [[0, 5], [5, 10]]
    assert parts[2] == [[5, 10], [0, 5]]
    assert parts[3] == [[5, 10], [5, 10]]

    parts = SCT.split_S_block([20,15], 3)
    assert parts[0] == [[0, 13], [0, 8]]
    assert parts[1] == [[0, 13], [8, 15]]
    assert parts[2] == [[13, 20], [0, 15]]

    parts = SCT.split_S_block([100,2], 3)
    assert parts[0] == [[0, 33], [0, 2]]
    assert parts[1] == [[33, 67], [0, 2]]
    assert parts[2] == [[67, 100], [0, 2]]

  def test_weighted_2d(self):
    parts = SCT.split_S_block([20,15], 3, [.2, .3, .5])
    assert parts[0] == [[0, 10], [0, 6]]
    assert parts[1] == [[0, 10], [6, 15]]
    assert parts[2] == [[10, 20], [0, 15]]
    parts = SCT.split_S_block([20,15], 3, [.2, .5, .3])
    assert parts[0] == [[0, 10], [0, 6]]
    assert parts[1] == [[10, 20], [0, 15]]
    assert parts[2] == [[0, 10], [6, 15]]

  def test_unweighted_3d(self):
    parts = SCT.split_S_block([50, 43, 26], 6)
    assert parts[0] == [[0, 25], [0, 29], [0, 13]]
    assert parts[1] == [[0, 25], [0, 29], [13, 26]]
    assert parts[2] == [[0, 25], [29, 43], [0, 26]]
    assert parts[3] == [[25, 50], [0, 29], [0, 13]]
    assert parts[4] == [[25, 50], [0, 29], [13, 26]]
    assert parts[5] == [[25, 50], [29, 43], [0, 26]]

    assert len(SCT.split_S_block([50, 43, 26], 13)) == 13

  def test_weighted_3d(self):
    weights = [.2, .15, .15, .3, .1, .1]
    parts = SCT.split_S_block([50, 43, 26], 6, weights)
    assert parts[0] == [[0, 25], [0, 30], [0, 15]]
    assert parts[1] == [[0, 25], [0, 30], [15, 26]]
    assert parts[2] == [[0, 25], [30, 43], [0, 26]]
    assert parts[3] == [[25, 50], [17, 43], [0, 26]]
    assert parts[4] == [[25, 50], [0, 17], [0, 13]]
    assert parts[5] == [[25, 50], [0, 17], [13, 26]]
    error1 = [SCT.get_part_volume(part) / (50*43*26) - w \
        for part,w in zip(parts,weights)]

    parts = SCT.split_S_block([50, 43, 26], 6, weights,max_it=1)
    assert parts[0] == [[0, 25], [0, 30], [0, 15]]
    assert parts[1] == [[0, 25], [0, 30], [15, 26]]
    assert parts[2] == [[0, 25], [30, 43], [0, 26]]
    assert parts[3] == [[25, 50], [0, 34], [0, 19]]
    assert parts[4] == [[25, 50], [0, 34], [19, 26]]
    assert parts[5] == [[25, 50], [34, 43], [0, 26]]
    error2 = [SCT.get_part_volume(part) / (50*43*26) - w \
        for part,w in zip(parts,weights)]

    assert max(error1) <= max(error2)
