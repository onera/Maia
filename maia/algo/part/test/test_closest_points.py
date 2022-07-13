import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia              import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.factory      import dcube_generator as DCG
from maia.factory      import partition_dist_tree

from maia.algo.part import closest_points as CLO

@mark_mpi_test(2)
class Test_closest_points:
  src_clouds_per_rank = {
      0 : [(np.array([.25,.25,.75, .75,.25,75, .25,.75,.75, .75,.75,.75]), np.array([5,6,7,8], pdm_gnum_dtype))],
      1 : [(np.array([.25,.25,.25, .25,.75,.25]), np.array([1,3], pdm_gnum_dtype)), 
           (np.array([.75,.25,.25, .75,.75,.25]), np.array([2,4], pdm_gnum_dtype))]
      }
  tgt_clouds_per_rank = {
      0 : [(np.array([.6,.9,0]), np.array([2], pdm_gnum_dtype)), (np.array([1.6,.9,0]), np.array([4], pdm_gnum_dtype))],
      1 : [(np.array([.6,.9,0, .6,.9,.8, 1.2, 0.1, 0.1]), np.array([1,3,5], pdm_gnum_dtype))]
      }

  def test_empty_tgt(self, sub_comm):
    src_clouds = self.src_clouds_per_rank[sub_comm.Get_rank()]
    assert CLO._closest_points(src_clouds, [], sub_comm) == []

  @pytest.mark.parametrize("reverse", [False, True])
  def test_standard(self, reverse, sub_comm):
    src_clouds = self.src_clouds_per_rank[sub_comm.Get_rank()]
    tgt_clouds = self.tgt_clouds_per_rank[sub_comm.Get_rank()]

    if reverse:
      tgt_data, src_data = CLO._closest_points(src_clouds, tgt_clouds, sub_comm, n_pts=1, reverse=True)
    else:
      tgt_data = CLO._closest_points(src_clouds, tgt_clouds, sub_comm, n_pts=1, reverse=False)

    if sub_comm.Get_rank() == 0:
      expected_tgt_data = [{'closest_src_gnum' : [4], 'closest_src_distance' : [0.1075]},
                           {'closest_src_gnum' : [4], 'closest_src_distance' : [0.8075]}]
      expected_src_data = [{'tgt_in_src_idx' : [0,0,0,0,1], 'tgt_in_src' : [3], 'tgt_in_src_dist2' : [0.0475]}]
    elif sub_comm.Get_rank() == 1:
      expected_tgt_data = [{'closest_src_gnum' : [4,8,2], 'closest_src_distance' : [0.1075, 0.0475, 0.2475]}]
      expected_src_data = [{'tgt_in_src_idx' : [0,0,0], 'tgt_in_src' : [], 'tgt_in_src_dist2' : []},
                           {'tgt_in_src_idx' : [0,1,4], 'tgt_in_src' : [5,2,4,1], 'tgt_in_src_dist2' : [0.2475, 0.1075, 0.8075, 0.1075]}]

    for i_part, expct_data in enumerate(expected_tgt_data):
      for key in expct_data:
        assert np.allclose(tgt_data[i_part][key], expct_data[key])
    if reverse:
      for i_part, expct_data in enumerate(expected_src_data):
        for key in expct_data:
          assert np.allclose(src_data[i_part][key], expct_data[key])

  def test_mult_pts(self, sub_comm):
    src_clouds = self.src_clouds_per_rank[sub_comm.Get_rank()]
    tgt_clouds = self.tgt_clouds_per_rank[sub_comm.Get_rank()]

    tgt_data = CLO._closest_points(src_clouds, tgt_clouds, sub_comm, n_pts=3)

    if sub_comm.Get_rank() == 0:
      expected_tgt_data = [{'closest_src_gnum' : [4,3,2], 'closest_src_distance' : [0.1075, 0.2075, 0.5075]},
                           {'closest_src_gnum' : [4,2,8], 'closest_src_distance' : [0.8075, 1.2075, 1.3075]}]
    elif sub_comm.Get_rank() == 1:
      expected_tgt_data = [{'closest_src_gnum' : [4,3,2, 8,7,4, 2,4,1], 
                            'closest_src_distance' : [0.1075, 0.2075, 0.5075, 0.0475, 0.1475, 0.3475, 0.2475, 0.6475, 0.9475]}]

    for i_part, expct_data in enumerate(expected_tgt_data):
      for key in expct_data:
        assert np.allclose(tgt_data[i_part][key], expct_data[key])

@mark_mpi_test(3)
def test_localize_points(sub_comm):
  dtree_src = DCG.dcube_generate(5, 1., [0.,0.,0.], sub_comm)
  dtree_tgt = DCG.dcube_generate(4, 1., [.4,0.,0.], sub_comm)
  tree_src = partition_dist_tree(dtree_src, sub_comm)
  tree_tgt = partition_dist_tree(dtree_tgt, sub_comm)

  tree_src_back = I.copyTree(tree_src)
  CLO.find_closest_points(tree_src, tree_tgt, 'CellCenter', sub_comm)
  assert PT.is_same_tree(tree_src_back, tree_src)
  tgt_zone = I.getZones(tree_tgt)[0]
  clo_node = I.getNodeFromNameAndType(tgt_zone, 'ClosestPoint', 'DiscreteData_t')
  assert clo_node is not None and PT.Subset.GridLocation(clo_node) == 'CellCenter'

  if sub_comm.rank == 0:
    expected_src_id = np.array([3,19,51,20,16,4,8,7,52])
  elif sub_comm.rank == 1:
    expected_src_id = np.array([15,59,63,60,32,23,47,24,64])
  elif sub_comm.rank == 2:
    expected_src_id = np.array([4,48,60,52,24,8,20,16,64])

  assert (I.getNodeFromName1(clo_node, 'SrcId')[1] == expected_src_id).all()
