import os
import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia              import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.factory      import dcube_generator as DCG
from maia.factory      import partition_dist_tree

from maia.utils     import test_utils as TU
from maia.algo.part import closest_points as CLO

@pytest_parallel.mark.parallel(2)
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

  def test_empty_tgt(self, comm):
    src_clouds = self.src_clouds_per_rank[comm.Get_rank()]
    assert CLO._closest_points(src_clouds, [], comm) == []

  @pytest.mark.parametrize("reverse", [False, True])
  def test_standard(self, reverse, comm):
    src_clouds = self.src_clouds_per_rank[comm.Get_rank()]
    tgt_clouds = self.tgt_clouds_per_rank[comm.Get_rank()]

    if reverse:
      tgt_data, src_data = CLO._closest_points(src_clouds, tgt_clouds, comm, n_pts=1, reverse=True)
    else:
      tgt_data = CLO._closest_points(src_clouds, tgt_clouds, comm, n_pts=1, reverse=False)

    if comm.Get_rank() == 0:
      expected_tgt_data = [{'closest_src_gnum' : [4], 'closest_src_distance' : [0.1075]},
                           {'closest_src_gnum' : [4], 'closest_src_distance' : [0.8075]}]
      expected_src_data = [{'tgt_in_src_idx' : [0,0,0,0,1], 'tgt_in_src' : [3], 'tgt_in_src_dist2' : [0.0475]}]
    elif comm.Get_rank() == 1:
      expected_tgt_data = [{'closest_src_gnum' : [4,8,2], 'closest_src_distance' : [0.1075, 0.0475, 0.2475]}]
      expected_src_data = [{'tgt_in_src_idx' : [0,0,0], 'tgt_in_src' : [], 'tgt_in_src_dist2' : []},
                           {'tgt_in_src_idx' : [0,1,4], 'tgt_in_src' : [5,1,2,4], 'tgt_in_src_dist2' : [0.2475, 0.1075, 0.1075, 0.8075]}]

    for i_part, expct_data in enumerate(expected_tgt_data):
      for key in expct_data:
        assert np.allclose(tgt_data[i_part][key], expct_data[key])
    if reverse:
      for i_part, expct_data in enumerate(expected_src_data):
        for key in expct_data:
          assert np.allclose(src_data[i_part][key], expct_data[key])

  def test_mult_pts(self, comm):
    src_clouds = self.src_clouds_per_rank[comm.Get_rank()]
    tgt_clouds = self.tgt_clouds_per_rank[comm.Get_rank()]

    tgt_data = CLO._closest_points(src_clouds, tgt_clouds, comm, n_pts=3)

    if comm.Get_rank() == 0:
      expected_tgt_data = [{'closest_src_gnum' : [2,3,4], 'closest_src_distance' : [0.5075, 0.2075, 0.1075]},
                           {'closest_src_gnum' : [2,4,8], 'closest_src_distance' : [1.2075, 0.8075, 1.3075]}]
    elif comm.Get_rank() == 1:
      expected_tgt_data = [{'closest_src_gnum' : [2,3,4, 4,7,8, 1,2,4], 
                            'closest_src_distance' : [0.5075, 0.2075, 0.1075, 0.3475, 0.1475, 0.0475, 0.9475, 0.2475, 0.6475]}]

    for i_part, expct_data in enumerate(expected_tgt_data):
      for key in expct_data:
        assert np.allclose(tgt_data[i_part][key], expct_data[key])

@pytest_parallel.mark.parallel(1)
def test_closestpoint_mdom(comm):
  yaml_path = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dtree_src = maia.io.file_to_dist_tree(yaml_path, comm)
  dtree_tgt = DCG.dcube_generate(5, 4., [0.,0.,0.], comm)
  maia.algo.transform_affine(dtree_tgt, translation=np.array([13.25, 2.25, 0.25]))

  tree_src = partition_dist_tree(dtree_src, comm)
  tree_tgt = partition_dist_tree(dtree_tgt, comm)
  tgt_parts_per_dom = [PT.get_all_Zone_t(tree_tgt)]
  src_parts_per_dom = [PT.get_nodes_from_name_and_label(tree_src, 'Large*', 'Zone_t'),
                       PT.get_nodes_from_name_and_label(tree_src, 'Small*', 'Zone_t')]

  result, result_inv = CLO._find_closest_points(
      src_parts_per_dom, tgt_parts_per_dom, 'Vertex', 'Vertex', comm, reverse=True)
  _result = result[0][0]
  dom1_idx = np.where(_result['domain'] == 1)[0]
  dom2_idx = np.where(_result['domain'] == 2)[0] 
  assert dom1_idx.size == 87 #Carefull, 25 resulting points are on the interface, result can change
  assert dom2_idx.size == 38
  # Gnum should have been reshifted
  assert _result['closest_src_gnum'][dom1_idx].max() <= PT.Zone.n_vtx(src_parts_per_dom[0][0])
  assert _result['closest_src_gnum'][dom2_idx].max() <= PT.Zone.n_vtx(src_parts_per_dom[1][0])
  
@pytest_parallel.mark.parallel(3)
def test_closest_points(comm):
  dtree_src = DCG.dcube_generate(5, 1., [0.,0.,0.], comm)
  dtree_tgt = DCG.dcube_generate(4, 1., [.4,-0.01,-0.01], comm)
  tree_src = partition_dist_tree(dtree_src, comm)
  tree_tgt = partition_dist_tree(dtree_tgt, comm)

  tree_src_back = PT.deep_copy(tree_src)
  CLO.find_closest_points(tree_src, tree_tgt, 'CellCenter', comm)
  assert PT.is_same_tree(tree_src_back, tree_src)
  tgt_zone = PT.get_all_Zone_t(tree_tgt)[0]
  clo_node = PT.get_node_from_name_and_label(tgt_zone, 'ClosestPoint', 'DiscreteData_t')
  assert clo_node is not None and PT.Subset.GridLocation(clo_node) == 'CellCenter'
  assert PT.get_value(PT.get_child_from_name(clo_node, 'DomainList')) == "Base/zone"

  # Check result on dist tree to not rely on partitioning
  maia.transfer.part_tree_to_dist_tree_all(dtree_tgt, tree_tgt, comm)
  if comm.rank == 0:
    expected_dsrc_id = np.array([3,4,4,7,8,8,15,16,16])
  elif comm.rank == 1:
    expected_dsrc_id = np.array([19,20,20,23,24,24,31,32,32])
  elif comm.rank == 2:
    expected_dsrc_id = np.array([51,52,52,55,56,56,63,64,64])
  assert (PT.get_node_from_name(dtree_tgt, 'SrcId')[1] == expected_dsrc_id).all()
