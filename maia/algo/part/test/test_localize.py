import pytest
import pytest_parallel
import os
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia              import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.factory      import dcube_generator as DCG
from maia.factory      import partition_dist_tree

from maia.utils     import test_utils as TU
from maia.algo.part import localize as LOC

@pytest_parallel.mark.parallel(1)
def test_get_part_data(comm):
  dtree = DCG.dcube_generate(3, 1., [0.,0.,0.], comm)
  tree = partition_dist_tree(dtree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  data = LOC._get_part_data(zone)
  assert len(data) == 8
  assert (data[2] == MT.getGlobalNumbering(zone, 'Cell')[1]).all()
  assert data[6].size == 3*PT.Zone.n_vtx(zone) #Coords

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("reverse", [False, True])
def test_mesh_location(reverse, comm):
  dtree = DCG.dcube_generate(3, 1., [0.,0.,0.], comm)
  if comm.Get_rank() == 0:
    zone_to_parts = {'Base/zone' : [.5]}
    tgt_clouds = [(np.array([.6,.9,0]), np.array([2], pdm_gnum_dtype)), (np.array([1.6,.9,0]), np.array([4], pdm_gnum_dtype))]
  else:
    zone_to_parts = {'Base/zone' : [.25, .25]}
    tgt_clouds = [(np.array([.6,.9,0, .6,.9,.8, 1.2, 0.1, 0.1]), np.array([1,3,5], pdm_gnum_dtype))]
  tree = partition_dist_tree(dtree, comm, zone_to_parts=zone_to_parts)
  src_parts = [LOC._get_part_data(zone) for zone in PT.get_all_Zone_t(tree)]

  if reverse:
    tgt_data, src_data =  LOC._mesh_location(src_parts, [], comm, reverse)
    assert all([data['elt_pts_inside_idx'].sum() == 0 for data in src_data])
    assert all([data['elt_pts_inside_idx'].size-1 == PT.Zone.n_cell(part) for data,part in zip(src_data, PT.get_all_Zone_t(tree))])
    assert all([data['points_gnum'].size == 0 for data in src_data])
  else:
    tgt_data = LOC._mesh_location(src_parts, [], comm, reverse)
  assert tgt_data == []

  if reverse:
    tgt_data, src_data = LOC._mesh_location(src_parts, tgt_clouds, comm, reverse)
  else:
    tgt_data = LOC._mesh_location(src_parts, tgt_clouds, comm, reverse)

  if comm.rank == 0:
    expected_tgt_data = [{'located_ids' : [0], 'unlocated_ids' : [], 'location' : [4]},
                         {'located_ids' : [], 'unlocated_ids' : [0], 'location' : []}]
    expected_src_data = [{'elt_pts_inside_idx' : [0,0,0,0,1], 'points_gnum' : [3]}]
  if comm.rank == 1:
    expected_tgt_data = [{'located_ids' : [0,1], 'unlocated_ids' : [2], 'location' : [4,8]}]
    expected_src_data = [{'points_gnum' : []},
                         {'elt_pts_inside_idx' : [0,0,2], 'points_gnum' : [1,2]}]

  for i_part, expct_data in enumerate(expected_tgt_data):
    for key in expct_data:
      assert (tgt_data[i_part][key] == expct_data[key]).all()
  if reverse:
    for i_part, expct_data in enumerate(expected_src_data):
      for key in expct_data:
        assert np.allclose(src_data[i_part][key], expct_data[key])

@pytest_parallel.mark.parallel(1)
def test_mesh_location_mdom(comm):
  yaml_path = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dtree_src = DCG.dcube_generate(11, 20., [0.,0.,0.], comm)
  dtree_tgt = maia.io.file_to_dist_tree(yaml_path, comm)

  tree_src = partition_dist_tree(dtree_src, comm)
  tree_tgt = partition_dist_tree(dtree_tgt, comm)

  tgt_parts_per_dom = [PT.get_nodes_from_name_and_label(tree_tgt, 'Large*', 'Zone_t'),
                       PT.get_nodes_from_name_and_label(tree_tgt, 'Small*', 'Zone_t')]
  src_parts_per_dom = [PT.get_all_Zone_t(tree_src)]

  result, result_inv = LOC._localize_points(
      src_parts_per_dom, tgt_parts_per_dom, 'CellCenter', comm, reverse=True)
  # We should get all the cells of Large + 4*4*6 cells of Small
  _result_inv = result_inv[0][0]
  dom1_idx = np.where(_result_inv['domain'] == 1)[0]
  dom2_idx = np.where(_result_inv['domain'] == 2)[0]
  assert dom1_idx.size == PT.Zone.n_cell(tgt_parts_per_dom[0][0])
  assert dom2_idx.size == 4*4*6
  # Gnum should have been reshifted
  assert _result_inv['points_gnum'][dom1_idx].max() <= PT.Zone.n_cell(tgt_parts_per_dom[0][0])
  assert _result_inv['points_gnum'][dom2_idx].max() <= PT.Zone.n_cell(tgt_parts_per_dom[1][0])
  

@pytest_parallel.mark.parallel(3)
def test_localize_points(comm):
  dtree_src = DCG.dcube_generate(5, 1., [0.,0.,0.], comm)
  dtree_tgt = DCG.dcube_generate(4, 1., [.4,.05,.05], comm)
  tree_src = partition_dist_tree(dtree_src, comm)
  tree_tgt = partition_dist_tree(dtree_tgt, comm)

  tree_src_back = PT.deep_copy(tree_src)
  LOC.localize_points(tree_src, tree_tgt, 'CellCenter', comm)
  assert PT.is_same_tree(tree_src_back, tree_src)
  tgt_zone = PT.get_all_Zone_t(tree_tgt)[0]
  loc_node = PT.get_node_from_name_and_label(tgt_zone, 'Localization', 'DiscreteData_t')
  assert loc_node is not None and PT.Subset.GridLocation(loc_node) == 'CellCenter'
  assert PT.get_value(PT.get_child_from_name(loc_node, 'DomainList')) == "Base/zone"

  # Check result on dist tree to not rely on partitioning
  maia.transfer.part_tree_to_dist_tree_all(dtree_tgt, tree_tgt, comm)
  if comm.rank == 0:
    expected_dsrc_id = np.array([3,4,-1,11,12,-1,15,16,-1])
  elif comm.rank == 1:
    expected_dsrc_id = np.array([35,36,-1,43,44,-1,47,48,-1])
  elif comm.rank == 2:
    expected_dsrc_id = np.array([51,52,-1,59,60,-1,63,64,-1])

  assert (PT.get_node_from_name(dtree_tgt, 'SrcId')[1] == expected_dsrc_id).all()
