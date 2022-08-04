import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia              import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.factory      import dcube_generator as DCG
from maia.factory      import partition_dist_tree

from maia.algo.part import localize as LOC

@mark_mpi_test(1)
def test_get_part_data(sub_comm):
  dtree = DCG.dcube_generate(3, 1., [0.,0.,0.], sub_comm)
  tree = partition_dist_tree(dtree, sub_comm)
  zone = I.getZones(tree)[0]
  data = LOC._get_part_data(zone)
  assert len(data) == 8
  assert (data[2] == MT.getGlobalNumbering(zone, 'Cell')[1]).all()
  assert data[6].size == 3*PT.Zone.n_vtx(zone) #Coords

@mark_mpi_test(2)
@pytest.mark.parametrize("reverse", [False, True])
def test_mesh_location(reverse, sub_comm):
  dtree = DCG.dcube_generate(3, 1., [0.,0.,0.], sub_comm)
  if sub_comm.Get_rank() == 0:
    zone_to_parts = {'Base/zone' : [.5]}
    tgt_clouds = [(np.array([.6,.9,0]), np.array([2], pdm_gnum_dtype)), (np.array([1.6,.9,0]), np.array([4], pdm_gnum_dtype))]
  else:
    zone_to_parts = {'Base/zone' : [.25, .25]}
    tgt_clouds = [(np.array([.6,.9,0, .6,.9,.8, 1.2, 0.1, 0.1]), np.array([1,3,5], pdm_gnum_dtype))]
  tree = partition_dist_tree(dtree, sub_comm, zone_to_parts=zone_to_parts)
  src_parts = [LOC._get_part_data(zone) for zone in I.getZones(tree)]

  if reverse:
    tgt_data, src_data =  LOC._mesh_location(src_parts, [], sub_comm, reverse)
    assert all([data['elt_pts_inside_idx'].sum() == 0 for data in src_data])
    assert all([data['elt_pts_inside_idx'].size-1 == PT.Zone.n_cell(part) for data,part in zip(src_data, I.getZones(tree))])
    assert all([data['points_gnum'].size == 0 for data in src_data])
  else:
    tgt_data = LOC._mesh_location(src_parts, [], sub_comm, reverse)
  assert tgt_data == []

  if reverse:
    tgt_data, src_data = LOC._mesh_location(src_parts, tgt_clouds, sub_comm, reverse)
  else:
    tgt_data = LOC._mesh_location(src_parts, tgt_clouds, sub_comm, reverse)

  if sub_comm.rank == 0:
    expected_tgt_data = [{'located_ids' : [0], 'unlocated_ids' : [], 'location' : [4]},
                         {'located_ids' : [], 'unlocated_ids' : [0], 'location' : []}]
    expected_src_data = [{'elt_pts_inside_idx' : [0,0,0,0,1], 'points_gnum' : [3]}]
  if sub_comm.rank == 1:
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


@mark_mpi_test(3)
def test_localize_points(sub_comm):
  dtree_src = DCG.dcube_generate(5, 1., [0.,0.,0.], sub_comm)
  dtree_tgt = DCG.dcube_generate(4, 1., [.4,0.,0.], sub_comm)
  tree_src = partition_dist_tree(dtree_src, sub_comm)
  tree_tgt = partition_dist_tree(dtree_tgt, sub_comm)

  tree_src_back = I.copyTree(tree_src)
  LOC.localize_points(tree_src, tree_tgt, 'CellCenter', sub_comm)
  assert PT.is_same_tree(tree_src_back, tree_src)
  tgt_zone = I.getZones(tree_tgt)[0]
  loc_node = I.getNodeFromNameAndType(tgt_zone, 'Localization', 'DiscreteData_t')
  assert loc_node is not None and PT.Subset.GridLocation(loc_node) == 'CellCenter'

  if sub_comm.rank == 0:
    expected_src_id = np.array([3,19,51,36,16,4,8,11,52])
  elif sub_comm.rank == 1:
    expected_src_id = np.array([15,59,63,56,32,23,47,40,64])
  elif sub_comm.rank == 2:
    expected_src_id = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1])

  assert (I.getNodeFromName1(loc_node, 'SrcId')[1] == expected_src_id).all()
