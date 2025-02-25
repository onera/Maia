import os
import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT

import maia

from maia.utils     import test_utils as TU
from maia.algo.dist import closest_points as CLO

@pytest_parallel.mark.parallel(1)
def test_closestpoint_mdom(comm):
  yaml_path = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  tree_src = maia.io.file_to_dist_tree(yaml_path, comm)
  tree_tgt = maia.factory.generate_dist_block(5, 'Poly', comm, length=4)
  maia.algo.transform_affine(tree_tgt, translation=np.array([13.25, 2.25, 0.25]))

  src_doms = PT.get_all_Zone_t(tree_src)
  tgt_doms = PT.get_all_Zone_t(tree_tgt)

  result, result_inv = CLO._find_closest_points(src_doms, tgt_doms, 'Vertex', 'Vertex', comm, reverse=True)
  
  _result = result[0]
  dom1_idx = np.where(_result['domain'] == 1)[0]
  dom2_idx = np.where(_result['domain'] == 2)[0] 
  assert dom1_idx.size == 87 #Carefull, 25 resulting points are on the interface, result can change
  assert dom2_idx.size == 38
  # Gnum should have been reshifted
  assert _result['closest_src_gnum'][dom1_idx].max() <= PT.Zone.n_vtx(src_doms[0])
  assert _result['closest_src_gnum'][dom2_idx].max() <= PT.Zone.n_vtx(src_doms[1])

@pytest_parallel.mark.parallel(2)
def test_closest_points_lowdim(comm):
  tree_src = maia.factory.generate_dist_sphere(5, 'TRI_3', comm)
  tree_tgt = maia.factory.generate_dist_block(6, 'BAR_2', comm, origin=[1.,0,-0.5], length=[0., 0., 1.])

  CLO.find_closest_points(tree_src, tree_tgt, 'CellCenter', comm)

  tgt_zone = PT.get_all_Zone_t(tree_tgt)[0]
  clo_node = PT.get_node_from_name_and_label(tgt_zone, 'ClosestPoint', 'DiscreteData_t')
  assert clo_node is not None and PT.Subset.GridLocation(clo_node) == 'CellCenter'
  assert PT.get_value(PT.get_child_from_name(clo_node, 'DomainList')) == "Base/zone"

  # Check result on dist tree to not rely on partitioning
  if comm.rank == 0:
    expected_dsrc_id = np.array([133,150,149])
  elif comm.rank == 1:
    expected_dsrc_id = np.array([131,128])
  assert (PT.get_node_from_name(tree_tgt, 'SrcId')[1] == expected_dsrc_id).all()

  
@pytest_parallel.mark.parallel(3)
def test_closest_points(comm):
  tree_src = maia.factory.generate_dist_block(5, 'Poly', comm, origin=[0.,0.,0.])
  tree_tgt = maia.factory.generate_dist_block(4, 'Poly', comm, origin=[.4,-0.01,-0.01])

  maia.algo.find_closest_points(tree_src, tree_tgt, 'CellCenter', comm)

  tgt_zone = PT.get_all_Zone_t(tree_tgt)[0]
  clo_node = PT.get_node_from_name_and_label(tgt_zone, 'ClosestPoint', 'DiscreteData_t')
  assert clo_node is not None and PT.Subset.GridLocation(clo_node) == 'CellCenter'
  assert PT.get_value(PT.get_child_from_name(clo_node, 'DomainList')) == "Base/zone"

  # Check result on dist tree to not rely on partitioning
  if comm.rank == 0:
    expected_dsrc_id = np.array([3,4,4,7,8,8,15,16,16])
  elif comm.rank == 1:
    expected_dsrc_id = np.array([19,20,20,23,24,24,31,32,32])
  elif comm.rank == 2:
    expected_dsrc_id = np.array([51,52,52,55,56,56,63,64,64])

  assert (PT.get_node_from_name(tree_tgt, 'SrcId')[1] == expected_dsrc_id).all()
