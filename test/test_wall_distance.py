import pytest
import pytest_parallel
import os
import numpy as np

import maia.pytree        as PT

import maia

mesh_dir = maia.utils.test_utils.mesh_dir 
ref_dir  = os.path.join(os.path.dirname(__file__), 'references')

@pytest.mark.parametrize("method", ["cloud"])
@pytest_parallel.mark.parallel([1,4])
def test_wall_distance_S(method, comm, write_output):

  mesh_file = os.path.join(mesh_dir, 'S_twoblocks.yaml')
  ref_file  = os.path.join(ref_dir,  'S_twoblocks_walldist.yaml')

  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  # Partitioning
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, graph_part_tool='ptscotch')

  # Wall distance computation (BC already have wall type in the CGNS file)
  maia.algo.part.compute_wall_distance(part_tree, comm, method=method)

  # Save file and compare
  maia.transfer.part_tree_to_dist_tree_all(dist_tree, part_tree, comm)

  if write_output:
    out_dir = maia.utils.test_utils.create_pytest_output_dir(comm)
    maia.io.write_trees(part_tree, os.path.join(out_dir, 'parttree_out.hdf'), comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'result.hdf'), comm)

  # Compare to reference solution
  refence_solution = maia.io.file_to_dist_tree(ref_file, comm)
  for d_base in PT.iter_all_CGNSBase_t(dist_tree):
    for d_zone in PT.iter_all_Zone_t(d_base):
      zone_path = '/'.join([PT.get_name(d_base), PT.get_name(d_zone)])
      ref_wall_dist = PT.get_node_from_path(refence_solution, zone_path + '/WallDistance')
      assert maia.pytree.is_same_tree(ref_wall_dist, PT.get_child_from_name(d_zone, 'WallDistance'), 
          type_tol=True, abs_tol=1E-14)

wall_dist_methods = ["cloud"]
if maia.pdma_enabled:
  wall_dist_methods.append("propagation")
@pytest.mark.parametrize("method", wall_dist_methods)
@pytest_parallel.mark.parallel([1, 3])
def test_wall_distance_U(method, comm, write_output):

  mesh_file       = os.path.join(mesh_dir, 'U_ATB_45.yaml')
  ref_file        = os.path.join(ref_dir,  'U_ATB_45_walldist.yaml')
  ref_file_perio  = os.path.join(ref_dir,  'U_ATB_45_walldist_perio.yaml')

  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  #Let WALL family be autodetected by setting its type to wall:
  wall_family = PT.get_node_from_name(dist_tree, 'WALL', depth=2)
  family_bc = PT.get_child_from_name(wall_family, 'FamilyBC')
  PT.set_value(family_bc, 'BCWall')

  # Partitioning
  zone_to_parts = maia.factory.partitioning.compute_regular_weights(dist_tree, comm, 3)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts)

  # Wall distance computation
  perio = (method == 'cloud') #Perio not available for propagation method
  maia.algo.part.compute_wall_distance(part_tree, comm, method=method, perio=perio)

  # Save file and compare
  maia.transfer.part_tree_to_dist_tree_only_labels(dist_tree, part_tree, ['FlowSolution_t'], comm)

  if write_output:
    out_dir = maia.utils.test_utils.create_pytest_output_dir(comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'result.hdf'), comm)

  # Compare to reference solution
  if method == 'cloud':
    refence_solution = maia.io.file_to_dist_tree(ref_file_perio, comm)
  elif method == 'propagation':
    refence_solution = maia.io.file_to_dist_tree(ref_file, comm)
  for d_base in PT.iter_all_CGNSBase_t(dist_tree):
    for d_zone in PT.iter_all_Zone_t(d_base):
      zone_path = '/'.join([PT.get_name(d_base), PT.get_name(d_zone)])
      ref_wall_dist = PT.get_node_from_path(refence_solution, zone_path + '/WallDistance')
      assert maia.pytree.is_same_tree(ref_wall_dist, PT.get_child_from_name(d_zone, 'WallDistance'),
          type_tol=True, abs_tol=1E-14)

