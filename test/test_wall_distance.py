import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import numpy as np

import Converter.Internal as I
import Converter.PyTree   as C
import maia.pytree        as PT

import maia

mesh_dir = maia.utils.test_utils.mesh_dir 
ref_dir  = os.path.join(os.path.dirname(__file__), 'references')

@pytest.mark.parametrize("method", ["cloud"])
@mark_mpi_test([1,4])
def test_wall_distance_S(method, sub_comm, write_output):

  mesh_file = os.path.join(mesh_dir, 'S_twoblocks.yaml')
  ref_file  = os.path.join(ref_dir,  'S_twoblocks_walldist.yaml')

  dist_tree = maia.io.file_to_dist_tree(mesh_file, sub_comm)

  # Families are not present in the tree, we need to add it
  base = I.getBases(dist_tree)[0]
  fam = I.newFamily('WALL', parent=base)
  I.newFamilyBC(parent=fam)
  for bc in PT.iter_nodes_from_label(dist_tree, 'BC_t'):
    if I.getValue(bc) == 'BCWall':
      I.setValue(bc, 'FamilySpecified')
      I.createChild(bc, 'FamilyName', 'FamilyName_t', 'WALL')

  # Partitioning
  part_tree = maia.factory.partition_dist_tree(dist_tree, sub_comm, graph_part_tool='ptscotch')

  # Wall distance computation
  families = ['WALL']
  maia.algo.part.compute_wall_distance(part_tree, sub_comm, method=method, families=families)

  # Save file and compare
  maia.transfer.part_tree_to_dist_tree_all(dist_tree, part_tree, sub_comm)

  if write_output:
    out_dir = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    C.convertPyTree2File(part_tree, os.path.join(out_dir, f'parttree_out_{sub_comm.Get_rank()}.hdf'))
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'result.hdf'), sub_comm)

  # Compare to reference solution
  refence_solution = maia.io.file_to_dist_tree(ref_file, sub_comm)
  for d_base in I.getBases(dist_tree):
    for d_zone in I.getZones(d_base):
      zone_path = '/'.join([I.getName(d_base), I.getName(d_zone)])
      ref_wall_dist = I.getNodeFromPath(refence_solution, zone_path + '/WallDistance')
      assert maia.pytree.is_same_tree(ref_wall_dist, PT.get_child_from_name(d_zone, 'WallDistance'), 
          type_tol=True, abs_tol=1E-14)

wall_dist_methods = ["cloud"]
if maia.pdma_enabled:
  wall_dist_methods.append("propagation")
@pytest.mark.parametrize("method", wall_dist_methods)
@mark_mpi_test([1, 3])
def test_wall_distance_U(method, sub_comm, write_output):

  mesh_file = os.path.join(mesh_dir, 'U_ATB_45.yaml')
  ref_file  = os.path.join(ref_dir,     'U_ATB_45_walldist.yaml')

  dist_tree = maia.io.file_to_dist_tree(mesh_file, sub_comm)

  #Let WALL family be autodetected by setting its type to wall:
  wall_family = PT.get_node_from_name(dist_tree, 'WALL', depth=2)
  family_bc = PT.get_child_from_name(wall_family, 'FamilyBC')
  I.setValue(family_bc, 'BCWall')

  # Partitioning
  zone_to_parts = maia.factory.partitioning.compute_regular_weights(dist_tree, sub_comm, 3)
  part_tree = maia.factory.partition_dist_tree(dist_tree, sub_comm, zone_to_parts=zone_to_parts)

  # Wall distance computation
  maia.algo.part.compute_wall_distance(part_tree, sub_comm, method=method)

  # Save file and compare
  maia.transfer.part_tree_to_dist_tree_only_labels(dist_tree, part_tree, ['FlowSolution_t'], sub_comm)

  if write_output:
    out_dir = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'result.hdf'), sub_comm)

  # Compare to reference solution
  refence_solution = maia.io.file_to_dist_tree(ref_file, sub_comm)
  for d_base in I.getBases(dist_tree):
    for d_zone in I.getZones(d_base):
      zone_path = '/'.join([I.getName(d_base), I.getName(d_zone)])
      ref_wall_dist = I.getNodeFromPath(refence_solution, zone_path + '/WallDistance')
      assert maia.pytree.is_same_tree(ref_wall_dist, PT.get_child_from_name(d_zone, 'WallDistance'),
          type_tol=True, abs_tol=1E-14)

