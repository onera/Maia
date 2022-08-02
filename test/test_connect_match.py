import pytest
from   pytest_mpi_check._decorator import mark_mpi_test

import numpy as np
import os
import Converter.Internal as I
import maia.pytree as PT

from maia import io      as MIO
from maia import factory as MF
from maia import utils   as MU

import maia.algo.dist as dist_algo
import maia.algo.part as part_algo

@mark_mpi_test([2])
def test_single_block(sub_comm):

  # > Create dist tree
  dist_tree    = MF.generate_dist_block(10, "Poly", sub_comm, origin=[0,0,0])

  # > This algorithm works on partitioned trees
  part_tree = MF.partition_dist_tree(dist_tree, sub_comm)

  # > Partioning procduce one matching gc
  gc  = I.getNodeFromType(part_tree, 'GridConnectivity_t')
  # > Test setup -- copy this jn as a bc (without pl donor) to test connect match
  bc = I.copyTree(gc)
  I.setName(bc, 'ToMatch')
  I.setType(bc, 'BC_t')
  I.setValue(bc, 'FamilySpecified')
  I.createNode('FamilyName', 'FamilyName_t', 'JN', parent=bc)
  PT.rm_children_from_name(bc, 'GridConnectivityType')
  PT.rm_children_from_name(bc, 'PointListDonor')
  I._addChild(I.getNodeFromType(part_tree, 'ZoneBC_t'), bc)

  base = I.getBases(part_tree)[0]
  I.newFamily('JN', parent=base)

  # > Connect match
  part_algo.connect_match_from_family(part_tree, ['JN'], sub_comm,
                                      match_type = ['FaceCenter'], rel_tol=1.e-5)

  #PLDonor are well recovered
  new_gc = PT.get_nodes_from_label(part_tree, 'GridConnectivity_t')[-1]
  for name in ['PointList', 'PointListDonor', 'GridConnectivityType', 'GridLocation']:
    assert (I.getNodeFromName(gc, name)[1] == I.getNodeFromName(new_gc, name)[1]).all()


@mark_mpi_test([1])
def test_two_blocks(sub_comm):

  mesh_file = os.path.join(MU.test_utils.mesh_dir, 'S_twoblocks.yaml')
  dist_treeS = MIO.file_to_dist_tree(mesh_file, sub_comm)

  # > Input is structured, so convert it to an unstructured tree
  dist_tree = dist_algo.convert_s_to_ngon(dist_treeS, sub_comm)

  # > This algorithm works on partitioned trees
  part_tree = MF.partition_dist_tree(dist_tree, sub_comm)

  # > Backup GridConnectivity for verification
  large_zone = PT.get_nodes_from_name(part_tree, "Large*")[0]
  small_zone = PT.get_nodes_from_name(part_tree, "Small*")[0]
  large_jn = I.getNodeFromType(large_zone, 'GridConnectivity_t')
  small_jn = I.getNodeFromType(small_zone, 'GridConnectivity_t')
  PT.rm_nodes_from_label(part_tree, 'ZoneGridConnectivity_t')

  # > Test setup -- Create BC
  large_bc = I.copyTree(large_jn)
  I.setName(large_bc, 'ToMatch')
  I.setType(large_bc, 'BC_t')
  I.setValue(large_bc, 'FamilySpecified')
  I.createNode('FamilyName', 'FamilyName_t', 'LargeJN', parent=large_bc)
  PT.rm_children_from_name(large_bc, 'GridConnectivityType')
  PT.rm_children_from_name(large_bc, 'PointListDonor')
  I._addChild(I.getNodeFromType(large_zone, 'ZoneBC_t'), large_bc)

  small_bc = I.copyTree(small_jn)
  I.setName(small_bc, 'ToMatch')
  I.setType(small_bc, 'BC_t')
  I.setValue(small_bc, 'FamilySpecified')
  I.createNode('FamilyName', 'FamilyName_t', 'SmallJN', parent=small_bc)
  PT.rm_children_from_name(small_bc, 'GridConnectivityType')
  PT.rm_children_from_name(small_bc, 'PointListDonor')
  I._addChild(I.getNodeFromType(small_zone, 'ZoneBC_t'), small_bc)

  bc = I.getNodeFromName(part_tree, 'Front')
  I.createNode('FamilyName', 'FamilyName_t', 'OtherFamily', parent=bc)


  # > Extra family can be present
  part_algo.connect_match_from_family(part_tree, ['LargeJN', 'SmallJN', 'OtherFamily'], sub_comm,
                                      match_type = ['FaceCenter'], rel_tol=1.e-5)

  # > Check (order can differ)
  new_large_jn = I.getNodeFromType(large_zone, 'GridConnectivity_t')
  new_small_jn = I.getNodeFromType(small_zone, 'GridConnectivity_t')
  assert (np.sort(I.getNodeFromName(new_large_jn, 'PointList')[1]) == \
          np.sort(I.getNodeFromName(large_jn, 'PointList')[1])).all()
  assert (np.sort(I.getNodeFromName(new_small_jn, 'PointList')[1]) == \
          np.sort(I.getNodeFromName(small_jn, 'PointList')[1])).all()
  assert (I.getNodeFromName(new_small_jn, 'PointList')[1] == \
          I.getNodeFromName(new_large_jn, 'PointListDonor')[1]).all()
  assert (I.getNodeFromName(new_small_jn, 'PointListDonor')[1] == \
          I.getNodeFromName(new_large_jn, 'PointList')[1]).all()
