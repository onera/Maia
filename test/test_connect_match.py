import pytest
import pytest_parallel

import numpy as np
import os
import maia.pytree as PT

from maia import io      as MIO
from maia import factory as MF
from maia import utils   as MU

import maia.algo.dist as dist_algo
import maia.algo.part as part_algo

@pytest_parallel.mark.parallel([2])
def test_single_block(comm):

  # > Create dist tree
  dist_tree    = MF.generate_dist_block(10, "Poly", comm, origin=[0,0,0])

  # > This algorithm works on partitioned trees
  part_tree = MF.partition_dist_tree(dist_tree, comm)

  # > Partioning procduce one matching gc
  gc  = PT.get_node_from_label(part_tree, 'GridConnectivity_t')
  # > Test setup -- copy this jn as a bc (without pl donor) to test connect match
  bc = PT.deep_copy(gc)
  PT.set_name(bc, 'ToMatch')
  PT.set_label(bc, 'BC_t')
  PT.set_value(bc, 'FamilySpecified')
  PT.new_node('FamilyName', 'FamilyName_t', 'JN', parent=bc)
  PT.rm_children_from_name(bc, 'GridConnectivityType')
  PT.rm_children_from_name(bc, 'PointListDonor')
  PT.add_child(PT.get_node_from_label(part_tree, 'ZoneBC_t'), bc)

  base = PT.get_all_CGNSBase_t(part_tree)[0]
  PT.new_Family('JN', parent=base)

  # > Connect match
  part_algo.connect_match_from_family(part_tree, ['JN'], comm,
                                      match_type = ['FaceCenter'], rel_tol=1.e-5)

  #PLDonor are well recovered
  new_gc = PT.get_nodes_from_label(part_tree, 'GridConnectivity_t')[-1]
  for name in ['PointList', 'PointListDonor', 'GridConnectivityType', 'GridLocation']:
    assert (PT.get_node_from_name(gc, name)[1] == PT.get_node_from_name(new_gc, name)[1]).all()


@pytest_parallel.mark.parallel([1])
def test_two_blocks(comm):

  mesh_file = os.path.join(MU.test_utils.mesh_dir, 'S_twoblocks.yaml')
  dist_treeS = MIO.file_to_dist_tree(mesh_file, comm)

  # > Input is structured, so convert it to an unstructured tree
  dist_tree = dist_algo.convert_s_to_ngon(dist_treeS, comm)

  # > This algorithm works on partitioned trees
  part_tree = MF.partition_dist_tree(dist_tree, comm)

  # > Backup GridConnectivity for verification
  large_zone = PT.get_nodes_from_name(part_tree, "Large*")[0]
  small_zone = PT.get_nodes_from_name(part_tree, "Small*")[0]
  large_jn = PT.get_node_from_label(large_zone, 'GridConnectivity_t')
  small_jn = PT.get_node_from_label(small_zone, 'GridConnectivity_t')
  PT.rm_nodes_from_label(part_tree, 'ZoneGridConnectivity_t')

  # > Test setup -- Create BC
  large_bc = PT.deep_copy(large_jn)
  PT.set_name(large_bc, 'ToMatch')
  PT.set_label(large_bc, 'BC_t')
  PT.set_value(large_bc, 'FamilySpecified')
  PT.new_node('FamilyName', 'FamilyName_t', 'LargeJN', parent=large_bc)
  PT.rm_children_from_name(large_bc, 'GridConnectivityType')
  PT.rm_children_from_name(large_bc, 'PointListDonor')
  PT.add_child(PT.get_node_from_label(large_zone, 'ZoneBC_t'), large_bc)

  small_bc = PT.deep_copy(small_jn)
  PT.set_name(small_bc, 'ToMatch')
  PT.set_label(small_bc, 'BC_t')
  PT.set_value(small_bc, 'FamilySpecified')
  PT.new_node('FamilyName', 'FamilyName_t', 'SmallJN', parent=small_bc)
  PT.rm_children_from_name(small_bc, 'GridConnectivityType')
  PT.rm_children_from_name(small_bc, 'PointListDonor')
  PT.add_child(PT.get_node_from_label(small_zone, 'ZoneBC_t'), small_bc)

  bc = PT.get_node_from_name(part_tree, 'Front')
  PT.new_node('FamilyName', 'FamilyName_t', 'OtherFamily', parent=bc)


  # > Extra family can be present
  part_algo.connect_match_from_family(part_tree, ['LargeJN', 'SmallJN', 'OtherFamily'], comm,
                                      match_type = ['FaceCenter'], rel_tol=1.e-5)

  # > Check (order can differ)
  new_large_jn = PT.get_node_from_label(large_zone, 'GridConnectivity_t')
  new_small_jn = PT.get_node_from_label(small_zone, 'GridConnectivity_t')
  assert (np.sort(PT.get_node_from_name(new_large_jn, 'PointList')[1]) == \
          np.sort(PT.get_node_from_name(large_jn, 'PointList')[1])).all()
  assert (np.sort(PT.get_node_from_name(new_small_jn, 'PointList')[1]) == \
          np.sort(PT.get_node_from_name(small_jn, 'PointList')[1])).all()
  assert (PT.get_node_from_name(new_small_jn, 'PointList')[1] == \
          PT.get_node_from_name(new_large_jn, 'PointListDonor')[1]).all()
  assert (PT.get_node_from_name(new_small_jn, 'PointListDonor')[1] == \
          PT.get_node_from_name(new_large_jn, 'PointList')[1]).all()
