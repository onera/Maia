import pytest
import pytest_parallel

import numpy as np
import os
import maia.pytree as PT

import maia
from maia import io      as MIO
from maia import utils   as MU

import maia.algo.dist as dist_algo

@pytest_parallel.mark.parallel([1])
def test_two_blocks(comm):

  mesh_file = os.path.join(MU.test_utils.mesh_dir, 'S_twoblocks.yaml')
  dist_treeS = MIO.file_to_dist_tree(mesh_file, comm)

  # > Input is structured, so convert it to an unstructured tree
  dist_tree = dist_algo.convert_s_to_ngon(dist_treeS, comm)

  # > Backup GridConnectivity for verification
  large_zone = PT.get_nodes_from_name(dist_tree, "Large*")[0]
  small_zone = PT.get_nodes_from_name(dist_tree, "Small*")[0]
  large_jn = PT.get_node_from_label(large_zone, 'GridConnectivity_t')
  small_jn = PT.get_node_from_label(small_zone, 'GridConnectivity_t')
  PT.rm_nodes_from_label(dist_tree, 'ZoneGridConnectivity_t')

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

  bc = PT.get_node_from_name(dist_tree, 'Front')
  PT.new_node('FamilyName', 'FamilyName_t', 'OtherFamily', parent=bc)


  # > Extra family can be present
  dist_algo.connect_1to1_families(dist_tree, ('LargeJN', 'SmallJN'), comm)

  # > Check
  new_large_jn = PT.get_node_from_label(large_zone, 'GridConnectivity_t')
  new_small_jn = PT.get_node_from_label(small_zone, 'GridConnectivity_t')
  assert (PT.get_node_from_name(new_large_jn, 'PointList')[1] == \
          PT.get_node_from_name(large_jn, 'PointList')[1]).all()
  assert (PT.get_node_from_name(new_small_jn, 'PointList')[1] == \
          PT.get_node_from_name(small_jn, 'PointList')[1]).all()
  assert (PT.get_node_from_name(new_small_jn, 'PointList')[1] == \
          PT.get_node_from_name(new_large_jn, 'PointListDonor')[1]).all()
  assert (PT.get_node_from_name(new_small_jn, 'PointListDonor')[1] == \
          PT.get_node_from_name(new_large_jn, 'PointList')[1]).all()

@pytest_parallel.mark.parallel([3])
def test_perio_elements(comm):
  tree = maia.factory.generate_dist_block(11, 'TETRA_4', comm) #Nb : use odd number to have conform faces
  maia.io.dist_tree_to_file(tree, 'out.cgns', comm)

  xmin = PT.get_node_from_name(tree, 'Xmin')
  PT.new_child(xmin, 'FamilyName', 'FamilyName_t', 'matchA')
  xmax = PT.get_node_from_name(tree, 'Xmax')
  PT.new_child(xmax, 'FamilyName', 'FamilyName_t', 'matchB')

  periodic = {'translation' : np.array([1.0, 0, 0], np.float32)}
  dist_algo.connect_1to1_families(tree, ('matchA', 'matchB'), comm, periodic=periodic)

  # Use rank 0 to perform geometric test
  dist_algo.redistribute_tree(tree, 'gather', comm)
  if comm.Get_rank() == 0:
    zone = PT.get_all_Zone_t(tree)[0]
    face_center = maia.algo.part.compute_face_center(zone)
    xmin = PT.get_node_from_predicates(zone, ['Xmin_0', 'PointList'])[1] - PT.Zone.n_cell(zone) - 1
    xmax = PT.get_node_from_predicates(zone, ['Xmax_0', 'PointList'])[1] - PT.Zone.n_cell(zone) - 1
    assert np.allclose(face_center[3*xmin+0], face_center[3*xmax+0]-1)
    assert np.allclose(face_center[3*xmin+1], face_center[3*xmax+1])
    assert np.allclose(face_center[3*xmin+2], face_center[3*xmax+2])
