import pytest
import pytest_parallel
import os
import numpy as np

import maia.pytree        as PT

import cmaia
import maia.io
import maia.utils.test_utils as TU

from maia.algo.dist import elements_to_ngons, generate_ngon_from_std_elements

@pytest.mark.skipif(not cmaia.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
@pytest_parallel.mark.parallel([1,4])
def test_elements_to_ngons_maia(comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  # > As simple as it looks
  elements_to_ngons(dist_tree, comm)

  # > Old elements are cleaned up
  assert len(PT.get_nodes_from_label(dist_tree, 'Elements_t')) == 2
  # > Poly sections appear
  ngon_node  = PT.request_node_from_name(dist_tree, 'NGON_n')
  nface_node = PT.request_node_from_name(dist_tree, 'NFACE_n')

  assert PT.get_child_from_name(ngon_node, 'ParentElements')
  assert PT.get_child_from_name(ngon_node, 'ParentElementsPosition')

  # > Some non-regression checks
  assert np.all(PT.get_value(PT.get_child_from_name(ngon_node , 'ElementRange')) == [1,2694])
  assert np.all(PT.get_value(PT.get_child_from_name(nface_node, 'ElementRange')) == [2695,3990])

  if write_output:
    out_dir = TU.create_pytest_output_dir(comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing_ngon.cgns'), comm)

@pytest_parallel.mark.parallel([1,4])
def test_elements_to_ngons_pdm(comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  # > As simple as it looks
  generate_ngon_from_std_elements(dist_tree, comm)

  # > Old elements are cleaned up
  assert len(PT.get_nodes_from_label(dist_tree, 'Elements_t')) == 2
  ngon_node  = PT.request_node_from_name(dist_tree, 'NGonElements')
  nface_node = PT.request_node_from_name(dist_tree, 'NFaceElements')

  assert np.all(PT.get_value(PT.get_child_from_name(ngon_node , 'ElementRange')) == [1,2694])
  assert np.all(PT.get_value(PT.get_child_from_name(nface_node, 'ElementRange')) == [2695,3990])

  if write_output:
    out_dir = TU.create_pytest_output_dir(comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing_ngon.hdf'), comm)

