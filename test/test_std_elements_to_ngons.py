import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import os
import numpy as np

import Converter.Internal as I

import cmaia
import maia.io
import maia.utils.test_utils as TU

from maia.algo.dist import convert_std_elements_to_ngons, generate_ngon_from_std_elements

@pytest.mark.skipif(not cmaia.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
@mark_mpi_test([1,4])
def test_std_elements_to_ngons_maia(sub_comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, sub_comm)

  # > As simple as it looks
  convert_std_elements_to_ngons(dist_tree, sub_comm)

  # > Old elements are cleaned up
  assert len(I.getNodesFromType(dist_tree, 'Elements_t')) == 2
  # > Poly sections appear
  ngon_node  = I.getNodeFromName(dist_tree, 'NGON_n')
  nface_node = I.getNodeFromName(dist_tree, 'NFACE_n')
  assert ngon_node  is not None
  assert nface_node is not None

  assert I.getNodeFromName(ngon_node, 'ParentElements')
  assert I.getNodeFromName(ngon_node, 'ParentElementsPosition')

  # > Some non-regression checks
  assert np.all(I.getVal(I.getNodeFromName(ngon_node , 'ElementRange')) == [1,2694])
  assert np.all(I.getVal(I.getNodeFromName(nface_node, 'ElementRange')) == [2695,3990])

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing_ngon.cgns'), sub_comm)

@mark_mpi_test([1,4])
def test_std_elements_to_ngons_pdm(sub_comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, sub_comm)

  # > As simple as it looks
  generate_ngon_from_std_elements(dist_tree, sub_comm)

  # > Old elements are cleaned up
  assert len(I.getNodesFromType(dist_tree, 'Elements_t')) == 2
  ngon_node  = I.getNodeFromName(dist_tree, 'NGonElements')
  nface_node = I.getNodeFromName(dist_tree, 'NFaceElements')
  assert ngon_node is not None
  assert nface_node is not None

  assert np.all(I.getVal(I.getNodeFromName(ngon_node , 'ElementRange')) == [1,2694])
  assert np.all(I.getVal(I.getNodeFromName(nface_node, 'ElementRange')) == [2695,3990])

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing_ngon.hdf'), sub_comm)

