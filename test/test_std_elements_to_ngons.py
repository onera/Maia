import pytest
from pytest_mpi_check._decorator import mark_mpi_test
from maia.utils import test_utils as TU
import os
import numpy as np

import Converter.Internal as I
from maia.cgns_io import cgns_io_tree as IOT

import maia.transform
from maia.connectivity       import generate_ngon_from_std_elements as FTH

@pytest.mark.skipif(not maia.transform.cpp20_enabled, reason="Require ENABLE_CPP20 compilation flag")
@mark_mpi_test([1,4])
def test_std_elements_to_ngons_maia(sub_comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = maia.cgns_io.cgns_io_tree.file_to_dist_tree(mesh_file, sub_comm)

  # > As simple as it looks
  maia.transform.std_elements_to_ngons(dist_tree, sub_comm)

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
    IOT.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing_ngon.cgns'), sub_comm)

@mark_mpi_test([1,4])
def test_std_elements_to_ngons_pdm(sub_comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  # > As simple as it looks
  FTH.generate_ngon_from_std_elements(dist_tree, sub_comm)

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
    IOT.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing_ngon.hdf'), sub_comm)

