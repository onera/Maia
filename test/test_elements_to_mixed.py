import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import os
import numpy as np

import maia.pytree        as PT

import maia.io
import maia.utils.test_utils as TU

from maia.algo.dist import convert_elements_to_mixed

@mark_mpi_test([1,4])
def test_elements_to_mixed(sub_comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, sub_comm)

  # > As simple as it looks
  convert_elements_to_mixed(dist_tree, sub_comm)
  
  zone = PT.get_node_from_label(dist_tree,'Zone_t')
  
  assert len(PT.Zone.get_ordered_elements(zone)) == 1
  
  mixed_node = PT.Zone.get_ordered_elements(zone)[0]

  assert PT.Element.CGNSName(mixed_node) == 'MIXED'
  
  assert PT.get_child_from_name(mixed_node, 'ElementConnectivity')
  assert PT.get_child_from_name(mixed_node, 'ElementStartOffset')

  # > Some non-regression checks
  assert np.all(PT.get_value(PT.get_child_from_name(mixed_node, 'ElementRange')) == [1,1500])

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_M6Wing_mixed.cgns'), sub_comm)

