import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import numpy as np

import Converter.Internal     as I
import maia.sids.Internal_ext as IE

from maia.cgns_io import cgns_io_tree as IOT

from maia.sids  import pytree     as PT
from maia.utils import test_utils as TU
from maia.utils import parse_yaml_cgns

from maia.connectivity import vertex_list as VL

ref_dir = os.path.join(os.path.dirname(__file__), 'references')

@mark_mpi_test([1,3])
def test_jn_vertexlist(sub_comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'U_Naca0012_multizone.yaml')
  ref_file  = os.path.join(ref_dir,     'U_Naca0012_multizone_vl.yaml')

  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  n_jn_ini = len(I.getNodesFromType(dist_tree, 'GridConnectivity_t'))

  # Generate a GridConnectivity node with GridLocation==Vertex for each face GridConnectivity
  # found in the tree
  VL.generate_jns_vertex_list(dist_tree, sub_comm)

  assert len(I.getNodesFromType(dist_tree, 'GridConnectivity_t')) == 2*n_jn_ini

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    IOT.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'U_Naca0012_multizone_with_vertexlist.hdf'), sub_comm)

  # Compare to reference solution
  with open(ref_file, 'r') as f:
    reference_tree = parse_yaml_cgns.to_cgns_tree(f)
  for ref_gc in I.getNodesFromType(reference_tree, 'GridConnectivity_t'):
    gc = I.getNodeFromName(dist_tree, I.getName(ref_gc))
    distri = IE.getDistribution(gc, 'Index')[1]
    for ref_node in I.getChildren(ref_gc):
      node = I.getNodeFromName1(gc, I.getName(ref_node))
      if I.getName(node) in ['PointList', 'PointListDonor']:
        ref_node[1] = np.array([ref_node[1][0][distri[0]:distri[1]]])  # Extract distributed array
      assert PT.is_same_node(ref_node, node)

