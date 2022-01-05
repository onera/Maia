import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os

import Converter.Internal as I

import maia
from   maia.sids             import sids
from   maia.sids             import pytree       as PT
from   maia.utils            import test_utils   as TU
from   maia.cgns_io          import cgns_io_tree as IOT

from maia.transform.dist_tree import convert_s_to_u as S2U

ref_dir = os.path.join(os.path.dirname(__file__), 'references')

@pytest.mark.parametrize("subset_output_loc", ["FaceCenter", "Vertex"])
@mark_mpi_test([1,3])
def test_s2u(sub_comm, subset_output_loc, write_output):
  mesh_file = os.path.join(TU.mesh_dir,  'S_twoblocks.yaml')
  ref_file  = os.path.join(ref_dir,     f'U_twoblocks_{subset_output_loc.lower()}_subset_s2u.yaml')

  dist_treeS = IOT.file_to_dist_tree(mesh_file, sub_comm)

  dist_treeU = S2U.convert_s_to_u(dist_treeS, sub_comm, \
      bc_output_loc=subset_output_loc, gc_output_loc=subset_output_loc)

  for zone in I.getZones(dist_treeU):
    assert sids.Zone.Type(zone) == 'Unstructured'
    for node in I.getNodesFromType(zone, 'BC_t') + I.getNodesFromType(zone, 'GridConnectivity_t'):
      assert sids.GridLocation(node) == subset_output_loc

  # Compare to reference
  ref_tree = IOT.file_to_dist_tree(ref_file, sub_comm)
  for zone in I.getZones(dist_treeU):
    ref_zone = I.getNodeFromName2(ref_tree, I.getName(zone))
    for node_name in ["ZoneBC", "ZoneGridConnectivity"]:
      assert PT.is_same_tree(I.getNodeFromName(zone, node_name), I.getNodeFromName(ref_zone, node_name))

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    IOT.dist_tree_to_file(dist_treeU, os.path.join(out_dir, 'tree_U.hdf'), sub_comm)

