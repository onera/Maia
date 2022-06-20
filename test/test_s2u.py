import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os

import Converter.Internal as I
import maia.pytree        as PT

import maia.io      as MIO
import maia.utils   as MU
import maia.factory as MF

from maia.algo.dist import convert_s_to_u, convert_s_to_ngon

ref_dir = os.path.join(os.path.dirname(__file__), 'references')

@pytest.mark.parametrize("subset_output_loc", ["FaceCenter", "Vertex"])
@mark_mpi_test([1,3])
def test_s2u(sub_comm, subset_output_loc, write_output):
  mesh_file = os.path.join(MU.test_utils.mesh_dir,  'S_twoblocks.yaml')
  ref_file  = os.path.join(ref_dir,     f'U_twoblocks_{subset_output_loc.lower()}_subset_s2u.yaml')

  dist_treeS = MIO.file_to_dist_tree(mesh_file, sub_comm)

  subset_loc = {key: subset_output_loc for key in ['BC_t', 'GC_t']}
  dist_treeU = convert_s_to_u(dist_treeS, 'NGON_n', sub_comm, subset_loc)

  for zone in I.getZones(dist_treeU):
    assert PT.Zone.Type(zone) == 'Unstructured'
    for node in I.getNodesFromType(zone, 'BC_t') + I.getNodesFromType(zone, 'GridConnectivity_t'):
      assert PT.Subset.GridLocation(node) == subset_output_loc

  # Compare to reference
  ref_tree = MIO.file_to_dist_tree(ref_file, sub_comm)
  for zone in I.getZones(dist_treeU):
    ref_zone = I.getNodeFromName2(ref_tree, I.getName(zone))
    for node_name in ["ZoneBC", "ZoneGridConnectivity"]:
      assert PT.is_same_tree(I.getNodeFromName(zone, node_name), I.getNodeFromName(ref_zone, node_name))

  if write_output:
    out_dir = MU.test_utils.create_pytest_output_dir(sub_comm)
    MIO.dist_tree_to_file(dist_treeU, os.path.join(out_dir, 'tree_U.hdf'), sub_comm)

@mark_mpi_test([1])
def test_s2u_withdata(sub_comm, write_output):
  mesh_file = os.path.join(MU.test_utils.mesh_dir,  'S_twoblocks.yaml')

  dist_treeS = MIO.file_to_dist_tree(mesh_file, sub_comm)

  # Use only small zone for simplicity
  I._rmNodesByName(dist_treeS, 'Large')
  I._rmNodesByType(dist_treeS, 'ZoneGridConnectivity_t')

  # Add some BCDataFace data
  bc_right = MU.yaml.parse_yaml_cgns.to_node(
    """
    Right BC_t 'BCInflow':
      PointRange IndexRange_t I4 [[1, 6], [1, 1], [1, 4]]:
      GridLocation GridLocation_t 'JFaceCenter':
      WholeDSFace BCDataSet_t "Null":
        NeumannData BCData_t:
          lid DataArray_t I4 [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]:
      SubDSFace BCDataSet_t "Null":
        GridLocation GridLocation_t "JFaceCenter":
        PointRange IndexRange_t I4 [[1,2], [1,1], [1,4]]:
        DirichletData BCData_t:
          lid DataArray_t I4 [1,2,3,4,5,6,7,8]:
      SubDSVtx BCDataSet_t "Null":
        GridLocation GridLocation_t "Vertex":
        PointRange IndexRange_t I4 [[1, 2], [1, 1], [1, 5]]:
        DirichletData BCData_t:
          lid DataArray_t I4 [1,2,3,4,5,6,7,8,9,10]:
    """)
  I._rmNodesByName(dist_treeS, 'Right')
  I._addChild(I.getNodeFromType(dist_treeS, 'ZoneBC_t'), \
      MF.full_to_dist.distribute_pl_node(bc_right, sub_comm))

  dist_treeU = convert_s_to_ngon(dist_treeS, sub_comm)

  # Some checks
  bc_right_u = I.getNodeFromName(dist_treeU, 'Right')
  assert I.getNodeFromPath(bc_right_u, 'WholeDSFace/GridLocation') is None
  assert PT.Subset.GridLocation(I.getNodeFromName(bc_right_u, 'SubDSFace')) == 'FaceCenter'
  assert PT.Subset.GridLocation(I.getNodeFromName(bc_right_u, 'SubDSVtx')) == 'Vertex'
  assert I.getNodeFromPath(bc_right_u, 'WholeDSFace/PointList') is None
  assert (I.getNodeFromPath(bc_right_u, 'SubDSFace/PointList')[1] == [225,226,279,280,333,334,387,388]).all()
  assert (I.getNodeFromPath(bc_right_u, 'SubDSVtx/PointList')[1] == [1,2,64,65,127,128,190,191,253,254]).all()
  for bcds in I.getNodesFromType(bc_right_u, 'BCDataSet_t'): #Data should be the same
    bcds_s = I.getNodeFromName(bc_right, I.getName(bcds))
    assert (I.getNodeFromName(bcds, 'lid')[1] == I.getNodeFromName(bcds_s, 'lid')[1]).all()

  if write_output:
    out_dir = MU.test_utils.create_pytest_output_dir(sub_comm)
    MIO.dist_tree_to_file(dist_treeU, os.path.join(out_dir, 'tree_U.hdf'), sub_comm)

