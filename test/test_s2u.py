import pytest
import pytest_parallel
import os

import maia.pytree        as PT

import maia.io      as MIO
import maia.utils   as MU
import maia.factory as MF

from maia.algo.dist import convert_s_to_u, convert_s_to_ngon

ref_dir = os.path.join(os.path.dirname(__file__), 'references')

@pytest.mark.parametrize("subset_output_loc", ["FaceCenter", "Vertex"])
@pytest_parallel.mark.parallel([1,3])
def test_s2u(comm, subset_output_loc, write_output):
  mesh_file = os.path.join(MU.test_utils.mesh_dir,  'S_twoblocks.yaml')
  ref_file  = os.path.join(ref_dir,     f'U_twoblocks_{subset_output_loc.lower()}_subset_s2u.yaml')

  dist_treeS = MIO.file_to_dist_tree(mesh_file, comm)

  subset_loc = {key: subset_output_loc for key in ['BC_t', 'GC_t']}
  dist_treeU = convert_s_to_u(dist_treeS, 'NGON_n', comm, subset_loc)

  for zone in PT.iter_all_Zone_t(dist_treeU):
    assert PT.Zone.Type(zone) == 'Unstructured'
    for node in PT.get_nodes_from_label(zone, 'BC_t') + PT.get_nodes_from_label(zone, 'GridConnectivity_t'):
      assert PT.Subset.GridLocation(node) == subset_output_loc

  # Compare to reference
  ref_tree = MIO.file_to_dist_tree(ref_file, comm)
  for zone in PT.iter_all_Zone_t(dist_treeU):
    ref_zone = PT.get_node_from_name(ref_tree, PT.get_name(zone), depth=2)
    for node_name in ["ZoneBC", "ZoneGridConnectivity"]:
      assert PT.is_same_tree(PT.get_child_from_name(zone, node_name), PT.get_child_from_name(ref_zone, node_name))

  if write_output:
    out_dir = MU.test_utils.create_pytest_output_dir(comm)
    MIO.dist_tree_to_file(dist_treeU, os.path.join(out_dir, 'tree_U.hdf'), comm)

@pytest_parallel.mark.parallel(2)
def test_s2u_hybrid(comm, write_output):
  mesh_file = os.path.join(MU.test_utils.mesh_dir, 'H_elt_and_s.yaml')
  dist_tree = MIO.file_to_dist_tree(mesh_file, comm)

  dist_tree_u = convert_s_to_ngon(dist_tree, comm)

  for zone in PT.iter_all_Zone_t(dist_tree_u):
    assert PT.Zone.Type(zone) == 'Unstructured'
    for pl in PT.iter_nodes_from_label(zone, 'IndexArray_t'):
      assert PT.get_value(pl).shape[0] == 1 #All PLs should be (1,N)

  if write_output:
    out_dir = MU.test_utils.create_pytest_output_dir(comm)
    MIO.dist_tree_to_file(dist_tree_u, os.path.join(out_dir, 'tree_U.hdf'), comm)

@pytest_parallel.mark.parallel([1])
def test_s2u_withdata(comm, write_output):
  mesh_file = os.path.join(MU.test_utils.mesh_dir,  'S_twoblocks.yaml')

  dist_treeS = MIO.file_to_dist_tree(mesh_file, comm)

  # Use only small zone for simplicity
  PT.rm_nodes_from_name(dist_treeS, 'Large')
  PT.rm_nodes_from_label(dist_treeS, 'ZoneGridConnectivity_t')

  # Add some BCDataFace data
  bc_right = PT.yaml.parse_yaml_cgns.to_node(
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
  PT.rm_nodes_from_name(dist_treeS, 'Right')
  PT.add_child(PT.get_node_from_label(dist_treeS, 'ZoneBC_t'), \
      MF.full_to_dist.distribute_pl_node(bc_right, comm))

  dist_treeU = convert_s_to_ngon(dist_treeS, comm)

  # Some checks
  bc_right_u = PT.get_node_from_name(dist_treeU, 'Right')
  assert PT.get_node_from_path(bc_right_u, 'WholeDSFace/GridLocation') is None
  assert PT.Subset.GridLocation(PT.get_node_from_name(bc_right_u, 'SubDSFace')) == 'FaceCenter'
  assert PT.Subset.GridLocation(PT.get_node_from_name(bc_right_u, 'SubDSVtx')) == 'Vertex'
  assert PT.get_node_from_path(bc_right_u, 'WholeDSFace/PointList') is None
  assert (PT.get_node_from_path(bc_right_u, 'SubDSFace/PointList')[1] == [225,226,279,280,333,334,387,388]).all()
  assert (PT.get_node_from_path(bc_right_u, 'SubDSVtx/PointList')[1] == [1,2,64,65,127,128,190,191,253,254]).all()
  for bcds in PT.get_children_from_label(bc_right_u, 'BCDataSet_t'): #Data should be the same
    bcds_s = PT.get_node_from_name(bc_right, PT.get_name(bcds))
    assert (PT.get_node_from_name(bcds, 'lid')[1] == PT.get_node_from_name(bcds_s, 'lid')[1]).all()

  if write_output:
    out_dir = MU.test_utils.create_pytest_output_dir(comm)
    MIO.dist_tree_to_file(dist_treeU, os.path.join(out_dir, 'tree_U.hdf'), comm)

