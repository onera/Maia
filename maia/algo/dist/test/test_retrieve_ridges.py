import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree as PT

import maia.algo.dist.retrieve_ridges as RR

from   maia.utils            import par_utils
from   maia.utils            import test_utils as TU
from   maia.utils.test_utils import mesh_dir

from   maia import npy_pdm_gnum_dtype as pdm_dtype
dtype = 'I4' if pdm_dtype == np.int32 else 'I8'


def test_replace_bc_identifiers():
  yt = """
  Zone Zone_t [[11, 10, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZoneBC ZoneBC_t:
      BC1 BC_t "BCWall":
        FamilyName FamilyName_t "FAM1":
      BC2 BC_t "BCWall":
        FamilyName FamilyName_t "FAM2":
      BC3 BC_t "BCWall":
        FamilyName FamilyName_t "FAM1":
      BC4 BC_t "BCWall":
  """
  tree = PT.yaml.to_cgns_tree(yt)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  # maia.io.write_tree(tree, 'out.cgns', comm)

  bc_identifiers = [["BC1"], ["BC3"]]
  rplcd_bc_identifiers = RR.replace_bc_identifiers(tree, bc_identifiers)
  assert bc_identifiers == rplcd_bc_identifiers
  
  bc_identifiers = ["FAM2", "FAM1", ["BC_4"]]
  rplcd_bc_identifiers = RR.replace_bc_identifiers(tree, bc_identifiers)
  assert [["BC2"], ["BC1", "BC3"], ["BC_4"]] == rplcd_bc_identifiers

  bc_identifiers = [["BC1", "BC2"], ["BC3"]]
  rplcd_bc_identifiers = RR.replace_bc_identifiers(tree, bc_identifiers)
  assert [["BC1", "BC2"], ["BC3"]] == rplcd_bc_identifiers

  bc_identifiers = [["BC1"], "FAM1"]
  with pytest.raises(ValueError):
    rplcd_bc_identifiers = RR.replace_bc_identifiers(tree, bc_identifiers)

  bc_identifiers = [1, "FAM1"]
  with pytest.raises(ValueError):
    rplcd_bc_identifiers = RR.replace_bc_identifiers(tree, bc_identifiers)

  bc_identifiers = ["FAM1", "FAM3"]
  with pytest.raises(ValueError):
    rplcd_bc_identifiers = RR.replace_bc_identifiers(tree, bc_identifiers)


@pytest_parallel.mark.parallel(3)
def test_share_parent_bc_info(comm):
  dedges_partial_distrib = [np.array([ 0,  7, 16]),
                            np.array([ 7, 10, 16]),
                            np.array([10, 16, 16])][comm.rank]
  dgroup_edge_idx        = [np.array([0, 3, 4, 7]),
                            np.array([0, 0, 2, 3]),
                            np.array([0, 1, 2, 6])][comm.rank]
  dgroup_edge            = [np.array([2, 5, 7, 4, 1, 3, 6]),
                            np.array([8, 10, 9]),
                            np.array([11, 13, 12, 14, 15, 16])][comm.rank]
  dridge_face_group_idx  = [np.array([0, 1, 2, 3, 5, 6, 7, 8]),
                            np.array([0, 2, 3, 5]),
                            np.array([0, 1, 2, 4, 5, 6, 7])][comm.rank]
  dridge_face_group      = [np.array([1, 2, 2, 1, 2]),
                            np.array([1, 2, 1, 2, 2, 2, 2]),
                            np.array([2, 1, 2, 1, 2, 1, 2, 1])][comm.rank]

  dgroup_edges = [dgroup_edge[dgroup_edge_idx[i]:dgroup_edge_idx[i+1]] for i in range(len(dgroup_edge_idx)-1)]
  parents = RR.share_parent_bc_info(dedges_partial_distrib, dgroup_edges,
                                    dridge_face_group_idx, dridge_face_group, 
                                    comm)

  for result, expected in zip(parents, [np.array([2]), np.array([1,2]), np.array([1])]):
    assert np.array_equal(result, expected)


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize('elmt_t', ["Poly", "HEXA_8"])
def test_find_boundary_edges(comm, elmt_t):
  dist_tree = maia.factory.generate_dist_block(3, elmt_t, comm)

  bcs_identifiers = [["Xmin"], ["Ymin", "Zmax"]]
  new_edge_path = RR.find_boundary_edges(dist_tree, comm,  bcs_identifiers)
  
  maia.io.dist_tree_to_file(dist_tree, 'out.cgns', comm)

  assert new_edge_path==['Base/zone/topo_edge']
  bar_n = PT.get_node_from_path(dist_tree, new_edge_path[0])
  bar_elmt_range = PT.get_child_from_name(bar_n, 'ElementRange')[1]
  expected_elmt_range = np.array([37, 53]) if elmt_t=="Poly" else np.array([33, 49])

  is_edge_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=="EdgeCenter"
  edge_bcs = PT.get_nodes_from_predicate(dist_tree, is_edge_bc)
  assert len(edge_bcs)==3

@pytest_parallel.mark.parallel(2)
def test_extract_elmt_connectivity_from_pl(comm):
  from maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'multi_element.yaml', comm)
  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')

  if comm.rank==0:
    pl = np.array([  1, # TETRA_4.0
                   173, # TRI_3.0
                  ], dtype=pdm_dtype)
    expected_strd = np.array([        4,      3], dtype=pdm_dtype)
    expected_conn = np.array([3,4,46,70, 3,4,46], dtype=pdm_dtype)
  else :
    pl = np.array([ 24, # TETRA_4.0
                    49, # TETRA_4.1
                   216, # TRI_3.0
                  ], dtype=pdm_dtype)
    expected_strd = np.array([          4,           4,      3], dtype=pdm_dtype)
    expected_conn = np.array([50,23,52,51, 19,20,45,69, 8,25,5], dtype=pdm_dtype)

  def check_result(result):
    assert np.array_equal(expected_strd, result[0])
    assert np.array_equal(expected_conn, result[1])

  strd_and_conn = RR.extract_elmt_connectivity_from_pl(dist_zone, pl, comm)
  check_result(strd_and_conn)

  predicate = lambda n: PT.predicate.is_elmt_of_type(n, dim=3)
  with pytest.raises(RuntimeError): # Not TRI Elements so fails cause some pl elements not found
    strd_and_conn = RR.extract_elmt_connectivity_from_pl(dist_zone, pl, comm, 
                                                         elmt_predicate=predicate)

  predicate = lambda n: PT.predicate.is_elmt_of_type(n, cgns_name='TETRA_4')
  with pytest.raises(RuntimeError): # Not TRI Elements so fails cause some pl elements not found
    strd_and_conn = RR.extract_elmt_connectivity_from_pl(dist_zone, pl, comm, 
                                                         elmt_predicate=predicate)

@pytest_parallel.mark.parallel(2)
def test_extract_bcs_from_pl(comm):
  if comm.rank==0:
    yt = f"""
    ZoneBC ZoneBC_t:
      BC1 BC_t "BCWall":
        GridLocation GridLocation_t "EdgeCenter":
        PointList IndexArray_t {dtype} [[12, 10, 11]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 3, 5]:
      BC2 BC_t "BCWall":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t {dtype} [[3, 6]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 2, 3]:
      BC3 BC_t "BCWall":
        GridLocation GridLocation_t "EdgeCenter":
        PointList IndexArray_t {dtype} [[7]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 1, 1]:
      BC4 BC_t "BCWall":
        PointList IndexArray_t {dtype} [[1, 8, 10]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 3, 6]:
    """
  else:
    yt = f"""
    ZoneBC ZoneBC_t:
      BC1 BC_t "BCWall":
        GridLocation GridLocation_t "EdgeCenter":
        PointList IndexArray_t {dtype} [[7, 8]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [3, 5, 5]:
      BC2 BC_t "BCWall":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t {dtype} [[4]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [2, 3, 3]:
      BC3 BC_t "BCWall":
        GridLocation GridLocation_t "EdgeCenter":
        PointList IndexArray_t {dtype} [[]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [1, 1, 1]:
      BC4 BC_t "BCWall":
        PointList IndexArray_t {dtype} [[3, 7, 9]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 3, 6]:
    """
  def check_result(zone_bc_n, expected_pls):
    for bc_name, expected_pl in expected_pls.items():
      bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
      bc_pl = PT.Subset.getPatch(bc_n)[1][0]
      bc_distri = PT.maia.getDistribution(bc_n, "Index")[1]
      expected_distri = par_utils.dn_to_distribution(expected_pl.size, comm)
      assert np.array_equal(bc_pl    , expected_pl)
      assert np.array_equal(bc_distri, expected_distri)

  tree = PT.yaml.to_cgns_tree(yt)
  zone_bc_n = PT.get_child_from_label(tree, "ZoneBC_t")

  pl = [np.array([12, 7], dtype=pdm_dtype),
        np.array([10]  , dtype=pdm_dtype)][comm.rank]
  distri_pl = par_utils.dn_to_distribution(pl.size, comm)

  expected_pls = {"BC1":[np.array([1, 3], dtype=pdm_dtype),
                         np.array([2]   , dtype=pdm_dtype)][comm.rank],
                  "BC3":[np.array([2]   , dtype=pdm_dtype),
                         np.array([]    , dtype=pdm_dtype)][comm.rank],
                  "BC4":[np.array([3]   , dtype=pdm_dtype),
                         np.array([2]   , dtype=pdm_dtype)][comm.rank]}
  extract_zone_bc_n = RR.extract_bcs_from_pl(zone_bc_n, pl, distri_pl, comm)
  check_result(extract_zone_bc_n, expected_pls)

  expected_pls = {"BC1":[np.array([1, 3], dtype=pdm_dtype),
                         np.array([2]   , dtype=pdm_dtype)][comm.rank],
                  "BC3":[np.array([2]   , dtype=pdm_dtype),
                         np.array([]    , dtype=pdm_dtype)][comm.rank]}
  extract_zone_bc_n = RR.extract_bcs_from_pl(zone_bc_n, pl, distri_pl, comm, 
    bc_predicate=lambda n:PT.predicate.is_bc_of_loc(n, 'EdgeCenter'))
  check_result(extract_zone_bc_n, expected_pls)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('root_t', ["CGNSTree_t", "Zone_t"])
def test_extract_edges(comm, root_t):
  from maia.utils.test_utils import mesh_dir
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'axisym_mesh.yaml', comm)

  is_edge_bc = lambda n: PT.predicate.is_bc_of_loc(n, 'EdgeCenter')
  point_list = [PT.Subset.getPatch(n)[1][0] for n in PT.get_nodes_from_predicate(dist_tree, is_edge_bc)[2:6]]

  if root_t=='Zone_t':
    _dist_tree = PT.get_node_from_label(dist_tree, 'Zone_t')
    domain_pl = np.concatenate(point_list)
  else:
    _dist_tree = dist_tree
    domain_pl = {'cube/zone': np.concatenate(point_list)}

  edge_dist_tree = RR.extract_edges(_dist_tree, domain_pl, comm)

  if root_t=='Zone_t':
    _edge_dist_zone = edge_dist_tree
  else:
    _edge_dist_zone = PT.get_node_from_label(edge_dist_tree, 'Zone_t')

  assert PT.Zone.n_vtx (_edge_dist_zone)==9
  assert PT.Zone.n_cell(_edge_dist_zone)==8
  assert len(PT.get_nodes_from_label(_edge_dist_zone, 'BC_t'))==4
