import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

import maia.algo.dist.extract_part as EP

from   maia.utils            import par_utils
from   maia.utils.test_utils import mesh_dir

from   maia import npy_pdm_gnum_dtype as pdm_dtype
dtype = 'I4' if pdm_dtype == np.int32 else 'I8'


@pytest_parallel.mark.parallel(2)
def test_extract_elmt_connectivity_from_pl(comm):
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'multi_element.yaml', comm)
  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')

  if comm.rank==0:
    pl = np.array([  1, # TETRA_4.0
                   173, # TRI_3.0
                  ], dtype=pdm_dtype)
    expected_strd = np.array([0,      4,      7], dtype=pdm_dtype)
    expected_conn = np.array([3,4,46,70, 3,4,46], dtype=pdm_dtype)
  else :
    pl = np.array([ 24, # TETRA_4.0
                    49, # TETRA_4.1
                   216, # TRI_3.0
                  ], dtype=pdm_dtype)
    expected_strd = np.array([0,        4,           8,     11], dtype=pdm_dtype)
    expected_conn = np.array([50,23,52,51, 19,20,45,69, 8,25,5], dtype=pdm_dtype)


  all_elmt_nodes = PT.get_nodes_from_label(dist_zone, 'Elements_t')
  strd_and_conn = EP.extract_elmt_connectivity_from_pl(dist_zone, all_elmt_nodes, pl, comm)
  assert np.array_equal(expected_strd, strd_and_conn[0])
  assert np.array_equal(expected_conn, strd_and_conn[1])

  elmt_3d_nodes = PT.Zone.get_ordered_elements_per_dim(dist_zone)[3]
  with pytest.raises(RuntimeError): # Not TRI Elements so fails cause some pl elements not found
    strd_and_conn = EP.extract_elmt_connectivity_from_pl(dist_zone, elmt_3d_nodes, pl, comm)

  tetra_elmt_nodes = PT.get_nodes_from_predicate(dist_zone, PT.pred.is_elmt_of_type('TETRA_4'))
  with pytest.raises(RuntimeError): # Not TRI Elements so fails cause some pl elements not found
    strd_and_conn = EP.extract_elmt_connectivity_from_pl(dist_zone, tetra_elmt_nodes, pl, comm)

@pytest_parallel.mark.parallel(2)
def test_extract_bcs_from_pl(comm):
  if comm.rank==0:
    yt = f"""
    ZoneBC ZoneBC_t:
      BC1 BC_t "BCWall":
        GridLocation GridLocation_t "EdgeCenter":
        PointList IndexArray_t [[12, 10, 11]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 3, 5]:
      BC2 BC_t "BCWall":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[3, 6]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 2, 3]:
      BC3 BC_t "BCWall":
        GridLocation GridLocation_t "EdgeCenter":
        PointList IndexArray_t [[7]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 1, 1]:
      BC4 BC_t "BCWall":
        PointList IndexArray_t [[1, 8, 10]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 3, 6]:
    """
  else:
    yt = f"""
    ZoneBC ZoneBC_t:
      BC1 BC_t "BCWall":
        GridLocation GridLocation_t "EdgeCenter":
        PointList IndexArray_t [[7, 8]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [3, 5, 5]:
      BC2 BC_t "BCWall":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[4]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [2, 3, 3]:
      BC3 BC_t "BCWall":
        GridLocation GridLocation_t "EdgeCenter":
        PointList IndexArray_t [[]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [1, 1, 1]:
      BC4 BC_t "BCWall":
        PointList IndexArray_t [[3, 7, 9]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {dtype} [0, 3, 6]:
    """
  def check_result(zone_bc_n, expected_pls):
    for bc_name, expected_pl in expected_pls.items():
      bc_n = PT.get_child_from_name(zone_bc_n, bc_name)
      bc_pl = PT.Subset.getPatch(bc_n)[1][0]
      bc_distri = MT.distribution_value(bc_n, "Index")
      expected_distri = par_utils.dn_to_distribution(expected_pl.size, comm)
      assert np.array_equal(bc_pl    , expected_pl)
      assert np.array_equal(bc_distri, expected_distri)
      assert bc_pl.dtype == expected_pl.dtype and bc_distri.dtype == pdm_dtype

  zone_bc_n = PT.yaml.to_node(yt)

  pl = [np.array([12, 7], np.int32),
        np.array([10]  ,  np.int32)][comm.rank]
  distri_pl = par_utils.dn_to_distribution(pl.size, comm)

  expected_pls = {"BC1":[np.array([1, 3], dtype=np.int32),
                         np.array([2]   , dtype=np.int32)][comm.rank],
                  "BC3":[np.array([2]   , dtype=np.int32),
                         np.array([]    , dtype=np.int32)][comm.rank],
                  "BC4":[np.array([3]   , dtype=np.int32),
                         np.array([2]   , dtype=np.int32)][comm.rank]}
  extract_zone_bc_n = EP.extract_bcs_from_pl(zone_bc_n, pl, distri_pl, comm)
  check_result(extract_zone_bc_n, expected_pls)

  expected_pls = {"BC1":[np.array([1, 3], dtype=np.int32),
                         np.array([2]   , dtype=np.int32)][comm.rank],
                  "BC3":[np.array([2]   , dtype=np.int32),
                         np.array([]    , dtype=np.int32)][comm.rank]}
  extract_zone_bc_n = EP.extract_bcs_from_pl(zone_bc_n, pl, distri_pl, comm, 
    bc_predicate=PT.pred.is_bc_of_loc('EdgeCenter'))
  check_result(extract_zone_bc_n, expected_pls)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('root_t', ["CGNSTree_t", "Zone_t"])
def test_extract_edges(comm, root_t):
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'axisym_mesh.yaml', comm)

  point_list = [PT.Subset.getPatch(n)[1][0] \
    for n in PT.get_nodes_from_predicate(dist_tree, PT.pred.is_bc_of_loc('EdgeCenter'))[2:6]]

  if root_t=='Zone_t':
    _dist_tree = PT.get_node_from_label(dist_tree, 'Zone_t')
    domain_pl = np.concatenate(point_list)
  else:
    _dist_tree = dist_tree
    domain_pl = {'cube/zone': np.concatenate(point_list)}

  edge_dist_tree = EP.extract_edges(_dist_tree, domain_pl, comm)

  if root_t=='Zone_t':
    _edge_dist_zone = edge_dist_tree
  else:
    _edge_dist_zone = PT.get_node_from_label(edge_dist_tree, 'Zone_t')

  assert PT.Zone.n_vtx (_edge_dist_zone)==9
  assert PT.Zone.n_cell(_edge_dist_zone)==8
  assert len(PT.get_nodes_from_label(_edge_dist_zone, 'BC_t'))==4
