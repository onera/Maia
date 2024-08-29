import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree as PT

import maia.algo.dist.retrieve_ridges as RR

from   maia.utils.test_utils import mesh_dir
from maia.utils import test_utils as TU
from   maia                              import npy_pdm_gnum_dtype as pdm_dtype


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
def test_find_boundary_edges(comm):
  dist_tree = maia.factory.generate_dist_block(3, 'Poly', comm)

  bcs_identifiers = [["Xmin"], ["Ymin", "Zmax"]]
  new_edge_path = RR.find_boundary_edges(dist_tree, comm,  bcs_identifiers)
  
  maia.io.dist_tree_to_file(dist_tree, 'out.cgns', comm)

  assert new_edge_path==['Base/zone/BAR_2']
  bar_n = PT.get_node_from_path(dist_tree, new_edge_path[0])
  bar_elmt_range = PT.get_child_from_name(bar_n, 'ElementRange')[1]
  assert np.array_equal(bar_elmt_range, np.array([44, 60]))

  is_edge_bc = lambda n: PT.get_label(n)=='BC_t' and PT.Subset.GridLocation(n)=="EdgeCenter"
  edge_bcs = PT.get_nodes_from_predicate(dist_tree, is_edge_bc)
  assert len(edge_bcs)==3