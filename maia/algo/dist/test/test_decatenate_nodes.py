import pytest
import pytest_parallel
import os

import maia
import maia.pytree        as PT

from maia.algo.dist import concat_nodes as GN
from maia.algo.dist import decatenate_nodes as DN

import maia.utils.test_utils as TU

@pytest.mark.parametrize("specified", [True, False])
@pytest_parallel.mark.parallel(3)
def test_decatenate_patch(specified, comm):
  mesh_path = os.path.join(TU.mesh_dir,'flat_plate_3d.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_path, comm)

  def tag_fam_in_bcs(dist_tree, bc_names, family_name):
    for bc_name in bc_names:
      bc_n = PT.get_node_from_name_and_label(dist_tree, bc_name, 'BC_t')
      PT.new_FamilyName(family_name, parent=bc_n)
  
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in range(0, 5)], 'WALL')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in range(5, 9)], 'SYM')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in range(9,10)], 'FARFIELD')
  tag_fam_in_bcs(dist_tree, [f'ridge.{i}'   for i in range(0,20)], 'RIDGE')

  dist_tree_cp = PT.deep_copy(dist_tree)

  if specified:
    families = ['WALL','FARFIELD','RIDGE']
    GN.concatenate_subset_from_families(dist_tree, comm, families)
  else:
    families = ['WALL', 'SYM', 'FARFIELD','RIDGE']
    GN.concatenate_subset_from_families(dist_tree, comm)

  DN.decatenate_subset_from_predicate(dist_tree, comm, families)

  assert PT.is_same_tree(dist_tree, dist_tree_cp)