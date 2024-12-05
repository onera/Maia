import pytest
import pytest_parallel
import os

import maia
import maia.pytree as PT

from maia.algo.dist import concat_nodes as GN
from maia.algo.part import deconcatenate_nodes as DN

import maia.utils.test_utils as TU

@pytest.mark.parametrize("specified", [True, False])
@pytest_parallel.mark.parallel(3)
def test_part_deconcatenate_patch(specified, comm):
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
    GN.concatenate_subsets_from_families(dist_tree, comm, families)
  else:
    GN.concatenate_subsets_from_families(dist_tree, comm)

  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, data_transfer='ALL')
  to_copy1 = ["CGNSBase_t", "Zone_t", "ZoneBC_t", "BC_t", "Descriptor_t"]
  maia.transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, to_copy1, comm)

  if specified:
    DN.deconcatenate_subsets_from_families(part_tree, comm, families)
  else:
    DN.deconcatenate_subsets_from_families(part_tree, comm)

  dist_tree = maia.factory.recover_dist_tree(part_tree, comm, data_transfer='ALL')
  assert PT.is_same_tree(dist_tree, dist_tree_cp, type_tol=True)