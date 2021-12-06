import pytest
from   pytest_mpi_check._decorator import mark_mpi_test

import numpy as np
import os
import Converter.Internal as I

from maia.cgns_io             import cgns_io_tree                    as IOT
from maia.partitioning        import part                            as PPA
from maia.geometry            import connect_match                   as CMA
from maia.generate            import dcube_generator                 as DCG
from maia.transform.dist_tree import convert_s_to_u
from maia.utils               import test_utils                      as TU


@mark_mpi_test([2])
def test_single_block(sub_comm):

  # > Create dist tree
  dist_tree    = DCG.dcube_generate(10, 1., origin=[0,0,0], comm=sub_comm)

  # > This algorithm works on partitioned trees
  part_tree = PPA.partitioning(dist_tree, sub_comm)

  # > Partioning procduce one matching gc
  gc  = I.getNodeFromType(part_tree, 'GridConnectivity_t')
  # > Test setup -- copy this jn as a bc (without pl donor) to test connect match
  bc = I.copyTree(gc)
  I.setName(bc, 'ToMatch')
  I.setType(bc, 'BC_t')
  I.setValue(bc, 'FamilySpecified')
  I.createNode('FamilyName', 'FamilyName_t', 'JN', parent=bc)
  I._rmNodesByName(bc, 'GridConnectivityType')
  I._rmNodesByName(bc, 'PointListDonor')
  I._addChild(I.getNodeFromType(part_tree, 'ZoneBC_t'), bc)

  base = I.getBases(part_tree)[0]
  I.newFamily('JN', parent=base)

  # > Connect match
  CMA.connect_match_from_family(part_tree, ['JN'], sub_comm,
                                match_type = ['FaceCenter'], rel_tol=1.e-5)

  #PLDonor are well recovered
  new_gc = I.getNodesFromType(part_tree, 'GridConnectivity_t')[-1]
  for name in ['PointList', 'PointListDonor', 'GridConnectivityType', 'GridLocation']:
    assert (I.getNodeFromName(gc, name)[1] == I.getNodeFromName(new_gc, name)[1]).all()


@mark_mpi_test([1])
def test_two_blocks(sub_comm):

  mesh_file = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dist_treeS = IOT.file_to_dist_tree(mesh_file, sub_comm)

  # > Input is structured, so convert it to an unstructured tree
  dist_tree = convert_s_to_u.convert_s_to_u(dist_treeS, sub_comm)

  # > This algorithm works on partitioned trees
  part_tree = PPA.partitioning(dist_tree, sub_comm)

  # > Backup GridConnectivity for verification
  large_zone = I.getNodesFromName(part_tree, "Large*")[0]
  small_zone = I.getNodesFromName(part_tree, "Small*")[0]
  large_jn = I.getNodeFromType(large_zone, 'GridConnectivity_t')
  small_jn = I.getNodeFromType(small_zone, 'GridConnectivity_t')
  I._rmNodesByType(part_tree, 'ZoneGridConnectivity_t')

  # > Test setup -- Create BC
  large_bc = I.copyTree(large_jn)
  I.setName(large_bc, 'ToMatch')
  I.setType(large_bc, 'BC_t')
  I.setValue(large_bc, 'FamilySpecified')
  I.createNode('FamilyName', 'FamilyName_t', 'LargeJN', parent=large_bc)
  I._rmNodesByName(large_bc, 'GridConnectivityType')
  I._rmNodesByName(large_bc, 'PointListDonor')
  I._addChild(I.getNodeFromType(large_zone, 'ZoneBC_t'), large_bc)

  small_bc = I.copyTree(small_jn)
  I.setName(small_bc, 'ToMatch')
  I.setType(small_bc, 'BC_t')
  I.setValue(small_bc, 'FamilySpecified')
  I.createNode('FamilyName', 'FamilyName_t', 'SmallJN', parent=small_bc)
  I._rmNodesByName(small_bc, 'GridConnectivityType')
  I._rmNodesByName(small_bc, 'PointListDonor')
  I._addChild(I.getNodeFromType(small_zone, 'ZoneBC_t'), small_bc)

  bc = I.getNodeFromName(part_tree, 'Front')
  I.createNode('FamilyName', 'FamilyName_t', 'OtherFamily', parent=bc)


  # > Extra family can be present
  CMA.connect_match_from_family(part_tree, ['LargeJN', 'SmallJN', 'OtherFamily'], sub_comm,
                                match_type = ['FaceCenter'], rel_tol=1.e-5)

  # > Check (order can differ)
  new_large_jn = I.getNodeFromType(large_zone, 'GridConnectivity_t')
  new_small_jn = I.getNodeFromType(small_zone, 'GridConnectivity_t')
  assert (np.sort(I.getNodeFromName(new_large_jn, 'PointList')[1]) == \
          np.sort(I.getNodeFromName(large_jn, 'PointList')[1])).all()
  assert (np.sort(I.getNodeFromName(new_small_jn, 'PointList')[1]) == \
          np.sort(I.getNodeFromName(small_jn, 'PointList')[1])).all()
  assert (I.getNodeFromName(new_small_jn, 'PointList')[1] == \
          I.getNodeFromName(new_large_jn, 'PointListDonor')[1]).all()
  assert (I.getNodeFromName(new_small_jn, 'PointListDonor')[1] == \
          I.getNodeFromName(new_large_jn, 'PointList')[1]).all()
