import pytest
import pytest_parallel

import maia
import maia.pytree as PT

from maia.factory.partitioning import part_bound_orient as PBO

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("dim", [3, 2])
@pytest.mark.parametrize("with_pe", [True, False])
def test_orientation_preserved(with_pe, dim, comm):
  if dim == 2:
    tree  = maia.factory.generate_dist_block(11, 'QUAD_4', comm)
    maia.algo.dist.convert_elements_to_ngon(tree, comm)
  else:
    tree  = maia.factory.generate_dist_block(11, 'Poly', comm)

  ptree  = maia.factory.partition_dist_tree(tree, comm, preserve_orientation=False)
  pzones = PT.get_all_Zone_t(ptree) 
  if not with_pe:
    PT.rm_nodes_from_name(ptree, 'ParentElements')

  # Not preserved, but if comm.size == 1 it is as if it was preserved
  assert PBO.orientation_preserved(pzones, comm) == (comm.size == 1)


  ptree  = maia.factory.partition_dist_tree(tree, comm, preserve_orientation=True)
  pzones = PT.get_all_Zone_t(ptree) 
  if not with_pe:
    PT.rm_nodes_from_name(ptree, 'ParentElements')

  assert PBO.orientation_preserved(pzones, comm) == True

@pytest_parallel.mark.parallel([4])
@pytest.mark.parametrize("with_nface", [True, False])
def test_preserve_orientation(with_nface, comm):
  tree  = maia.factory.generate_dist_block(21, 'Poly', comm)
  ptree = maia.factory.partition_dist_tree(tree, comm, preserve_orientation=False)
  pzones = PT.get_all_Zone_t(ptree) 

  if with_nface:
    PT.rm_nodes_from_name(ptree, 'ParentElements')
  else:
    PT.rm_nodes_from_name(ptree, 'NFaceElements')

  PBO.preserve_orientation(pzones, comm)
  assert PBO.orientation_preserved(pzones, comm) == True

@pytest_parallel.mark.parallel(3)
def test_shallow_preserve_orientation(comm):
  tree  = maia.factory.generate_dist_block(21, 'Poly', comm)
  ptree = maia.factory.partition_dist_tree(tree, comm, preserve_orientation=False)
  pzones = PT.get_all_Zone_t(ptree) 

  pzones_bck = [PT.deep_copy(z) for z in pzones]

  pzones_oriented = PBO.shallow_preserve_orientation(pzones, comm)

  assert PBO.orientation_preserved(pzones_oriented, comm)
  for z_bck, z in zip(pzones_bck, pzones):
    assert PT.is_same_tree(z_bck, z)
