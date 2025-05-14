import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import fix_orient as FO

@pytest.mark.parametrize('with_nface', [False, True])
@pytest_parallel.mark.parallel(2)
def test_enforce_boundary_pe_left(with_nface, comm):
  tree = maia.factory.generate_dist_block(5, 'Poly', comm)
  zone = PT.find_node_from_label(tree, 'Zone_t')

  # Prepare case by swapping some bnd cells in PE
  ngon = PT.Zone.NGonNode(zone)
  face_vtx = MT.Element.connectivity(ngon)
  pe = PT.get_np_value(PT.find_child_from_name(ngon, 'ParentElements'))
  
  if comm.rank == 0:
    pe[0,1] = pe[0,0]
    pe[0,0] = 0
    pe[1,1] = pe[1,0]
    pe[1,0] = 0
  else:
    pe[-1,1] = pe[-1,0]
    pe[-1,0] = 0
    pe[-2,1] = pe[-2,0]
    pe[-2,0] = 0

  if with_nface:
    maia.algo.pe_to_nface(tree, comm)
    
  FO.enforce_boundary_pe_left(tree, comm)

  if with_nface:
    nface = PT.Zone.NFaceNode(zone)
    cell_face = MT.Element.connectivity(nface)
  
  if comm.rank == 0:
    assert (pe[0:2, :] == np.array([[241, 0], [242,0]])).all()
    assert (face_vtx[0] == [2,7,6,1]).all()
    assert (face_vtx[1] == [3,8,7,2]).all()
    if with_nface:
      assert (cell_face[0] == [-177,-97,-17,1,81,161]).all()
      assert (cell_face[1] == [-181,-113,-18,2,97,165]).all()

  if comm.rank == 1:
    assert (pe[-2:, :] == np.array([[288, 0], [304,0]])).all()
    assert (face_vtx[-1] == [99,100,125,124]).all()
    assert (face_vtx[-2] == [74,75,100,99]).all()
    if with_nface:
      assert (cell_face[15] == [239,-64,48,140,156,223]).all()
      assert (cell_face[31] == [240,64,80,144,160,224]).all()


@pytest_parallel.mark.parallel(2)
def test_enforce_boundary_pe_left_early_return(comm):
  tree = maia.factory.generate_dist_block(5, 'Poly', comm)
  tree_bck = PT.deep_copy(tree)
  FO.enforce_boundary_pe_left(tree, comm)
  assert PT.is_same_tree(tree, tree_bck)


@pytest_parallel.mark.parallel(2)
def test_fix_normal_orientation(comm):
  tree = maia.factory.generate_dist_block(5, 'Poly', comm)
  maia.algo.pe_to_nface(tree, comm)
  tree_bck = PT.deep_copy(tree)
  zone = PT.find_node_from_label(tree, 'Zone_t')

  # Prepare case by swapping some bnd cells in PE
  ngon = PT.Zone.NGonNode(zone)
  face_vtx = MT.Element.connectivity(ngon)

  # Swap some faces normal to create test (some are internal, some are external)
  face_vtx[0] = face_vtx[0][::-1] # Bnd face
  face_vtx[1] = face_vtx[1][::-1] # Bnd face
  face_vtx[-2] = face_vtx[-2][::-1] # Bnd face
  face_vtx[-1] = face_vtx[-1][::-1] # Bnd face

  FO.fix_normal_orientation(tree, comm)

  # Faces connectivity is swapped back to respect pe/nface orientation
  assert PT.is_same_tree(tree, tree_bck)