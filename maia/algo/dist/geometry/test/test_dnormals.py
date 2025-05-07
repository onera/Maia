import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree        as PT

from maia.algo.dist.geometry import normals as GEO

@pytest.mark.parametrize('unitary', [False, True])
@pytest_parallel.mark.parallel(3)
def test_compute_face_normal3d_ng(unitary, comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  face_normal = GEO.compute_face_normal(zone, comm, unitary)

  # All face area are 0.25
  coef = 0.25 if not unitary else 1
  if comm.Get_rank() == 0:
    expected_face_normal = coef * np.array([0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
                                            0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
                                            0,0, 1, 0,0, 1, 0,0, 1, 0,0, 1])
  elif comm.Get_rank() == 1:
    expected_face_normal = coef * np.array([-1,0,0, -1,0,0, -1,0,0, -1,0,0,
                                            -1,0,0, -1,0,0, -1,0,0, -1,0,0,
                                             1,0,0,  1,0,0,  1,0,0,  1,0,0])

  if comm.Get_rank() == 2:
    expected_face_normal = coef * np.array([0,-1,0,  0,-1,0,  0,-1,0,  0,-1,0,
                                            0,-1,0,  0,-1,0,  0,-1,0,  0,-1,0,
                                            0, 1,0,  0, 1,0,  0, 1,0,  0, 1,0])

  assert (face_normal == expected_face_normal).all()

@pytest.mark.parametrize('unitary', [False, True])
@pytest_parallel.mark.parallel(1)
def test_compute_face_normal3d_elt(unitary, comm):
  tree = maia.factory.generate_dist_block([3,3,2], 'HEXA_8', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  # Computed only for boundary faces
  face_normal = GEO.compute_face_normal(zone, comm, unitary)

  # All face area are 0.25
  c1 = 0.25 if not unitary else 1
  c2 = 0.50 if not unitary else 1

  expected_face_normal =  np.array([ 0, 0,-c1,   0, 0,-c1,  0, 0,-c1,  0, 0,-c1,
                                     0, 0, c1,   0, 0, c1,  0, 0, c1,  0, 0, c1,
                                    -c2, 0, 0,  -c2, 0, 0,  c2, 0, 0,  c2, 0, 0,
                                     0,-c2, 0,   0,-c2, 0,  0, c2, 0,  0, c2, 0])

  assert (face_normal == expected_face_normal).all()

@pytest.mark.parametrize('elt_kind', ['Poly', 'Standard', 'S'])
@pytest_parallel.mark.parallel(2)
def test_compute_face_normal2d(elt_kind, comm):
  # NB this test compute face normal even if CellDim is 2, because PhyDim is 3
  tree = maia.factory.generate_dist_block([3,3], 'S', comm, length=[2,1])
  # Make a "roof" shape to have different normals
  cz = PT.get_np_value(PT.find_node_from_name(tree, 'CoordinateZ'))
  if comm.rank == 0:
    cz[[1,4]] += 1
  else:
    cz[[2]] += 1

  if elt_kind != 'S':
    maia.algo.dist.convert_s_to_u(tree, elt_kind, comm)

  zone = PT.get_all_Zone_t(tree)[0]
  face_normal = GEO._compute_elements_normal(zone, comm)

  # Both rank have same pair of value, because of geometry
  assert (face_normal == np.array([-0.5,0.,0.5,  0.5,0.,0.5])).all()


@pytest.mark.parametrize('cell_dim', [2,3])
@pytest_parallel.mark.parallel(1)
def test_compute_elements_normal(cell_dim, comm):
  # NB : in this test we just check the placement, not the values
  n_vtx = [4,4,4] if cell_dim == 3 else [4,4]
  base_tree = maia.factory.generate_dist_block(n_vtx, 'S', comm)
  
  # ----- PhyDim = 3 (Face normals) -----
  # > S 
  tree = PT.deep_copy(base_tree)
  zone = PT.get_all_Zone_t(tree)[0]
  GEO.compute_elements_normal(zone, comm)

  if cell_dim == 3:
    assert PT.get_node_from_name(zone, 'Geometry_2d') is None
    for dir in ['I', 'J', 'K']:
      container = PT.find_node_from_name(zone, f'Geometry_2d_{dir}')
      assert PT.Subset.GridLocation(container) == f'{dir}FaceCenter'
      assert PT.get_child_from_name(container, 'PointRange') is not None
      assert PT.get_child_from_name(container, 'NormalZ') is not None
  else:
    container = PT.find_node_from_name(zone, f'Geometry_2d')
    assert PT.Subset.GridLocation(container) == 'CellCenter'
    assert PT.get_child_from_name(container, 'PointRange') is None
    assert PT.get_child_from_name(container, 'NormalZ') is not None

  # > U
  expt_loc = 'FaceCenter' if cell_dim == 3 else 'CellCenter'
  for cnt in ['Poly', 'Standard']:
    # > Poly
    tree = PT.deep_copy(base_tree)
    maia.algo.dist.convert_s_to_ngon(tree, comm)
    if cnt == 'Standard':
      # NB : convert_s_to_u with STD elements seems to produce bad face orientation !
      maia.algo.dist.convert_ngon_to_elements(tree, comm)

    zone = PT.get_all_Zone_t(tree)[0]
    GEO.compute_elements_normal(zone, comm, unitary=True)

    container = PT.find_node_from_name(zone, 'Geometry_2d')
    assert PT.Subset.GridLocation(container) == expt_loc
    assert (PT.get_child_from_name(container, 'PointList') is not None) == (cell_dim == 3)
    assert PT.get_child_from_name(container, 'UnitNormalZ') is not None



  
  