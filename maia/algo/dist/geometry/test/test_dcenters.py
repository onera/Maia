import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia

from maia.algo.dist.geometry import centers as GEO

def to_expected_cyl(expected_cart):
  x = expected_cart[0::3]
  y = expected_cart[1::3]
  z = expected_cart[2::3]
  expected_cyl = np.empty_like(expected_cart)
  expected_cyl[0::3] = np.sqrt(x**2 + y**2) #R
  expected_cyl[1::3] = np.arctan2(y,x)      #O
  expected_cyl[2::3] = z                    #Z
  return expected_cyl


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center3d(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm)

  if comm.Get_rank() == 0:
    expected_face_center = np.array([
        0.25, 0.25, 0. ,  0.75, 0.25, 0. ,  0.25, 0.75, 0. ,  0.75, 0.75, 0. , 
        0.25, 0.25, 0.5,  0.75, 0.25, 0.5,  0.25, 0.75, 0.5,  0.75, 0.75, 0.5, 
        0.25, 0.25, 1. ,  0.75, 0.25, 1. ,  0.25, 0.75, 1. ,  0.75, 0.75, 1. ,
    ])
  elif comm.Get_rank() == 1:
    expected_face_center = np.array([
        0. , 0.25, 0.25,  0. , 0.75, 0.25,  0.,  0.25, 0.75,  0. , 0.75, 0.75,
        0.5, 0.25, 0.25,  0.5, 0.75, 0.25,  0.5, 0.25, 0.75,  0.5, 0.75, 0.75,
        1. , 0.25, 0.25,  1. , 0.75, 0.25,  1.,  0.25, 0.75,  1. , 0.75, 0.75,
    ])
  if comm.Get_rank() == 2:
    expected_face_center = np.array([
        0.25, 0. , 0.25,  0.25, 0. , 0.75,  0.75, 0. , 0.25,  0.75, 0. , 0.75, 
        0.25, 0.5, 0.25,  0.25, 0.5, 0.75,  0.75, 0.5, 0.25,  0.75, 0.5, 0.75,
        0.25, 1. , 0.25,  0.25, 1. , 0.75,  0.75, 1. , 0.25,  0.75, 1. , 0.75,
    ])

  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center2d(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm)

  if cylindrical:
    if comm.Get_rank() == 0:
      assert np.allclose(face_center, np.array([0.23570226,0.78539816,0,  0.47140452,0.78539816,0,  0.68718427,0.24497866,0,  0.89752747,0.38050638,0]))
    if comm.Get_rank() == 1:
      assert np.allclose(face_center, np.array([0.68718427,1.32581766,0,  0.89752747,1.19028995,0,  0.94280904,0.78539816,0,  1.17851130,0.78539816,0]))
  else:
    if comm.Get_rank() == 0:
      assert (face_center == np.array([1.,1,0, 2,2,0, 4,1,0, 5,2,0]) / 6.).all()
    if comm.Get_rank() == 1:
      assert (face_center == np.array([1.,4,0, 2,5,0, 4,4,0, 5,5,0]) / 6.).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ["S", "NFACE_n", "Poly"])
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_cell_center(elt_kind, cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))
  
  cell_center = GEO.compute_cell_center(zone, comm)
  
  if cylindrical:
      from math import pi
      expt_cell_center = [
        np.array([0.35355339, pi/4, 0.25,  0.79056942,0.32175055,0.25,  0.79056942,1.24904577,0.25,  1.06066017,pi/4,0.25]),
        np.array([0.35355339, pi/4, 0.75,  0.79056942,0.32175055,0.75,  0.79056942,1.24904577,0.75,  1.06066017,pi/4,0.75])
      ][comm.Get_rank()]
  else:
      expt_cell_center = [
        np.array([0.25,0.25,0.25, 0.75,0.25,0.25, 0.25,0.75,0.25, 0.75,0.75,0.25]),
        np.array([0.25,0.25,0.75, 0.75,0.25,0.75, 0.25,0.75,0.75, 0.75,0.75,0.75])
      ][comm.Get_rank()]

  assert np.allclose(expt_cell_center, cell_center)
