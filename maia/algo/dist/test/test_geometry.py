import pytest
import pytest_parallel
from   mpi4py import MPI
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia

from maia.algo.dist import geometry

@pytest_parallel.mark.parallel(3)
def test_compute_face_normal3d(comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  face_normal = geometry.compute_face_normal(zone, comm)

  # All face area are 0.25
  if comm.Get_rank() == 0:
    expected_face_normal = 0.25 * np.array([0,0, 1, 0,0, 1, 0,0, 1, 0,0, 1,
                                            0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
                                            0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1])
  elif comm.Get_rank() == 1:
    expected_face_normal = 0.25 * np.array([ 1,0,0,  1,0,0,  1,0,0,  1,0,0,
                                            -1,0,0, -1,0,0, -1,0,0, -1,0,0,
                                            -1,0,0, -1,0,0, -1,0,0, -1,0,0])

  if comm.Get_rank() == 2:
    expected_face_normal = 0.25 * np.array([0, 1,0,  0, 1,0,  0, 1,0,  0, 1,0,
                                            0,-1,0,  0,-1,0,  0,-1,0,  0,-1,0,
                                            0,-1,0,  0,-1,0,  0,-1,0,  0,-1,0])

  assert (face_normal == expected_face_normal).all()

@pytest_parallel.mark.parallel(2)
def test_compute_face_normal2d(comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  face_normal = geometry.compute_face_normal(zone, comm)

  assert (face_normal == np.array([0.,0.,0.125,  0.,0.,0.125,   0.,0.,0.125, 0.,0.,0.125])).all()

