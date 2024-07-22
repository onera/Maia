import pytest
import pytest_parallel

import numpy as np

from maia.algo.dist import connectivity_utils as CU

@pytest_parallel.mark.parallel(2)
def test_combine_face_edge_and_edge_vtx(comm):
  edge_distri_full = np.array([0, 12, 24])
  if comm.rank == 0:
    face_edge_idx = np.array([0,4,8,12,16,20])
    face_edge = np.array([1,3,5,7,  -5,2,6,9,   -6,4,8,11,  -7,10,12,14, -12,-9,13,16])
    edge_vtx = np.array([1,2,2,3,4,1,3,4,2,6,3,7,6,5,4,8,7,6,9,5,8,7,6,10])
    expected_face_vtx = [1,2,6,5,  6,2,3,7,  7,3,4,8,  5,6,10,9,  10,6,7,11]
  else:
    face_edge_idx = np.array([20,24,28,32,36])
    face_edge = np.array([-13,-11,15,18,  -14,17,19,21,  -19,-16,20,23,  -20,-18,22,24])

    edge_vtx = np.array([7,11,10,9,8,12,11,10,13,9,12,11,10,14,11,15,14,13,12,16,15,14,16,15])
    expected_face_vtx = [11,7,8,12,  9,10,14,13,  14,10,11,15,  15,11,12,16]

  face_vtx = CU.combine_face_edge_and_edge_vtx(face_edge_idx, face_edge, edge_distri_full, edge_vtx, comm)

  assert np.array_equal(face_vtx, expected_face_vtx)



