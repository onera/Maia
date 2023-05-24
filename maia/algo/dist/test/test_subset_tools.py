import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.factory.dcube_generator import dcube_generate
from maia.algo.dist import subset_tools

@pytest_parallel.mark.parallel(2)
def test_vtx_ids_to_face_ids(comm):
  tree = dcube_generate(3, 1., [0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]
  ngon = PT.Zone.NGonNode(zone)
  if comm.Get_rank() == 0:
    vtx_ids = np.array([4,5,17,18], pdm_dtype)
    expected_face_ids = np.array([3], pdm_dtype)
  elif comm.Get_rank() == 1:
    vtx_ids = np.array([5,7,27,25,26,8], pdm_dtype)
    expected_face_ids = np.array([36], pdm_dtype)
  face_ids = subset_tools.vtx_ids_to_face_ids(vtx_ids, ngon, comm)
  assert (face_ids == expected_face_ids).all()

