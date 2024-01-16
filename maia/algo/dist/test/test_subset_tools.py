import pytest
import pytest_parallel
import numpy as np

import maia
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

@pytest_parallel.mark.parallel([1,2])
def test_convert_subset_as_facelist(comm):
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  maia.algo.dist.convert_s_to_u(tree, 'NGON', comm)

  subset_tools.convert_subset_as_facelist(tree, 'Base/zone/ZoneBC/Xmax', comm)

  xmax = PT.get_node_from_name(tree, 'Xmax')
  distri = MT.getDistribution(xmax, 'Index')[1]
  pl = PT.get_child_from_name(xmax, 'PointList')[1]
  assert PT.Subset.GridLocation(xmax) == 'FaceCenter'
  assert pl.ndim == 2
  assert (pl[0] == [3,6,9,12][distri[0]:distri[1]]).all()
