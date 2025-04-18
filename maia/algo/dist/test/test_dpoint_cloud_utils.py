import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT

import maia

from maia.algo.dist import point_cloud_utils as PCU

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ['S', 'TRI_3'])
@pytest.mark.parametrize("location", ['Vertex', 'CellCenter'])
def test_get_point_cloud(elt_kind, location, comm):
  n_vtx = [3,3] if elt_kind == 'S' else 3
  tree = maia.factory.generate_dist_block(n_vtx, elt_kind, comm)

  pt_cloud = PCU.get_point_cloud(PT.get_all_Zone_t(tree)[0], comm, location)

  if location == 'Vertex':
    expected_lngn = [np.array([1,2,3,4,5]), np.array([6,7,8,9])][comm.rank]
    expected_coor = [np.array([0,0,0, .5,0,0, 1,0,0,   0,.5,0, .5,.5,0]),
                      np.array([1,.5,0,   0,1,0, .5,1,0, 1,1,0])][comm.rank]
  else:
    if elt_kind == 'S':
      expected_lngn = [np.array([1,2]), np.array([3,4])][comm.rank]
      expected_coor = [np.array([.25,.25,0, .75,.25,0]), np.array([.25,.75,0, .75,.75,0])][comm.rank]
    else:
      expected_lngn = [np.array([1,2,3,4]), np.array([5,6,7,8])][comm.rank]
      expected_coor = [np.array([1,1,0, 2,2,0, 4,1,0, 5,2,0]) / 6.,
                       np.array([1,4,0, 2,5,0, 4,4,0, 5,5,0]) / 6.][comm.rank]

  assert np.array_equal(pt_cloud[0], expected_coor)
  assert np.array_equal(pt_cloud[1], expected_lngn)

@pytest_parallel.mark.parallel(1)
def test_get_point_cloud_from_ctn(comm):
  tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  maia.algo.compute_elements_center(zone, 'CellCenter', comm)
  ctn = PT.get_node_from_name(zone, 'Geometry_3d')
  for child in PT.get_children(ctn):
    PT.set_name(child, PT.get_name(child).replace('Center', 'Coordinate'))

  pt_cloud = PCU.get_point_cloud(zone, comm, 'Geometry_3d')
  assert (pt_cloud[1] == np.arange(1, PT.Zone.n_cell(zone)+1)).all()

  with pytest.raises(RuntimeError):
    PCU.get_point_cloud(zone, comm, 'MissingNode')