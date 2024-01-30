import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree as PT

from maia.algo.dist import transform

@pytest_parallel.mark.parallel(2)
def test_transform_affine_zone(comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')
  cx,cy,cz = PT.Zone.coordinates(zone)
  PT.new_FlowSolution('FS', fields={'cx':cx, 'cy':cy, 'cz':cz}, parent=zone)
  zone_bck = PT.deep_copy(zone)

  vtx_ids = np.array([], int)
  transform.transform_affine_zone(zone, vtx_ids, comm, translation=[10, 0, 0])
  assert PT.is_same_tree(zone, zone_bck)

  if comm.Get_rank() == 0:
    vtx_ids = np.array([3,21], int)
  elif comm.Get_rank() == 1:
    vtx_ids = np.array([21,1,18], int)
  transform.transform_affine_zone(zone, vtx_ids, comm, translation=[10, 0, 0])
  
  cx, cy, cz = PT.Zone.coordinates(zone)
  cx_bck, cy_bck, cz_bck = PT.Zone.coordinates(zone_bck)
  if comm.Get_rank() == 0:
    assert (cx == cx_bck + np.array([10, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all() # 1 to 14
  elif comm.Get_rank() == 1:
    assert (cx == cx_bck + np.array([0, 0, 0, 10, 0, 0, 10, 0, 0, 0, 0, 0, 0])).all() # 15 to 27
  assert (cy == cy_bck).all() and (cz == cz_bck).all()
  assert (PT.get_node_from_name(zone, 'cx')[1] == cx).all()

