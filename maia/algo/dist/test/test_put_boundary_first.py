import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import put_boundary_first


@pytest_parallel.mark.parallel(2)
def test_put_boundary_first(comm):
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  maia.algo.dist.convert_s_to_ngon(tree, comm)
  put_boundary_first.put_boundary_first(tree, comm)

  expt_cx_f = [0,.5,1, 0,.5,1, 0,.5,1., 0,.5,1., 0.,1., 0,.5,1, 0,.5,1, 0,.5,1, 0,.5,1, .5]
  expt_cy_f = [0,0,0, .5,.5,.5, 1,1,1,  0,0,0,  .5,.5,  1,1,1,  0,0,0, .5,.5,.5, 1,1,1, .5]
  expt_cz_f = [0,0,0, 0,0,0, 0,0,0,   .5,.5,.5, .5,.5, .5,.5,.5,  1,1,1, 1,1,1, 1,1,1,  .5]

  zone = PT.get_all_Zone_t(tree)[0]
  vtx_distri = MT.distribution_value(zone, 'Vertex')
  assert (PT.get_np_value(zone) == [27,8,26]).all()
  cx,cy,cz = PT.Zone.coordinates(zone)
  assert (cx == np.array(expt_cx_f)[vtx_distri[0]:vtx_distri[1]]).all()
  assert (cy == np.array(expt_cy_f)[vtx_distri[0]:vtx_distri[1]]).all()
  assert (cz == np.array(expt_cz_f)[vtx_distri[0]:vtx_distri[1]]).all()

  # Check random faces
  ng = PT.Zone.NGonNode(zone)
  assert (PT.get_np_value(ng) == [22,24]).all()
  face_vtx = MT.Element.connectivity(ng)
  if comm.rank == 0:
    assert (face_vtx[3] == [6,9,17,14]).all() # External face
  if comm.rank == 1:
    assert (face_vtx[25-18-1] == [2,5,27,11]).all() # Internal face

  # Check a some BCs
  expt_ymax = np.array([11,12,15,16])
  expt_zmin = np.array([17,18,19,20])
  for expt, name in zip([expt_ymax, expt_zmin], ['Ymax', 'Zmin']):
    bc = PT.find_node_from_name(tree, name)
    pl = PT.get_child_from_name(bc, 'PointList')
    distri = MT.distribution_value(bc, 'Index')
    assert (pl[1][0] == expt[distri[0]:distri[1]]).all()

@pytest_parallel.mark.parallel(1)
def test_put_boundary_first_2d_elt(comm):
  tree = maia.factory.generate_dist_block(4, 'QUAD_4', comm)
  put_boundary_first.put_boundary_first(tree, comm)

  zone = PT.find_node_from_label(tree, 'Zone_t')
  vtx_distri = MT.distribution_value(zone, 'Vertex')
  expt_cx = np.array([0, 1, 2, 3, 0, 3, 0, 3, 0, 1, 2, 3, 1, 2, 1, 2]) / 3,
  expt_cy = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2]) / 3
  
  cx,cy,cz = PT.Zone.coordinates(zone)
  assert np.allclose(cx, expt_cx[vtx_distri[0]:vtx_distri[1]])
  assert np.allclose(cy, expt_cy[vtx_distri[0]:vtx_distri[1]])

  bar = PT.find_node_from_predicate(zone, PT.pred.is_element_of_type('BAR_2'))
  bar_ec = PT.get_np_value(PT.find_child_from_name(bar, 'ElementConnectivity'))
  assert bar_ec.max() <= 12 # 12 external vtx
  

@pytest_parallel.mark.parallel(2)
def test_put_boundary_first_2d(comm):
  tree = maia.factory.generate_dist_block([4,4], 'S', comm, origin=[0,0])
  maia.algo.dist.convert_s_to_ngon(tree, comm)
  ztype = PT.get_np_value(PT.get_all_Zone_t(tree)[0]).dtype

  put_boundary_first.put_boundary_first(tree, comm)

  ftree = maia.factory.dist_to_full_tree(tree, comm)
  
  expt_tree = PT.new_CGNSTree()
  expt_base = PT.new_CGNSBase('Base', cell_dim=2, phy_dim=2, parent=expt_tree)
  expt_zone = PT.new_Zone('zone', type='Unstructured', size=np.array([[16,9,12]], ztype), parent=expt_base)
  PT.new_GridCoordinates(fields={
                            'CoordinateX' : np.array([0, 1, 2, 3, 0, 3, 0, 3, 0, 1, 2, 3, 1, 2, 1, 2]) / 3,
                            'CoordinateY' : np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2]) / 3
                          }, parent=expt_zone)

  expt_edge_ec = np.array([5,1, 4,6, 7,5, 6,8, 9,7, 8,12, 1,2, 2,3, 3,4, 10,9, 11,10, 12,11, # External
                           2,13, 3,14, 13,15, 14,16, 15,10, 16,11, 5,13, 13,14, 14,6, 7,15, 15,16, 16,8]) # Internal
  expt_edge_pe = np.array(
     [[25,0], [27,0], [28,0], [30,0], [31,0], [33,0], [25,0], [26,0], [27,0], [31,0], [32,0], [33,0],
      [25,26], [26,27], [28,29], [29,30], [31,32], [32,33], [28,25], [29,26], [30,27], [31,28], [32,29], [33,30]])
  expt_bar = PT.new_Elements('EdgeElements', 'BAR_2', erange=np.array([1,24], ztype), parent=expt_zone)
  PT.get_np_value(expt_bar)[1] = 12
  PT.new_DataArray('ElementConnectivity', expt_edge_ec.astype(ztype), parent=expt_bar)
  PT.new_DataArray('ParentElements', expt_edge_pe.astype(ztype), parent=expt_bar)

  expt_zbc = PT.new_ZoneBC(parent=expt_zone)
  expt_bc_pl = {'Xmin' : [1,3,5], 'Xmax' : [2,4,6], 'Ymin' : [7,8,9], 'Ymax' : [10,11,12]}
  for name,pl in expt_bc_pl.items():
    PT.new_BC(name, loc='EdgeCenter', point_list=np.array(pl, ztype).reshape((1,-1)), parent=expt_zbc)


  ok = False
  if comm.rank == 0:
    ok = PT.is_same_tree(ftree, expt_tree)
  assert comm.bcast(ok, root=0)

