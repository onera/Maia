import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import interpolation as ITP

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("all_cnt", [False, True])
def test_simple_2d(all_cnt, comm):
  src_tree = maia.factory.generate_dist_block(5, 'QUAD_4', comm)
  tgt_tree = maia.factory.generate_dist_block([17,21], 'S', comm)
  
  # Create sol for src tree
  maia.algo.compute_elements_center(src_tree, 'CellCenter', comm)
  for zone in PT.iter_all_Zone_t(src_tree):
    geo2d = PT.deep_copy(PT.find_child_from_name(zone, 'Geometry_2d'))
    PT.set_name(geo2d, 'Geometry_2d_dupl')
    PT.add_child(zone, geo2d)

  cnt_name = 'ALL' if all_cnt else ['Geometry_2d']
  ITP.interpolate(src_tree, tgt_tree, comm, cnt_name, 'CellCenter', strategy='LocationAndClosest')

  tgt_zone = PT.get_all_Zone_t(tgt_tree)[0]
  tgt_center = maia.algo.dist.geometry._compute_elements_center(tgt_zone, 'CellCenter', comm)
  cell_distri = MT.Zone.cell_distribution(tgt_zone)
  dn_cell = cell_distri[1] - cell_distri[0]
  sols_name = ['Geometry_2d', 'Geometry_2d_dupl'] if all_cnt else ['Geometry_2d']
  for sol_name in sols_name:
    sol = PT.get_node_from_name(tgt_zone, sol_name)
    assert PT.get_label(sol) == 'DiscreteData_t'
    assert PT.Container.GridLocation(sol) == 'CellCenter'
    for array in PT.get_children_from_label(sol, 'DataArray_t'):
      assert array[1].shape == (dn_cell,)
    cx = PT.get_child_from_name(sol, 'CenterX')[1]
    cy = PT.get_child_from_name(sol, 'CenterY')[1]
    # We have 4*5 tgt cells in each src cell, the interp. value is simply the center of the containing cell
    layer_x = np.floor(tgt_center[0::3] / 0.25)
    layer_y = np.floor(tgt_center[1::3] / 0.25)
    assert (cx == 0.125 + 0.25*layer_x).all()
    assert (cy == 0.125 + 0.25*layer_y).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_type", ['Poly', 'HEXA_8'])
@pytest.mark.parametrize("n_tgt"   , [3, 7])
@pytest.mark.parametrize("tgt_loc" , ['Vertex', 'CellCenter'])
@pytest.mark.parametrize("strategy", ['Location', 'LocationAndClosest'])
def test_interpolation_location(comm, elt_type, n_tgt, tgt_loc, strategy):
  src_tree = maia.factory.generate_dist_block(    5, elt_type, comm)
  tgt_tree = maia.factory.generate_dist_block(n_tgt, elt_type, comm, origin=np.array([0, 0, 0.249]))

  for zone in PT.iter_all_Zone_t(src_tree):
    cx,cy,cz = PT.Zone.coordinates(zone)
    PT.new_FlowSolution('FS', loc="Vertex", fields={'cx':cx, 'cy':cy, 'cz':cz}, parent=zone)

  interpolator = maia.algo.create_interpolator(src_tree, tgt_tree, comm, "Vertex", tgt_loc,
                                               strategy=strategy,
                                               n_closest_pt=1)
  interpolator.exchange_fields('FS', ITP.Interpolator._reduce_weighted_mean)

  # > Check result
  zone = PT.get_node_from_label(tgt_tree, "Zone_t")
  if tgt_loc=='Vertex':
    expected_cx = PT.get_node_from_name(tgt_tree, 'CoordinateX')[1]
    expected_cy = PT.get_node_from_name(tgt_tree, 'CoordinateY')[1]
    expected_cz = PT.get_node_from_name(tgt_tree, 'CoordinateZ')[1]
  elif tgt_loc=='CellCenter':
    cell_center = maia.algo.dist.geometry._compute_elements_center(zone, 3, comm)
    expected_cx = cell_center[0::3]
    expected_cy = cell_center[1::3]
    expected_cz = cell_center[2::3]
  is_in_src_pl = np.where(expected_cz<=1.)[0]
  no_in_src_pl = np.where(expected_cz> 1.)[0]
  if strategy=="Location":
    expected_cx[no_in_src_pl]=np.nan
    expected_cy[no_in_src_pl]=np.nan
    expected_cz[no_in_src_pl]=np.nan

  tgt_fs = PT.get_node_from_label(tgt_tree, 'FlowSolution_t')
  tgt_cx = PT.get_child_from_name(tgt_fs, 'cx')[1]
  tgt_cy = PT.get_child_from_name(tgt_fs, 'cy')[1]
  tgt_cz = PT.get_child_from_name(tgt_fs, 'cz')[1]
  if strategy=="Location":
    assert np.allclose(expected_cx, tgt_cx, atol=1e-15, equal_nan=True)
    assert np.allclose(expected_cy, tgt_cy, atol=1e-15, equal_nan=True)
    assert np.allclose(expected_cz, tgt_cz, atol=1e-15, equal_nan=True)
    pass
  else:
    assert np.allclose(expected_cx[is_in_src_pl], tgt_cx[is_in_src_pl], atol=1e-15, equal_nan=True)
    assert np.allclose(expected_cy[is_in_src_pl], tgt_cy[is_in_src_pl], atol=1e-15, equal_nan=True)
    assert np.allclose(expected_cz[is_in_src_pl], tgt_cz[is_in_src_pl], atol=1e-15, equal_nan=True)

    assert np.allclose(expected_cx[no_in_src_pl], tgt_cx[no_in_src_pl], atol=1e-1, equal_nan=True)
    assert np.allclose(expected_cy[no_in_src_pl], tgt_cy[no_in_src_pl], atol=1e-1, equal_nan=True)

