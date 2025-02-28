import pytest
import pytest_parallel
import maia.pytree        as PT

from maia         import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.factory import dcloud_generator

def check_dims(tree, celldim, phydim):
  for base in PT.iter_children_from_label(tree, 'CGNSBase_t'):
    assert (PT.get_value(base) == [celldim, phydim]).all()
    for zone in PT.get_all_Zone_t(base):
      assert PT.Zone.CellDimension(zone) == celldim

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("zone_type", ["Unstructured", "Structured"])
def test_generate_points(zone_type, comm):
  dist_tree = dcloud_generator.generate_dist_points([6,4,1], zone_type, comm)
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [3 if zone_type=="Unstructured" else 2, 3]).all()

  zone = PT.get_all_Zone_t(dist_tree)[0]
  assert PT.get_value(zone).dtype == pdm_gnum_dtype
  assert len(PT.get_children(zone)) == 3 # Only Coords, ZoneType + Distri node

  assert PT.Zone.Type(zone) == zone_type
  assert PT.Zone.n_vtx(zone) == 6*4*1

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("zone_type", ["Unstructured", "Structured"])
def test_generate_points_dims(zone_type, comm):

  dist_tree = dcloud_generator.generate_dist_points([6,4], zone_type, comm, origin=[0, 0.], max_coords=[1, 4.])
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  assert PT.get_value(zone).dtype == pdm_gnum_dtype
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [2,2]).all()
  assert PT.Zone.n_vtx(zone) == 6*4
  assert PT.get_node_from_name(zone, 'CoordinateZ') is None

  # Expand distri
  dist_tree = dcloud_generator.generate_dist_points(6, zone_type, comm, origin=[0, 0.], max_coords=[1, 4.])
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  assert PT.get_value(zone).dtype == pdm_gnum_dtype
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [2,2]).all()
  assert PT.Zone.n_vtx(zone) == 6*6
  assert PT.get_node_from_name(zone, 'CoordinateZ') is None

  dist_tree = dcloud_generator.generate_dist_points(6, zone_type, comm, origin=[0.], max_coords=[2.])
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  assert PT.get_value(zone).dtype == pdm_gnum_dtype
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [1,1]).all()
  assert PT.Zone.n_vtx(zone) == 6
  assert PT.get_node_from_name(zone, 'CoordinateY') is None
  assert PT.get_node_from_name(zone, 'CoordinateZ') is None
  

  check_dims(dcloud_generator.generate_dist_points([6,4,8], "S", comm), 3, 3)
  check_dims(dcloud_generator.generate_dist_points([6,4,1], "S", comm), 2, 3)
  check_dims(dcloud_generator.generate_dist_points([6,1,1], "S", comm), 1, 3)
  check_dims(dcloud_generator.generate_dist_points([6,4]  , "S", comm), 2, 3)
  check_dims(dcloud_generator.generate_dist_points([6,1]  , "S", comm), 1, 3)
  check_dims(dcloud_generator.generate_dist_points([6]    , "S", comm), 1, 3)
  check_dims(dcloud_generator.generate_dist_points(6      , "S", comm), 3, 3)

  for nvtx_list in [[6,4,9], [6,4,1], [6,1,1]]:
    with pytest.raises(AssertionError):
      dcloud_generator.generate_dist_points(nvtx_list, "S", comm, origin=[0., 0.], max_coords=[1., 1.])

  check_dims(dcloud_generator.generate_dist_points([6,4]  , "S", comm, origin=[0., 0.], max_coords=[1., 1.]), 2, 2)
  check_dims(dcloud_generator.generate_dist_points([6,1]  , "S", comm, origin=[0., 0.], max_coords=[1., 1.]), 1, 2)
  check_dims(dcloud_generator.generate_dist_points([6]    , "S", comm, origin=[0., 0.], max_coords=[1., 1.]), 1, 2)
  check_dims(dcloud_generator.generate_dist_points(6      , "S", comm, origin=[0., 0.], max_coords=[1., 1.]), 2, 2)

  for nvtx_list in [[4,7,2], [6,4,1], [6,1,1], [6,4], [6,1]]:
    with pytest.raises(AssertionError):
      dcloud_generator.generate_dist_points(nvtx_list, "S", comm, origin=[0.], max_coords=[1.])

  check_dims(dcloud_generator.generate_dist_points([6]    , "S", comm, origin=[0.], max_coords=[1.]), 1, 1)
  check_dims(dcloud_generator.generate_dist_points(6      , "S", comm, origin=[0.], max_coords=[1.]), 1, 1)
  # correct test
  with pytest.raises(ValueError, match="Unexpected value for zone_type parameter"):
    dcloud_generator.generate_dist_points(10, "Null", comm, origin=[0.], max_coords=[2.])
       

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("dim", [3,1])
def test_generate_points_random(dim, comm):
  
  if dim == 3:
    coords_min = [0., 0., 0.]
    coords_max = [1., 1., 2.]
  elif dim == 1:
    coords_min = [0.]
    coords_max = [10]

  dist_tree = dcloud_generator.dpoint_cloud_random_generate(42, coords_min, coords_max, comm)
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  assert PT.get_value(zone).dtype == pdm_gnum_dtype
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [dim,dim]).all()
  assert PT.Zone.n_vtx(zone) == 42

  if dim == 1:
    assert PT.get_node_from_name(zone, 'CoordinateY') is None
    assert PT.get_node_from_name(zone, 'CoordinateZ') is None
