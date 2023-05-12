import pytest
import pytest_parallel
import maia.pytree        as PT

from maia.factory import dcloud_generator

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("zone_type", ["Unstructured", "Structured"])
def test_generate_points(zone_type, comm):
  dist_tree = dcloud_generator.generate_dist_points([6,4,1], zone_type, comm)
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [3,3]).all()

  zone = PT.get_all_Zone_t(dist_tree)[0]
  assert len(PT.get_children(zone)) == 3 # Only Coords, ZoneType + Distri node

  assert PT.Zone.Type(zone) == zone_type
  assert PT.Zone.n_vtx(zone) == 6*4*1

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("zone_type", ["Unstructured", "Structured"])
def test_generate_points_dims(zone_type, comm):

  dist_tree = dcloud_generator.generate_dist_points([6,4], zone_type, comm, origin=[0, 0.], max_coords=[1, 4.])
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [2,2]).all()
  assert PT.Zone.n_vtx(zone) == 6*4
  assert PT.get_node_from_name(zone, 'CoordinateZ') is None

  # Expand distri
  dist_tree = dcloud_generator.generate_dist_points(6, zone_type, comm, origin=[0, 0.], max_coords=[1, 4.])
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [2,2]).all()
  assert PT.Zone.n_vtx(zone) == 6*6
  assert PT.get_node_from_name(zone, 'CoordinateZ') is None

  dist_tree = dcloud_generator.generate_dist_points(6, zone_type, comm, origin=[0.], max_coords=[2.])
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [1,1]).all()
  assert PT.Zone.n_vtx(zone) == 6
  assert PT.get_node_from_name(zone, 'CoordinateY') is None
  assert PT.get_node_from_name(zone, 'CoordinateZ') is None

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
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [dim,dim]).all()
  assert PT.Zone.n_vtx(zone) == 42

  if dim == 1:
    assert PT.get_node_from_name(zone, 'CoordinateY') is None
    assert PT.get_node_from_name(zone, 'CoordinateZ') is None
