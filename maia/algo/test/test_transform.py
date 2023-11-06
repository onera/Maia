import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.pytree.yaml  import parse_yaml_cgns
from maia.factory.dcube_generator import dcube_generate

from maia.algo import transform

def test_transformation_zone_void():
  yz = """
       Zone Zone_t I4 [[18,4,0]]:
         ZoneType ZoneType_t "Unstructured":
         GridCoordinates GridCoordinates_t:
           CoordinateX DataArray_t:
             R4 : [ 0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2,
                    0,1,2 ]
           CoordinateY DataArray_t:
             R4 : [ 0,0,0,
                    1,1,1,
                    2,2,2,
                    0,0,0,
                    1,1,1,
                    2,2,2 ]
           CoordinateZ DataArray_t:
             R4 : [ 0,0,0,
                    0,0,0,
                    0,0,0,
                    1,1,1,
                    1,1,1,
                    1,1,1 ]
       """
  zone            = parse_yaml_cgns.to_node(yz)
  zone_bck        = PT.deep_copy(zone)
  transform.transform_affine(zone)
  assert PT.is_same_tree(zone_bck, zone) 

@pytest_parallel.mark.parallel(1)
def test_transform_affine(comm):

  def check_vect_field(old_node, new_node, field_name):
    old_data = [PT.get_node_from_name(old_node, f"{field_name}{c}")[1] for c in ['X', 'Y', 'Z']]
    new_data = [PT.get_node_from_name(new_node, f"{field_name}{c}")[1] for c in ['X', 'Y', 'Z']]
    assert np.allclose(old_data[0], -new_data[0])
    assert np.allclose(old_data[1], -new_data[1])
    assert np.allclose(old_data[2],  new_data[2])
  def check_scal_field(old_node, new_node, field_name):
    old_data = PT.get_node_from_name(old_node, field_name)[1]
    new_data = PT.get_node_from_name(new_node, field_name)[1]
    assert (old_data == new_data).all()

  dist_tree = dcube_generate(4, 1., [0., -.5, -.5], comm)
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]

  # Initialise some fields
  cell_distri = MT.getDistribution(dist_zone, 'Cell')[1]
  n_cell_loc =  cell_distri[1] - cell_distri[0]
  fs = PT.new_FlowSolution('FlowSolution', loc='CellCenter', parent=dist_zone)
  PT.new_DataArray('scalar', np.random.random(n_cell_loc), parent=fs)
  PT.new_DataArray('fieldX', np.random.random(n_cell_loc), parent=fs)
  PT.new_DataArray('fieldY', np.random.random(n_cell_loc), parent=fs)
  PT.new_DataArray('fieldZ', np.random.random(n_cell_loc), parent=fs)

  dist_zone_ini = PT.deep_copy(dist_zone)
  transform.transform_affine(dist_zone, rotation_angle=np.array([0.,0.,np.pi]), apply_to_fields=True)

  check_vect_field(dist_zone_ini, dist_zone, "Coordinate")
  check_vect_field(dist_zone_ini, dist_zone, "field")
  check_scal_field(dist_zone_ini, dist_zone, "scalar")

