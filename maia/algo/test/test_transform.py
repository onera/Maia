import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia.pytree.yaml  import parse_yaml_cgns
from maia.factory.dcube_generator import dcube_generate

from maia.algo import transform

from maia.utils import logging as mlog

class log_capture:
  def __init__(self):
    self.logs = ''
  def log(self, msg):
    self.logs += msg

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

@pytest_parallel.mark.parallel(1)
def test_transform_affine_2d(comm):

  dist_tree = maia.factory.generate_dist_block(4, 'S', comm, origin=[.0, .0])
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]

  # Initialise some fields
  cell_distri = MT.getDistribution(dist_zone, 'Cell')[1]
  n_cell_loc =  cell_distri[1] - cell_distri[0]
  fs = PT.new_FlowSolution('FlowSolution', loc='CellCenter', parent=dist_zone)
  PT.new_DataArray('scalar', np.random.random(n_cell_loc), parent=fs)
  PT.new_DataArray('fieldX', np.random.random(n_cell_loc), parent=fs)
  PT.new_DataArray('fieldY', np.random.random(n_cell_loc), parent=fs)

  dist_zone_ini = PT.deep_copy(dist_zone)
  transform.transform_affine(dist_zone, rotation_center=np.zeros(2), translation=np.zeros(2), rotation_angle=np.pi, apply_to_fields=True)
  assert np.allclose(PT.get_node_from_name(dist_zone_ini, 'scalar')[1],
                     PT.get_node_from_name(dist_zone,     'scalar')[1])
  assert np.allclose(   PT.get_node_from_name(dist_zone_ini, 'fieldX')[1],
                     -1*PT.get_node_from_name(dist_zone,     'fieldX')[1])
  assert np.allclose(   PT.get_node_from_name(dist_zone_ini, 'fieldY')[1],
                     -1*PT.get_node_from_name(dist_zone,     'fieldY')[1])

@pytest_parallel.mark.parallel(2)
def test_transform_affine_s_part(comm):
  dist_tree = maia.factory.generate_dist_block(4, 'S', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  part_tree_bck = PT.deep_copy(part_tree)
  transform.transform_affine(part_tree, translation=[5,1,2])
  assert (PT.get_node_from_name(part_tree, 'CoordinateX')[1] == 5+PT.get_node_from_name(part_tree_bck, 'CoordinateX')[1]).all()
  assert (PT.get_node_from_name(part_tree, 'CoordinateY')[1] == 1+PT.get_node_from_name(part_tree_bck, 'CoordinateY')[1]).all()
  assert (PT.get_node_from_name(part_tree, 'CoordinateZ')[1] == 2+PT.get_node_from_name(part_tree_bck, 'CoordinateZ')[1]).all()

  dist_tree = maia.factory.generate_dist_block([4,4], 'S', comm, origin=[0,0])
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
  part_zone = PT.get_node_from_label(part_tree, 'Zone_t')
  fs = PT.new_FlowSolution('FlowSolution', loc='CellCenter', parent=part_zone)
  PT.new_DataArray('fieldX', np.random.random(PT.Zone.n_cell(part_zone)), parent=fs)
  PT.new_DataArray('fieldY', np.random.random(PT.Zone.n_cell(part_zone)), parent=fs)
  part_tree_bck = PT.deep_copy(part_tree)
  transform.transform_affine(part_tree, rotation_center=np.zeros(2), translation=np.zeros(2), rotation_angle=0.5*np.pi, apply_to_fields=True)
  assert np.allclose(PT.get_node_from_name(part_tree, 'fieldX')[1], -PT.get_node_from_name(part_tree_bck, 'fieldY')[1])
  assert np.allclose(PT.get_node_from_name(part_tree, 'fieldY')[1],  PT.get_node_from_name(part_tree_bck, 'fieldX')[1])

@pytest_parallel.mark.parallel(1)
def test_scale_mesh(comm):
  # To check if warning displays
  log_collector = log_capture()
  mlog.add_printer_to_logger('maia-warnings', log_collector)

  dist_tree = maia.factory.generate_dist_block(4, 'Poly', comm)
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]

  cx_bck = PT.get_node_from_name(dist_zone, 'CoordinateX')[1].copy()
  cy_bck = PT.get_node_from_name(dist_zone, 'CoordinateY')[1].copy()
  cz_bck = PT.get_node_from_name(dist_zone, 'CoordinateZ')[1].copy()

  transform.scale_mesh(dist_tree, [1.0, 2.0, 0.5])
  assert (PT.get_node_from_name(dist_tree, 'CoordinateX')[1] == cx_bck).all()
  assert (PT.get_node_from_name(dist_tree, 'CoordinateY')[1] == 2*cy_bck).all()
  assert (PT.get_node_from_name(dist_tree, 'CoordinateZ')[1] == 0.5*cz_bck).all()

  assert log_collector.logs == ''

  dist_tree = maia.factory.generate_dist_block([4,4], 'S', comm, origin=np.zeros(2))
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  PT.new_FlowSolution(loc='CellCenter', fields={'Field': np.ones(PT.Zone.n_cell(zone))}, parent=zone)
  dist_tree_bck = PT.deep_copy(dist_tree)
  transform.scale_mesh(dist_tree, 5)
  assert np.allclose(PT.get_node_from_name(dist_tree, 'CoordinateX')[1], 5*PT.get_node_from_name(dist_tree_bck, 'CoordinateX')[1])
  assert np.allclose(PT.get_node_from_name(dist_tree, 'CoordinateY')[1], 5*PT.get_node_from_name(dist_tree_bck, 'CoordinateY')[1])

  assert "Scaling mesh does not affect fields, and some are present in tree." in log_collector.logs

@pytest.mark.parametrize("revolution_axis", [(1, 0, 0), [0, 1, 0], (1, 0, 3), [1, 2, 3]])
def test_transform_matrix(revolution_axis):
   
  # Create the transform matrix and the reverse transform matrix 
  transform_matrix = maia.algo.transform.create_transform_matrix(revolution_axis=revolution_axis)
  transform_matrix_inv = np.linalg.inv(transform_matrix)
  id = np.dot(transform_matrix, transform_matrix_inv)
  
  # Transform the current revolution axis into a unit revolution axis in the new basis
  new_revolution_axis = np.dot(transform_matrix, revolution_axis)
  norm_new_revolution_axis = np.linalg.norm(new_revolution_axis)
  unit_revolution_axis = new_revolution_axis / norm_new_revolution_axis

  # Transform the unit revolution axis in the new basis into the former revolution axis in the former basis 
  reverse_unit_revolution_axis = np.dot(transform_matrix_inv, unit_revolution_axis)
  reverse_revolution_axis = reverse_unit_revolution_axis * norm_new_revolution_axis

  assert np.allclose(revolution_axis, reverse_revolution_axis)
  assert np.allclose(id, np.eye(3))

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("revolution_axis", [(1, 0, 0), [0, 1, 0], (1, 0, 3), [1, 2, 3]])
@pytest.mark.parametrize("zonetype", ['S', 'Poly'])
class Test_cart_to_cyl:
  def test_change_basis(self, zonetype, revolution_axis, comm):
    
    dist_tree = maia.factory.generate_dist_block(4, zonetype, comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover the intial cartesian coordinates
      init_cx, init_cy, init_cz = PT.Zone.coordinates(zone)

      # Create the transform matrix and the reverse transform matrix 
      transform_matrix = maia.algo.transform.create_transform_matrix(revolution_axis=revolution_axis)
      transform_matrix_inv = np.linalg.inv(transform_matrix)

      # Compute the new coordinates in the new basis
      maia.algo.transform.change_basis(part_tree, transform_matrix, name_in='GridCoordinates', name_out='GridCoordinatesTransform')
      
      # Compute the former coordinates in ther former basis
      maia.algo.transform.change_basis(part_tree, transform_matrix_inv, name_in='GridCoordinatesTransform', name_out='GridCoordinatesReverse')

      # Recover coordinates computed from the double basis change
      reverse_cx, reverse_cy, reverse_cz = PT.Zone.coordinates(zone, name='GridCoordinatesReverse')

      assert np.allclose(init_cx, reverse_cx)
      assert np.allclose(init_cy, reverse_cy)
      assert np.allclose(init_cz, reverse_cz)

  def test_cylinder_to_cartesian(self, zonetype, revolution_axis, comm): 

    dist_tree = maia.factory.generate_dist_block(4, zonetype, comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

    for zone in  PT.get_all_Zone_t(part_tree):
      # Recover the inital cartesian cooridnates
      init_cx, init_cy, init_cz = PT.Zone.coordinates(zone)
      
      # Compute the cylinder coordinates from the cartesian cooridnates
      maia.algo.transform.cartesian_to_cylinder(part_tree, revolution_axis=revolution_axis)

      # Remove the inital and the new cartesian coordinates
      PT.rm_nodes_from_name(part_tree, 'GridCoordinates')
      PT.rm_nodes_from_name(part_tree, 'GridCoordinatesTransform')

      # Compute the cartesian coordinates from the cylinder coordinates
      maia.algo.transform.cylinder_to_cartesian(part_tree, revolution_axis=revolution_axis)

      # Recover the cartesian coordinates computed from the double change coordinates
      cx, cy, cz = PT.Zone.coordinates(zone, name='GridCoordinates')

      assert np.allclose(init_cx, cx)
      assert np.allclose(init_cy, cy)
      assert np.allclose(init_cz, cz)
