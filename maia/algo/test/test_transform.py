import pytest
import pytest_parallel
import numpy as np

import maia.pytree          as PT
import maia.pytree.maia     as MT
from maia.utils             import np_utils

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

@pytest.mark.parametrize('revolution_axis', [(0, 1, 1), [1, 2, 3], (0, 0, 1), [0, 1, 0]])
@pytest.mark.parametrize('zonetype', ['S', 'Poly'])       
class Test:    
  def test_change_basis(self, zonetype, revolution_axis, comm):

      import maia
      import maia.pytree as PT
      import copy
      
      dist_tree = maia.factory.generate_dist_block(4, zonetype, comm)
      part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

      for zone in PT.get_all_Zone_t(part_tree):
        # Recover the intial cartesian coordinates
        coords = PT.Zone.coordinates(zone)
        children = copy.deepcopy(PT.get_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'))
        PT.set_name(children[0], 'DDX')
        PT.set_name(children[1], 'DDY')
        PT.set_name(children[2], 'DDZ')

        PT.new_FlowSolution('FlowSolution', fields={'FSX' : coords[0], 'FSY' : coords[1], 'FSZ' : coords[2]}, parent=zone)
        PT.new_ZoneSubRegion('ZoneSubRegion', fields={'ZSRX' : coords[0], 'ZSRY' : coords[1], 'ZSRZ' : coords[2]}, parent=zone)
        PT.new_node('DiscreteData', 'DiscreteData_t', children=children, parent=zone)

      # Create the transform matrix and the reverse transform matrix 
      transform_matrix = np_utils.create_transform_matrix(revolution_axis=revolution_axis)
      
      part_tree_cart_ref = copy.deepcopy(part_tree)
        
      # Compute the new coordinates in the new basis
      transform.change_basis(part_tree, transform_matrix, gc_name='GridCoordinatesBis', apply_to_fields=True)

      assert PT.is_same_tree(part_tree_cart_ref, part_tree)

      # Compute the new coordinates in the new basis
      transform.change_basis(part_tree, transform_matrix, gc_name='GridCoordinates', apply_to_fields=True)

      # Compute the new coordinates in the new basis
      transform.change_basis(part_tree, transform_matrix, gc_name='GridCoordinates', apply_to_fields=True)

      # Compute the former coordinates in ther former basis
      transform.change_basis(part_tree, None, gc_name='GridCoordinates', apply_to_fields=True)

      assert PT.is_same_tree(part_tree_cart_ref, part_tree, abs_tol=1e-10)

  def test_cylindric_to_cartesian(self, zonetype, revolution_axis, comm):

    import maia
    import maia.pytree as PT
    import numpy as np
    import copy
    
    dist_tree = maia.factory.generate_dist_block(3, zonetype, comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover the intial cartesian coordinates
      coords = PT.Zone.coordinates(zone)
      children = copy.deepcopy(PT.get_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'))
      PT.set_name(children[0], 'DDX')
      PT.set_name(children[1], 'DDY')
      PT.set_name(children[2], 'DDZ')

      # Create fields in zone
      PT.new_FlowSolution('FlowSolution', fields={'FSX' : coords[0], 'FSY' : coords[1], 'FSZ' : coords[2]}, parent=zone)
      PT.new_ZoneSubRegion('ZoneSubRegion', fields={'ZSRX' : coords[0], 'ZSRY' : coords[1], 'ZSRZ' : coords[2]}, parent=zone)
      PT.new_node('DiscreteData', 'DiscreteData_t', children=children, parent=zone)

    # Copy of the cartesian part tree
    part_tree_cart_ref = copy.deepcopy(part_tree)

    if revolution_axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
      cart2cyl = transform.cartesian_to_cylindric_from_unit_revolution_axis
      cyl2cart = transform.cylindric_to_cartesian_from_unit_revolution_axis
    else: 
      cart2cyl = transform.cartesian_to_cylindric
      cyl2cart = transform.cylindric_to_cartesian

    # Transform cartesian coordinates and fields into cylindric around a unit revolution axis
    cart2cyl(part_tree, revolution_axis=revolution_axis, gc_name='GridCoordinatesBis')
    assert PT.is_same_tree(part_tree_cart_ref, part_tree)
    cart2cyl(part_tree, revolution_axis=revolution_axis)

    # Copy of the cylindric part tree
    part_tree_cyl_ref = copy.deepcopy(part_tree)

    # Transform cylindric coordinates and fields into cartesian around a unit revolution axis
    cyl2cart(part_tree, revolution_axis=revolution_axis, gc_name='GridCoordinatesBis')
    assert PT.is_same_tree(part_tree_cyl_ref, part_tree)
    cyl2cart(part_tree, revolution_axis=revolution_axis)
    
    for zone in PT.get_all_Zone_t(part_tree):
      # Recover coordinates and fields in the new basis
      gc_n = PT.get_nodes_from_predicates(zone, 'GridCoordinates/DataArray_t')
      gc_coords = [PT.get_value(n) for n in gc_n]
      fs_n  = PT.get_nodes_from_predicates(zone, 'FlowSolution/DataArray_t')
      fs_coords = [PT.get_value(n) for n in fs_n]
      zsr_n  = PT.get_nodes_from_predicates(zone, 'ZoneSubRegion/DataArray_t')
      zsr_coords = [PT.get_value(n) for n in zsr_n]
      dd_n  = PT.get_nodes_from_predicates(zone, 'DiscreteData/DataArray_t')
      dd_coords = [PT.get_value(n) for n in dd_n]

      assert np.allclose(coords[0], gc_coords[0])
      assert np.allclose(coords[0], fs_coords[0])
      assert np.allclose(coords[0], zsr_coords[0])
      assert np.allclose(coords[0], dd_coords[0])

      assert np.allclose(coords[1], gc_coords[1])
      assert np.allclose(coords[1], fs_coords[1])
      assert np.allclose(coords[1], zsr_coords[1])
      assert np.allclose(coords[1], dd_coords[1])

      assert np.allclose(coords[2], gc_coords[2])
      assert np.allclose(coords[2], fs_coords[2])
      assert np.allclose(coords[2], zsr_coords[2])
      assert np.allclose(coords[2], dd_coords[2])

#%%%
@pytest_parallel.mark.parallel([1, 2])          
class Test_unit:
  revolution_axis = (1, 0, 0)
  def test_cartesian_to_cylindric_unit_S(self, comm):

      import maia
      import maia.pytree as PT
      import numpy as np
      import copy
      
      dist_tree = maia.factory.generate_dist_block(3, 'S', comm)
      part_tree = maia.factory.partition_dist_tree(dist_tree, comm)


      for zone in PT.get_all_Zone_t(part_tree):
        # Recover the intial cartesian coordinates
        coords = PT.Zone.coordinates(zone)
        children = copy.deepcopy(PT.get_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'))
        PT.set_name(children[0], 'DDX')
        PT.set_name(children[1], 'DDY')
        PT.set_name(children[2], 'DDZ')

        # Create fields in zone
        PT.new_FlowSolution('FlowSolution', fields={'FSX' : coords[0], 'FSY' : coords[1], 'FSZ' : coords[2]}, parent=zone)
        PT.new_ZoneSubRegion('ZoneSubRegion', fields={'ZSRX' : coords[0], 'ZSRY' : coords[1], 'ZSRZ' : coords[2]}, parent=zone)
        PT.new_node('DiscreteData', 'DiscreteData_t', children=children, parent=zone)

      # Copy of the cartesian part tree
      part_tree_cart_ref = copy.deepcopy(part_tree)

      # Transform cartesian coordinates and fields into cylindric from any revolution axis
      transform.cartesian_to_cylindric_from_unit_revolution_axis(part_tree, revolution_axis=self.revolution_axis, gc_name='GridCoordinatesBis')

      assert PT.is_same_tree(part_tree_cart_ref, part_tree)

      transform.cartesian_to_cylindric_from_unit_revolution_axis(part_tree, revolution_axis=self.revolution_axis) 

      if comm.size == 1:
        radius_ref = [0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1., 0.5, 0.5, 0.5, 0.70710678, 0.70710678, 0.70710678, 1.11803399, 1.11803399, 
                      1.11803399, 1., 1., 1., 1.11803399, 1.11803399, 1.11803399, 1.41421356, 1.41421356, 1.41421356]
        theta_ref  = [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.57079633, 1.57079633, 1.57079633, 0.78539816, 0.78539816, 0.78539816, 0.46364761, 0.46364761, 
                      0.46364761, 1.57079633, 1.57079633, 1.57079633, 1.10714872, 1.10714872, 1.10714872, 0.78539816, 0.78539816, 0.78539816]
        z_ref      = [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]
      elif comm.size == 2:
        if comm.rank == 0: 
          radius_ref = [0., 0., 0.5, 0.5, 1., 1., 0.5, 0.5, 0.70710678, 0.70710678, 1.11803399, 
                        1.11803399, 1., 1., 1.11803399, 1.11803399, 1.41421356, 1.41421356]
          theta_ref  = [0., 0., 0., 0., 0., 0., 1.57079633, 1.57079633, 0.78539816, 0.78539816, 0.46364761, 
                        0.46364761, 1.57079633, 1.57079633, 1.10714872, 1.10714872, 0.78539816, 0.78539816]
          z_ref      = [0., 0.5, 0., 0.5, 0., 0.5, 0., 0.5, 0., 0.5, 0., 0.5, 0., 0.5, 0., 0.5, 0., 0.5]
        elif comm.rank == 1:
          radius_ref = [0., 0., 0.5, 0.5, 1., 1., 0.5, 0.5, 0.70710678, 0.70710678, 1.11803399, 
                        1.11803399, 1., 1., 1.11803399, 1.11803399, 1.41421356, 1.41421356]
          theta_ref  = [0., 0., 0., 0., 0., 0., 1.57079633, 1.57079633, 0.78539816, 0.78539816, 0.46364761, 0.46364761,
                        1.57079633, 1.57079633, 1.10714872, 1.10714872, 0.78539816, 0.78539816]
          z_ref      = [0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1.]

      for zone in PT.get_all_Zone_t(part_tree):
        # Recover coordinates and fields in the new basis
        gc_n = PT.get_nodes_from_predicates(zone, 'GridCoordinates/DataArray_t')
        gc_coords = [PT.get_value(n) for n in gc_n]
        fs_n  = PT.get_nodes_from_predicates(zone, 'FlowSolution/DataArray_t')
        fs_coords = [PT.get_value(n) for n in fs_n]
        zsr_n  = PT.get_nodes_from_predicates(zone, 'ZoneSubRegion/DataArray_t')
        zsr_coords = [PT.get_value(n) for n in zsr_n]
        dd_n  = PT.get_nodes_from_predicates(zone, 'DiscreteData/DataArray_t')
        dd_coords = [PT.get_value(n) for n in dd_n]

        assert np.allclose(radius_ref, gc_coords[1].flatten('F'))
        assert np.allclose(radius_ref, fs_coords[1].flatten('F'))
        assert np.allclose(radius_ref, zsr_coords[1].flatten('F'))
        assert np.allclose(radius_ref, dd_coords[1].flatten('F'))

        assert np.allclose(theta_ref, gc_coords[2].flatten('F'))
        assert np.allclose(theta_ref, fs_coords[2].flatten('F'))
        assert np.allclose(theta_ref, zsr_coords[2].flatten('F'))
        assert np.allclose(theta_ref, dd_coords[2].flatten('F'))

        assert np.allclose(z_ref, gc_coords[0].flatten('F'))
        assert np.allclose(z_ref, fs_coords[0].flatten('F'))
        assert np.allclose(z_ref, zsr_coords[0].flatten('F'))
        assert np.allclose(z_ref, dd_coords[0].flatten('F'))

        with pytest.raises(AssertionError):
          transform.cartesian_to_cylindric_from_unit_revolution_axis(part_tree, revolution_axis=(0, 0, 0)) 

  def test_cartesian_to_cylindric_unit_U(self, comm):

      import maia
      import maia.pytree as PT
      import numpy as np
      import copy
      
      dist_tree = maia.factory.generate_dist_block(3, 'Poly', comm)
      part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

      revolution_axis = (1, 0, 0)

      for zone in PT.get_all_Zone_t(part_tree):
        # Recover the intial cartesian coordinates
        coords = PT.Zone.coordinates(zone)
        children = copy.deepcopy(PT.get_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'))
        PT.set_name(children[0], 'DDX')
        PT.set_name(children[1], 'DDY')
        PT.set_name(children[2], 'DDZ')

        # Create fields in zone
        PT.new_FlowSolution('FlowSolution', fields={'FSX' : coords[0], 'FSY' : coords[1], 'FSZ' : coords[2]}, parent=zone)
        PT.new_ZoneSubRegion('ZoneSubRegion', fields={'ZSRX' : coords[0], 'ZSRY' : coords[1], 'ZSRZ' : coords[2]}, parent=zone)
        PT.new_node('DiscreteData', 'DiscreteData_t', children=children, parent=zone)

      # Copy of the cartesian part tree
      part_tree_cart_ref = copy.deepcopy(part_tree)

      # Transform cartesian coordinates and fields into cylindric from any revolution axis
      transform.cartesian_to_cylindric_from_unit_revolution_axis(part_tree, revolution_axis=revolution_axis, gc_name='GridCoordinatesBis')
      assert PT.is_same_tree(part_tree_cart_ref, part_tree)
      transform.cartesian_to_cylindric_from_unit_revolution_axis(part_tree, revolution_axis=revolution_axis)   

      if comm.size == 1:
        radius_ref = [0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1., 0.5, 0.5, 0.5, 0.70710678, 0.70710678, 0.70710678, 1.11803399, 1.11803399,
                      1.11803399, 1., 1., 1., 1.11803399, 1.11803399, 1.11803399, 1.41421356, 1.41421356, 1.41421356]
        theta_ref  = [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.57079633, 1.57079633, 1.57079633, 0.78539816, 0.78539816, 0.78539816, 0.46364761, 0.46364761,
                      0.46364761, 1.57079633, 1.57079633, 1.57079633, 1.10714872, 1.10714872, 1.10714872, 0.78539816, 0.78539816, 0.78539816]
        z_ref      = [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]
      elif comm.size == 2:
        if comm.rank == 0: 
          radius_ref = [0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1., 0.5, 0.5, 0.5, 0.70710678, 0.70710678, 0.70710678, 1.11803399, 1.11803399, 1.11803399]
          theta_ref  = [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.57079633, 1.57079633, 1.57079633, 0.78539816, 0.78539816, 0.78539816, 0.46364761, 0.46364761, 0.46364761]
          z_ref      = [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]
        elif comm.rank == 1:
          radius_ref = [0.5, 0.5, 0.5, 0.70710678, 0.70710678, 0.70710678, 1.11803399, 1.11803399, 1.11803399, 
                        1., 1., 1., 1.11803399, 1.11803399, 1.11803399, 1.41421356, 1.41421356, 1.41421356]
          theta_ref  = [1.57079633, 1.57079633, 1.57079633, 0.78539816, 0.78539816, 0.78539816, 0.46364761, 0.46364761, 0.46364761, 
                        1.57079633, 1.57079633, 1.57079633, 1.10714872, 1.10714872, 1.10714872, 0.78539816, 0.78539816, 0.78539816]
          z_ref      = [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]

      for zone in PT.get_all_Zone_t(part_tree):
        # Recover coordinates and fields in the new basis
        gc_n = PT.get_nodes_from_predicates(zone, 'GridCoordinates/DataArray_t')
        gc_coords = [PT.get_value(n) for n in gc_n]
        fs_n  = PT.get_nodes_from_predicates(zone, 'FlowSolution/DataArray_t')
        fs_coords = [PT.get_value(n) for n in fs_n]
        zsr_n  = PT.get_nodes_from_predicates(zone, 'ZoneSubRegion/DataArray_t')
        zsr_coords = [PT.get_value(n) for n in zsr_n]
        dd_n  = PT.get_nodes_from_predicates(zone, 'DiscreteData/DataArray_t')
        dd_coords = [PT.get_value(n) for n in dd_n]

        assert np.allclose(radius_ref, gc_coords[1].flatten('F'))
        assert np.allclose(radius_ref, fs_coords[1].flatten('F'))
        assert np.allclose(radius_ref, zsr_coords[1].flatten('F'))
        assert np.allclose(radius_ref, dd_coords[1].flatten('F'))

        assert np.allclose(theta_ref, gc_coords[2].flatten('F'))
        assert np.allclose(theta_ref, fs_coords[2].flatten('F'))
        assert np.allclose(theta_ref, zsr_coords[2].flatten('F'))
        assert np.allclose(theta_ref, dd_coords[2].flatten('F'))

        assert np.allclose(z_ref, gc_coords[0].flatten('F'))
        assert np.allclose(z_ref, fs_coords[0].flatten('F'))
        assert np.allclose(z_ref, zsr_coords[0].flatten('F'))
        assert np.allclose(z_ref, dd_coords[0].flatten('F'))

        with pytest.raises(AssertionError):
          transform.cartesian_to_cylindric_from_unit_revolution_axis(part_tree, revolution_axis=(0, 0, 0))

#%%%
@pytest_parallel.mark.parallel([1, 2]) 
class Test_cart_to_cyl:
  revolution_axis = [1, 1, 0] 
  def test_cartesian_to_cylindric_S(self, comm):

    import maia
    import maia.pytree as PT
    import numpy as np
    import copy
    
    dist_tree = maia.factory.generate_dist_block(3, 'S', comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover the intial cartesian coordinates
      coords = PT.Zone.coordinates(zone)
      children = copy.deepcopy(PT.get_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'))
      PT.set_name(children[0], 'DDX')
      PT.set_name(children[1], 'DDY')
      PT.set_name(children[2], 'DDZ')
      
      # Create fields in zone 
      PT.new_FlowSolution('FlowSolution', fields={'FSX' : coords[0], 'FSY' : coords[1], 'FSZ' : coords[2]}, parent=zone)
      PT.new_ZoneSubRegion('ZoneSubRegion', fields={'ZSRX' : coords[0], 'ZSRY' : coords[1], 'ZSRZ' : coords[2]}, parent=zone)
      PT.new_node('DiscreteData', 'DiscreteData_t', children=children, parent=zone)

    # Copy of the cartesian part tree
    part_tree_cart_ref = copy.deepcopy(part_tree)

    # Transform cartesian coordinates and fields into cylindric from any revolution axis
    transform.cartesian_to_cylindric(part_tree, revolution_axis=self.revolution_axis, gc_name='GridCoordinatesBis')
    assert PT.is_same_tree(part_tree_cart_ref, part_tree)
    transform.cartesian_to_cylindric(part_tree, revolution_axis=self.revolution_axis)  
     
    if comm.size == 1:
      radius_ref = [0., 0.5, 1., 0.5, 0., 0.5, 1., 0.5, 0., 1., 1.11803399, 1.41421356, 1.11803399, 1., 1.11803399, 1.41421356, 
                    1.11803399, 1., 2., 2.06155281, 2.23606798, 2.06155281, 2., 2.06155281, 2.23606798, 2.06155281, 2.]
      theta_ref  = [0., 3.14159265, 3.14159265, 0., 0., 3.14159265, 0., 0., 0., 1.57079633, 2.03444394, 2.35619449, 1.10714872, 1.57079633, 2.03444394, 0.78539816, 
                    1.10714872, 1.57079633, 1.57079633, 1.81577499, 2.03444394, 1.32581766, 1.57079633, 1.81577499, 1.10714872, 1.32581766, 1.57079633]
      z_ref      = [0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2., 0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2., 0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2.]
    elif comm.size == 2:
      if comm.rank ==0 :
        radius_ref = [0., 0.5, 0.5, 0., 1., 0.5, 1., 1.11803399, 1.11803399, 1., 1.41421356, 1.11803399, 2., 2.06155281, 2.06155281, 2., 2.23606798, 2.06155281]
        theta_ref  = [0., 3.14159265, 0., 0., 0., 0., 1.57079633, 2.03444394, 1.10714872, 1.57079633, 0.78539816, 
                      1.10714872, 1.57079633, 1.81577499, 1.32581766, 1.57079633, 1.10714872, 1.32581766]
        z_ref      = [0., 0.5, 0.5, 1., 1., 1.5, 0., 0.5, 0.5, 1., 1., 1.5, 0., 0.5, 0.5, 1., 1., 1.5]
      elif comm.rank == 1:
        radius_ref = [0.5, 1., 0., 0.5, 0.5, 0., 1.11803399, 1.41421356, 1., 1.11803399, 1.11803399, 1., 2.06155281, 2.23606798, 2., 2.06155281, 2.06155281, 2.]
        theta_ref  = [3.14159265, 3.14159265, 0., 3.14159265, 0., 0., 2.03444394, 2.35619449, 1.57079633, 2.03444394,
                      1.10714872, 1.57079633, 1.81577499, 2.03444394, 1.57079633, 1.81577499, 1.32581766, 1.57079633]
        z_ref      = [0.5, 1., 1., 1.5, 1.5, 2., 0.5, 1., 1., 1.5, 1.5, 2., 0.5, 1., 1., 1.5, 1.5, 2.]

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover coordinates and fields in the new basis
      gc_n = PT.get_nodes_from_predicates(zone, 'GridCoordinates/DataArray_t')
      gc_coords = [PT.get_value(n) for n in gc_n]
      fs_n  = PT.get_nodes_from_predicates(zone, 'FlowSolution/DataArray_t')
      fs_coords = [PT.get_value(n) for n in fs_n]
      zsr_n  = PT.get_nodes_from_predicates(zone, 'ZoneSubRegion/DataArray_t')
      zsr_coords = [PT.get_value(n) for n in zsr_n]
      dd_n  = PT.get_nodes_from_predicates(zone, 'DiscreteData/DataArray_t')
      dd_coords = [PT.get_value(n) for n in dd_n]

      assert np.allclose(radius_ref, gc_coords[1].flatten('F'))
      assert np.allclose(radius_ref, fs_coords[1].flatten('F'))
      assert np.allclose(radius_ref, zsr_coords[1].flatten('F'))
      assert np.allclose(radius_ref, dd_coords[1].flatten('F'))

      assert np.allclose(theta_ref, gc_coords[2].flatten('F'))
      assert np.allclose(theta_ref, fs_coords[2].flatten('F'))
      assert np.allclose(theta_ref, zsr_coords[2].flatten('F'))
      assert np.allclose(theta_ref, dd_coords[2].flatten('F'))

      assert np.allclose(z_ref, gc_coords[0].flatten('F'))
      assert np.allclose(z_ref, fs_coords[0].flatten('F'))
      assert np.allclose(z_ref, zsr_coords[0].flatten('F'))
      assert np.allclose(z_ref, dd_coords[0].flatten('F'))

      with pytest.raises(AssertionError):
        transform.cartesian_to_cylindric(part_tree, revolution_axis=(0, 0, 0))

  def test_cartesian_to_cylindric_U(self, comm):

      import maia
      import maia.pytree as PT
      import numpy as np
      import copy
      
      dist_tree = maia.factory.generate_dist_block(3, 'Poly', comm)
      part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
      
      for zone in PT.get_all_Zone_t(part_tree):
        # Recover the intial cartesian coordinates
        coords = PT.Zone.coordinates(zone)
        children = copy.deepcopy(PT.get_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'))
        PT.set_name(children[0], 'DDX')
        PT.set_name(children[1], 'DDY')
        PT.set_name(children[2], 'DDZ')
        
        # Create fields in zone 
        PT.new_FlowSolution('FlowSolution', fields={'FSX' : coords[0], 'FSY' : coords[1], 'FSZ' : coords[2]}, parent=zone)
        PT.new_ZoneSubRegion('ZoneSubRegion', fields={'ZSRX' : coords[0], 'ZSRY' : coords[1], 'ZSRZ' : coords[2]}, parent=zone)
        PT.new_node('DiscreteData', 'DiscreteData_t', children=children, parent=zone)
      
      # Copy of the cartesian part tree
      part_tree_cart_ref = copy.deepcopy(part_tree)

      # Transform cartesian coordinates and fields into cylindric from any revolution axis
      transform.cartesian_to_cylindric(part_tree, revolution_axis=self.revolution_axis, gc_name='GridCoordinatesBis')
      assert PT.is_same_tree(part_tree_cart_ref, part_tree)
      transform.cartesian_to_cylindric(part_tree, revolution_axis=self.revolution_axis)

      if comm.size == 1:
        radius_ref = [0., 0.5, 1., 0.5, 0., 0.5, 1., 0.5, 0., 1., 1.11803399, 1.41421356, 1.11803399, 1., 1.11803399, 1.41421356, 
                      1.11803399, 1., 2., 2.06155281, 2.23606798, 2.06155281, 2., 2.06155281, 2.23606798, 2.06155281, 2.]
        theta_ref  = [0., 3.14159265, 3.14159265, 0., 0., 3.14159265, 0., 0., 0., 1.57079633, 2.03444394, 2.35619449, 1.10714872, 1.57079633, 2.03444394, 0.78539816,
                      1.10714872, 1.57079633, 1.57079633, 1.81577499, 2.03444394, 1.32581766, 1.57079633, 1.81577499, 1.10714872, 1.32581766, 1.57079633]
        z_ref      = [0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2., 0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2., 0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2.]
      elif comm.size == 2:
        if comm.rank == 0 :
          radius_ref = [0., 0.5, 1., 0.5, 0., 0.5, 1., 0.5, 0., 1., 1.11803399, 1.41421356, 1.11803399, 1., 1.11803399, 1.41421356, 1.11803399, 1.]
          theta_ref  = [0., 3.14159265, 3.14159265, 0., 0., 3.14159265, 0., 0., 0., 1.57079633, 2.03444394, 
                        2.35619449, 1.10714872, 1.57079633, 2.03444394, 0.78539816, 1.10714872, 1.57079633]
          z_ref      = [0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2., 0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2.]
        elif comm.rank == 1:
          radius_ref = [1., 1.11803399, 1.41421356, 1.11803399, 1., 1.11803399, 1.41421356, 1.11803399, 1., 2., 
                        2.06155281, 2.23606798, 2.06155281, 2., 2.06155281, 2.23606798, 2.06155281, 2.]
          theta_ref  = [1.57079633, 2.03444394, 2.35619449, 1.10714872, 1.57079633, 2.03444394, 0.78539816, 1.10714872, 1.57079633,
                        1.57079633, 1.81577499, 2.03444394, 1.32581766, 1.57079633, 1.81577499, 1.10714872, 1.32581766, 1.57079633]
          z_ref      = [0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2., 0., 0.5, 1., 0.5, 1., 1.5, 1., 1.5, 2.]
        
      for zone in PT.get_all_Zone_t(part_tree):
        # Recover coordinates and fields in the new basis
        gc_n = PT.get_nodes_from_predicates(zone, 'GridCoordinates/DataArray_t')
        gc_coords = [PT.get_value(n) for n in gc_n]
        fs_n  = PT.get_nodes_from_predicates(zone, 'FlowSolution/DataArray_t')
        fs_coords = [PT.get_value(n) for n in fs_n]
        zsr_n  = PT.get_nodes_from_predicates(zone, 'ZoneSubRegion/DataArray_t')
        zsr_coords = [PT.get_value(n) for n in zsr_n]
        dd_n  = PT.get_nodes_from_predicates(zone, 'DiscreteData/DataArray_t')
        dd_coords = [PT.get_value(n) for n in dd_n]

        assert np.allclose(radius_ref, gc_coords[1].flatten('F'))
        assert np.allclose(radius_ref, fs_coords[1].flatten('F'))
        assert np.allclose(radius_ref, zsr_coords[1].flatten('F'))
        assert np.allclose(radius_ref, dd_coords[1].flatten('F'))

        assert np.allclose(theta_ref, gc_coords[2].flatten('F'))
        assert np.allclose(theta_ref, fs_coords[2].flatten('F'))
        assert np.allclose(theta_ref, zsr_coords[2].flatten('F'))
        assert np.allclose(theta_ref, dd_coords[2].flatten('F'))

        assert np.allclose(z_ref, gc_coords[0].flatten('F'))
        assert np.allclose(z_ref, fs_coords[0].flatten('F'))
        assert np.allclose(z_ref, zsr_coords[0].flatten('F'))
        assert np.allclose(z_ref, dd_coords[0].flatten('F'))
        
        with pytest.raises(AssertionError):
          transform.cartesian_to_cylindric(part_tree, revolution_axis=(0, 0, 0))

@pytest.mark.parametrize('zonetype', ['S', 'Poly'])
class Test_transform:
  revolution_axis = (0, 0, 1)

  def test_transform_cartesian_to_cylindric_unit(self, zonetype, comm):

    import maia
    import maia.pytree as PT
    import numpy as np
    import copy
      
    dist_tree = maia.factory.generate_dist_block(3, zonetype, comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover the intial cartesian coordinates
      zone_ref = copy.deepcopy(zone)
      coords = PT.Zone.coordinates(zone)
      children = copy.deepcopy(PT.get_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'))
      PT.set_name(children[0], 'DDX')
      PT.set_name(children[1], 'DDY')
      PT.set_name(children[2], 'DDZ')
        
      # Create fields in zone 
      PT.new_FlowSolution('FlowSolution', fields={'FSX' : coords[0], 'FSY' : coords[1], 'FSZ' : coords[2]}, parent=zone)
      PT.new_ZoneSubRegion('ZoneSubRegion', fields={'ZSRX' : coords[0], 'ZSRY' : coords[1], 'ZSRZ' : coords[2]}, parent=zone)
      PT.new_node('DiscreteData', 'DiscreteData_t', children=children, parent=zone)

      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis, gc_name='GridCoordinatesBis')

      PT.is_same_node(zone_ref, zone)

      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis)
      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis, name='FlowSolution', basename='FS')
      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis, name='ZoneSubRegion', basename='ZSR')
      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis, name='DiscreteData', basename='DD')

      if comm.size == 1:
        radius_ref = [0., 0.5, 1., 0.5, 0.70710678, 1.11803399, 1., 1.11803399, 1.41421356, 0., 0.5, 1., 0.5, 0.70710678, 1.11803399, 
                      1., 1.11803399, 1.41421356, 0., 0.5, 1., 0.5, 0.70710678, 1.11803399, 1., 1.11803399, 1.41421356]
        theta_ref  = [0., 0., 0., 1.57079633, 0.78539816, 0.46364761, 1.57079633, 1.10714872, 0.78539816, 0., 0., 0., 1.57079633, 0.78539816, 0.46364761, 
                      1.57079633, 1.10714872, 0.78539816, 0., 0., 0., 1.57079633, 0.78539816, 0.46364761, 1.57079633, 1.10714872, 0.78539816]
        z_ref      = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1., 1., 1., 1., 1., 1., 1., 1.]

      # Recover coordinates and fields in the new basis
      gc_n = PT.get_nodes_from_predicates(zone, 'GridCoordinates/DataArray_t')
      gc_coords = [PT.get_value(n) for n in gc_n]
      fs_n  = PT.get_nodes_from_predicates(zone, 'FlowSolution/DataArray_t')
      fs_coords = [PT.get_value(n) for n in fs_n]
      zsr_n  = PT.get_nodes_from_predicates(zone, 'ZoneSubRegion/DataArray_t')
      zsr_coords = [PT.get_value(n) for n in zsr_n]
      dd_n  = PT.get_nodes_from_predicates(zone, 'DiscreteData/DataArray_t')
      dd_coords = [PT.get_value(n) for n in dd_n]

      assert np.allclose(radius_ref, gc_coords[0].flatten('F'))
      assert np.allclose(radius_ref, fs_coords[0].flatten('F'))
      assert np.allclose(radius_ref, zsr_coords[0].flatten('F'))
      assert np.allclose(radius_ref, dd_coords[0].flatten('F'))

      assert np.allclose(theta_ref, gc_coords[1].flatten('F'))
      assert np.allclose(theta_ref, fs_coords[1].flatten('F'))
      assert np.allclose(theta_ref, zsr_coords[1].flatten('F'))
      assert np.allclose(theta_ref, dd_coords[1].flatten('F'))

      assert np.allclose(z_ref, gc_coords[2].flatten('F'))
      assert np.allclose(z_ref, fs_coords[2].flatten('F'))
      assert np.allclose(z_ref, zsr_coords[2].flatten('F'))
      assert np.allclose(z_ref, dd_coords[2].flatten('F'))

      with pytest.raises(AssertionError):
        transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=(1, 2, 3))

  def test_transform_cylindric_to_cartesian_unit(self, zonetype, comm):

    import maia
    import maia.pytree as PT
    import numpy as np
    import copy
      
    dist_tree = maia.factory.generate_dist_block(3, zonetype, comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover the intial cartesian coordinates
      zone_ref = copy.deepcopy(zone)
      coords = PT.Zone.coordinates(zone)
      children = copy.deepcopy(PT.get_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'))
      PT.set_name(children[0], 'DDX')
      PT.set_name(children[1], 'DDY')
      PT.set_name(children[2], 'DDZ')
        
      # Create fields in zone 
      PT.new_FlowSolution('FlowSolution', fields={'FSX' : coords[0], 'FSY' : coords[1], 'FSZ' : coords[2]}, parent=zone)
      PT.new_ZoneSubRegion('ZoneSubRegion', fields={'ZSRX' : coords[0], 'ZSRY' : coords[1], 'ZSRZ' : coords[2]}, parent=zone)
      PT.new_node('DiscreteData', 'DiscreteData_t', children=children, parent=zone)

      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis, gc_name='GridCoordinatesBis')

      PT.is_same_node(zone_ref, zone)

      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis)
      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis, name='FlowSolution', basename='FS')
      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis, name='ZoneSubRegion', basename='ZSR')
      transform._transform_cartesian_to_cylindric_unit(zone, revolution_axis=self.revolution_axis, name='DiscreteData', basename='DD')

      transform._transform_cylindric_to_cartesian_unit(zone, revolution_axis=self.revolution_axis, gc_name='GridCoordinatesBis')

      PT.is_same_node(zone_ref, zone)

      transform._transform_cylindric_to_cartesian_unit(zone, revolution_axis=self.revolution_axis)
      transform._transform_cylindric_to_cartesian_unit(zone, revolution_axis=self.revolution_axis, name='FlowSolution', basename='FS')
      transform._transform_cylindric_to_cartesian_unit(zone, revolution_axis=self.revolution_axis, name='ZoneSubRegion', basename='ZSR')
      transform._transform_cylindric_to_cartesian_unit(zone, revolution_axis=self.revolution_axis, name='DiscreteData', basename='DD')


      # Recover coordinates and fields in the new basis
      gc_n = PT.get_nodes_from_predicates(zone, 'GridCoordinates/DataArray_t')
      gc_coords = [PT.get_value(n) for n in gc_n]
      fs_n  = PT.get_nodes_from_predicates(zone, 'FlowSolution/DataArray_t')
      fs_coords = [PT.get_value(n) for n in fs_n]
      zsr_n  = PT.get_nodes_from_predicates(zone, 'ZoneSubRegion/DataArray_t')
      zsr_coords = [PT.get_value(n) for n in zsr_n]
      dd_n  = PT.get_nodes_from_predicates(zone, 'DiscreteData/DataArray_t')
      dd_coords = [PT.get_value(n) for n in dd_n]

      assert np.allclose(coords[0], gc_coords[0])
      assert np.allclose(coords[0], fs_coords[0])
      assert np.allclose(coords[0], zsr_coords[0])
      assert np.allclose(coords[0], dd_coords[0])

      assert np.allclose(coords[1], gc_coords[1])
      assert np.allclose(coords[1], fs_coords[1])
      assert np.allclose(coords[1], zsr_coords[1])
      assert np.allclose(coords[1], dd_coords[1])

      assert np.allclose(coords[2], gc_coords[2])
      assert np.allclose(coords[2], fs_coords[2])
      assert np.allclose(coords[2], zsr_coords[2])
      assert np.allclose(coords[2], dd_coords[2])

      with pytest.raises(AssertionError):
        transform._transform_cylindric_to_cartesian_unit(zone, revolution_axis=(1, 2, 3))

