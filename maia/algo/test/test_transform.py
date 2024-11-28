import pytest
import pytest_parallel
import os
import numpy as np

import maia.pytree          as PT
import maia.pytree.maia     as MT
from maia.utils             import np_utils, par_utils, test_utils

import maia
from maia.factory.dcube_generator import dcube_generate

from maia.algo import transform

from maia.utils import logging as mlog

def check_perio(tree, jn_name, tol=1e-8):
  # Check if applying the transformation gives the opposite face center
  # Works only in sequential context, for FaceCenter GCs, single zone meshes
  z = PT.get_all_Zone_t(tree)[0]
  r1 = PT.get_node_from_name(tree, jn_name)
  pl1 = PT.get_child_from_name(r1, 'PointList')[1][0]
  pl2 = PT.get_child_from_name(r1, 'PointListDonor')[1][0]
  from maia.algo.part.geometry import _compute_elements_center
  center_face = _compute_elements_center(z, 2)
  center_face_x = center_face[0::3]
  center_face_y = center_face[1::3]
  center_face_z = center_face[2::3]
  center_pl1_x = center_face_x[pl1-1]
  center_pl1_y = center_face_y[pl1-1]
  center_pl1_z = center_face_z[pl1-1]
  center_pl2_x = center_face_x[pl2-1]
  center_pl2_y = center_face_y[pl2-1]
  center_pl2_z = center_face_z[pl2-1]
  gc_center, gc_angle, gc_trans = PT.GridConnectivity.periodic_values(r1)
  transfo = np_utils.transform_cart_vectors(center_pl1_x, center_pl1_y, center_pl1_z,
                                            gc_trans, gc_center, gc_angle)
  diff = np.sqrt((transfo[0] - center_pl2_x)**2 + (transfo[1] - center_pl2_y)**2 + (transfo[2] - center_pl2_z)**2)
  assert (diff < tol).all()


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
  zone            = PT.yaml.to_node(yz)
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
  transform.transform_affine(dist_zone, rotation_angle=np.array([0.,0.,np.pi]))

  check_vect_field(dist_zone_ini, dist_zone, "Coordinate")
  check_vect_field(dist_zone_ini, dist_zone, "field")
  check_scal_field(dist_zone_ini, dist_zone, "scalar")

@pytest_parallel.mark.parallel(1)
class Test_transform_affine_gc:

  @pytest.mark.parametrize('rotation', [(np.pi/2, 0, 0), (0,np.pi/2,0), (np.pi/2, np.pi/2,0)])
  def test_basic(self, rotation, comm):
    # This mesh has a simple Translation JN
    fname = os.path.join(test_utils.mesh_dir, 'cube_bcdataset_and_periodic.yaml')

    tree = maia.io.file_to_dist_tree(fname, comm)
    maia.algo.transform_affine(tree, rotation_angle=rotation, translation=[1,2,3]) #Translation should have no effect
    zmin_jn = PT.get_node_from_name(tree, 'Zmin_match')
    zmax_jn = PT.get_node_from_name(tree, 'Zmax_match')
    centermin, anglemin, transmin = PT.GridConnectivity.periodic_values(zmin_jn)
    centermax, anglemax, transmax = PT.GridConnectivity.periodic_values(zmax_jn)
    if rotation == (np.pi/2, 0, 0):
      expt_trans_min = [0,-1,0]
      expt_trans_max = [0,1,0]
    elif rotation == (0, np.pi/2, 0) or rotation == (np.pi/2, np.pi/2, 0):
      expt_trans_min = [1, 0,0]
      expt_trans_max = [-1,0,0]
    assert np.allclose(transmin, expt_trans_min) and np.allclose(anglemin, [0,0,0])
    assert np.allclose(transmax, expt_trans_max) and np.allclose(anglemax, [0,0,0])
  

  def test_full_transfo(self, comm):
    # This mesh has one Rotation JN (with RotCenter != 0) and one translation JN; with poor precision
    fname = os.path.join(test_utils.sample_mesh_dir, 'quarter_crown_square_8.yaml')
    JNS = ['MatchTranslationA', 'MatchTranslationB', 'MatchRotationA', 'MatchRotationB']

    tree = maia.io.file_to_dist_tree(fname, comm)
    # Lets apply a crazy transformation
    maia.algo.transform_affine(tree, rotation_angle=[np.pi/4, -np.pi/3, np.pi/2], rotation_center=[-1,0,1], translation=[4,3,2]) 
    for name in JNS:
        check_perio(tree, name, tol=5e-6)
      
  def test_2d(self, comm):
    tree = maia.factory.generate_dist_block([5,5], 'S', comm, origin=[0,0])
    xmin = PT.get_node_from_name(tree, 'Xmin')
    xmax = PT.get_node_from_name(tree, 'Xmax')
    
    PT.rm_nodes_from_name(tree, 'Xm*')
    PT.update_node(xmin, label='GridConnectivity1to1_t', value='zone')
    PT.new_IndexRange('PointRangeDonor', [[5,5],[1,5]], parent=xmin)
    PT.new_GridConnectivityProperty(periodic={'translation' : [1.,0], 'rotation_center':[0.,0], 'rotation_angle':[0.,0]}, 
                                    parent=xmin)
    PT.update_node(xmax, label='GridConnectivity1to1_t', value='zone')
    PT.new_IndexRange('PointRangeDonor', [[1,1],[1,5]], parent=xmax)
    PT.new_GridConnectivityProperty(periodic={'translation' : [-1.,0], 'rotation_center':[0.,0], 'rotation_angle':[0.,0]}, 
                                    parent=xmax)

    zgc = PT.new_ZoneGridConnectivity(parent=PT.get_node_from_label(tree, 'Zone_t'))
    PT.set_children(zgc, [xmin, xmax])

    maia.algo.transform_affine(tree, rotation_angle=np.pi/4, translation=np.zeros(2), rotation_center=np.zeros(2))

    assert np.allclose(PT.get_node_from_name(xmin, 'RotationAngle')[1], np.zeros(2))
    assert np.allclose(PT.get_node_from_name(xmin, 'RotationCenter')[1],np.zeros(2))
    assert np.allclose(PT.get_node_from_name(xmin, 'Translation')[1], [np.sqrt(2)/2, np.sqrt(2)/2])
    assert np.allclose(PT.get_node_from_name(xmax, 'RotationAngle')[1], np.zeros(2))
    assert np.allclose(PT.get_node_from_name(xmax, 'RotationCenter')[1],np.zeros(2))
    assert np.allclose(PT.get_node_from_name(xmax, 'Translation')[1], [-np.sqrt(2)/2, -np.sqrt(2)/2])

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
  transform.transform_affine(dist_zone, rotation_center=np.zeros(2), translation=np.zeros(2), rotation_angle=np.pi)
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
  transform.transform_affine(part_tree, rotation_center=np.zeros(2), translation=np.zeros(2), rotation_angle=0.5*np.pi)
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

@pytest.mark.parametrize('revolution_axis', [(0, 1, 0), [1, 2, 3]])
@pytest.mark.parametrize('zonetype', ['S', 'Poly'])       
@pytest.mark.parametrize('partitioned', [False, True])
@pytest_parallel.mark.parallel(2)
class Test_change_basis_simple:
  def test_auxiliary_coords(self, zonetype, partitioned, revolution_axis, comm):

      dist_tree = maia.factory.generate_dist_block(4, zonetype, comm)
      if partitioned:
        part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
      else:
        part_tree = dist_tree

      for zone in PT.get_all_Zone_t(part_tree):
        # Recover the intial cartesian coordinates
        coords = PT.Zone.coordinates(zone)

        PT.new_FlowSolution('FlowSolution', fields={f'FS{d}' : coords[i].copy() for i,d in enumerate(['X', 'Y', 'Z'])}, parent=zone)
        PT.new_ZoneSubRegion('ZoneSubRegion', fields={f'ZSR{d}' : coords[i].copy() for i,d in enumerate(['X', 'Y', 'Z'])}, parent=zone)

      # Create the transform matrix and the reverse transform matrix 
      transform_matrix = np_utils.create_transform_matrix(revolution_axis=revolution_axis)
      
      part_tree_cart_ref = PT.deep_copy(part_tree)
        
      transform.auxiliary_coords_system(part_tree, transform_matrix)
      transform.auxiliary_coords_system(part_tree, transform_matrix)

      # Compute the former coordinates in the former basis
      transform.auxiliary_coords_system(part_tree, None)
      assert PT.is_same_tree(part_tree_cart_ref, part_tree, abs_tol=1e-10)

  def test_cyl_cart(self, zonetype, partitioned, revolution_axis, comm):

    dist_tree = maia.factory.generate_dist_block(3, zonetype, comm)
    if partitioned:
      part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
    else:
      part_tree = dist_tree

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover the intial cartesian coordinates
      coords = PT.Zone.coordinates(zone)
      n_cell = PT.Zone.CellSize(zone) if partitioned else np.diff(MT.get_distribution(zone, 'Cell')[1])[0]
      n_cell = n_cell.tolist() if isinstance(n_cell, np.ndarray) else [n_cell]
      n_vtx = PT.Zone.VertexSize(zone) if partitioned else np.diff(MT.get_distribution(zone, 'Vertex')[1])[0]
      n_vtx = n_vtx.tolist() if isinstance(n_vtx, np.ndarray) else [n_vtx]

      # Create fields in zone
      PT.new_FlowSolution('FlowSolution', fields={f'Coordinate{d}' : coords[i].copy() for i,d in enumerate(['X', 'Y', 'Z'])}, parent=zone)
      dd = PT.new_ZoneSubRegion('DiscreteData', fields={f'Coordinate{d}' : coords[i].copy() for i,d in enumerate(['X', 'Y', 'Z'])}, parent=zone)
      PT.set_label(dd, 'DiscreteData_t')
      PT.new_ZoneSubRegion('SubRegionCC', loc='CellCenter', fields={f'Field{d}' : np.random.rand(*n_cell) for d in ['X', 'Y', 'Z']}, parent=zone),
      PT.new_ZoneSubRegion('SubRegionVtx', loc='Vertex', fields={f'Field{d}' : np.random.rand(*n_vtx) for d in ['X', 'Y', 'Z']}, parent=zone)
      if PT.Zone.Type(zone) == 'Unstructured':
        n_face = PT.Zone.n_face(zone) if partitioned else np.diff(MT.getDistribution(PT.Zone.NGonNode(zone), 'Element')[1])[0]
        PT.new_ZoneSubRegion('SubRegionFace', loc='FaceCenter', fields={f'Field{d}' : np.random.rand(n_face) for d in ['X', 'Y', 'Z']}, parent=zone)
        bc = PT.get_node_from_name(zone, 'Xmax')
        if bc is not None:
          pl = PT.get_child_from_name(bc, 'PointList')[1] 
          bcds = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
          bcda = PT.new_child(bcds, 'DirichletData', 'BCData_t')
          for name in['Scalar', 'FieldX', 'FieldY', 'FieldZ']: 
            PT.new_DataArray(name, np.random.rand(pl.size), parent=bcda)
      else: # Structured:
        bc = PT.get_node_from_name(zone, 'Xmax')
        if bc is not None:
          bc_size = PT.Subset.n_elem(bc) if partitioned else np.diff(MT.getDistribution(bc, 'Index')[1])[0]
          bcds = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
          bcda = PT.new_child(bcds, 'DirichletData', 'BCData_t')
          for name in['Scalar', 'FieldX', 'FieldY', 'FieldZ']: 
            PT.new_DataArray(name, np.random.rand(bc_size), parent=bcda)

        bc = PT.get_node_from_name(zone, 'Ymin') # Transform to JFaceCenter
        if bc is not None:
          PT.update_child(bc, 'GridLocation', value='JFaceCenter')
          if partitioned:
            pr_face = [[1,1],[1,1],[1,2]]
          else:
            pr_face = [[1,2],[1,1],[1,2]]
            MT.new_distribution({'Index' : par_utils.uniform_distribution(4, comm)}, bc)
          PT.update_child(bc, 'PointRange', value=pr_face)
          bc_size = PT.Subset.n_elem(bc) if partitioned else np.diff(MT.getDistribution(bc, 'Index')[1])[0]
          bcds = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
          bcda = PT.new_child(bcds, 'DirichletData', 'BCData_t')
          for name in['Scalar', 'FieldX', 'FieldY', 'FieldZ']: 
            PT.new_DataArray(name, np.random.rand(bc_size), parent=bcda)

    if revolution_axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
      cart2cyl = transform.cartesian_to_cylindrical_from_unit_revolution_axis
      cyl2cart = transform.cylindrical_to_cartesian_from_unit_revolution_axis
    else: 
      cart2cyl = transform.cartesian_to_cylindrical
      cyl2cart = transform.cylindrical_to_cartesian

    # Transform cartesian coordinates and fields into cylindric around a unit revolution axis
    tree_bck = PT.deep_copy(part_tree)
    cart2cyl(part_tree, revolution_axis, comm, True)
    # Transform cylindric coordinates and fields into cartesian around a unit revolution axis
    cyl2cart(part_tree, revolution_axis, comm, True)
    
    for zone in PT.get_all_Zone_t(part_tree):
      # Recover coordinates and fields in the new basis
      for container_name in ['GridCoordinates', 'FlowSolution', 'DiscreteData']:
        container = PT.get_child_from_name(zone, container_name)
        val_x, val_y, val_z = [PT.get_node_from_name(container, f'*{d}')[1] for d in ['X', 'Y', 'Z']]
        assert np.allclose(coords[0], val_x)
        assert np.allclose(coords[1], val_y)
        assert np.allclose(coords[2], val_z)
      
    assert PT.is_same_tree(part_tree, tree_bck, abs_tol=1e-12)

@pytest.mark.parametrize('revolution_axis', [(1, 1, 0), [2, 2, 0]])
@pytest_parallel.mark.parallel([1, 2]) 
class Test_cart_to_cyl:
  def test_S(self, revolution_axis, comm):

    dist_tree = maia.factory.generate_dist_block([3,2,2], 'S', comm)
    weights = maia.factory.partitioning.compute_regular_weights(dist_tree, comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=weights)

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover the intial cartesian coordinates
      coords = PT.Zone.coordinates(zone)
      
      # Create fields in zone 
      PT.new_FlowSolution('FlowSolution', fields={f'Coordinate{d}' : coords[i].copy() for i,d in enumerate(['X', 'Y', 'Z'])}, parent=zone)
      dd = PT.new_ZoneSubRegion('DiscreteData', fields={f'Coordinate{d}' : coords[i].copy() for i,d in enumerate(['X', 'Y', 'Z'])}, parent=zone)
      PT.set_label(dd, 'DiscreteData_t')

    # Transform cartesian coordinates and fields into cylindric from any revolution axis
    transform.cartesian_to_cylindrical(part_tree, revolution_axis)  
     
    if comm.size == 1:
      radius_ref = np.array([[[0. , 1.41421356], [1. ,  1.73205081]], [[0.5, 1.5       ], [0.5,  1.5       ]], [[1. , 1.73205081], [0. ,  1.41421356]]])
      theta_ref  = np.array([[[0., 1.57079633], [0., 0.95531662]], [[3.14159265, 1.91063324], [0., 1.23095942]], [[3.14159265, 2.18627604], [0., 1.57079633]]])
      z_ref      = np.array([[[0., 0.], [0.70710678, 0.70710678]], [[0.35355339, 0.35355339], [1.06066017, 1.06066017]], [[0.70710678, 0.70710678], [1.41421356, 1.41421356]]])
    elif comm.size == 2:
      if comm.rank == 0 :
        radius_ref = np.array([[[0., 1.41421356], [1.        , 1.73205081]], [[0.5       , 1.5       ], [0.5       , 1.5       ]]])
        theta_ref  = np.array([[[0., 1.57079633], [0.        , 0.95531662]], [[3.14159265, 1.91063324], [0.        , 1.23095942]]])
        z_ref      = np.array([[[0., 0.        ], [0.70710678, 0.70710678]], [[0.35355339, 0.35355339], [1.06066017, 1.06066017]]])
      elif comm.rank == 1:
        radius_ref = np.array([[[0.5       , 1.5       ], [0.5       , 1.5       ]], [[1.        , 1.73205081], [0.        , 1.41421356]]])
        theta_ref  = np.array([[[3.14159265, 1.91063324], [0.        , 1.23095942]], [[3.14159265, 2.18627604], [0.        , 1.57079633]]])
        z_ref      = np.array([[[0.35355339, 0.35355339], [1.06066017, 1.06066017]], [[0.70710678, 0.70710678], [1.41421356, 1.41421356]]])

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover coordinates and fields in the new basis
      for container_name in ['GridCoordinates', 'FlowSolution', 'DiscreteData']:
        container = PT.get_child_from_name(zone, container_name)
        val_r, val_theta, val_z = [PT.get_node_from_name(container, f'*{d}')[1] for d in ['R', 'Theta', 'Z']]

        assert np.allclose(radius_ref, val_r)
        assert np.allclose(theta_ref, val_theta)
        assert np.allclose(z_ref, val_z)

  def test_U(self, revolution_axis, comm):
    # NB : test with 1 rank is done on distributed mesh

    dist_tree = maia.factory.generate_dist_block(3, 'Poly', comm)
    if comm.Get_size() > 1:
      part_tree = maia.factory.partition_dist_tree(dist_tree, comm, graph_part_tool='gnum')
    else:
      part_tree = dist_tree

    for zone in PT.get_all_Zone_t(part_tree):
      # Recover the intial cartesian coordinates
      coords = PT.Zone.coordinates(zone)
      n_vtx = PT.Zone.n_vtx(zone)
      
      # Create fields in zone 
      fields = {f'Coordinate{d}' : coords[i].copy() for i,d in enumerate(['X', 'Y', 'Z'])} # Coords -> use coordinates formulae
      fields.update({'Scalar' : np.ones(n_vtx)}) # Scalar field    -> no transformation
      fields.update({'VectorX' : -0.5*np.ones(n_vtx), 'VectorY' : 0.5*np.ones(n_vtx), 'VectorZ' : 0*np.ones(n_vtx)}) # Vectorial field -> use fields formulae
      # Somehow these values leads to (1,0,0) in (eta, zeta, xi) basis
      PT.new_FlowSolution('FlowSolution', fields=fields, parent=zone)
      PT.new_ZoneSubRegion('ZoneSubRegion', fields=fields, parent=zone)
    
    # Transform cartesian coordinates and fields into cylindric from any revolution axis
    transform.cartesian_to_cylindrical(part_tree, revolution_axis)

    if comm.size == 1:
      radius_ref = [0., 0.5, 1., 0.5, 0., 0.5, 1., 0.5, 0., 0.70710678, 0.8660254, 1.22474487, 0.8660254, 0.70710678, 0.8660254, 1.22474487, 
                    0.8660254 , 0.70710678, 1.41421356, 1.5, 1.73205081, 1.5, 1.41421356, 1.5, 1.73205081, 1.5, 1.41421356]
      theta_ref  = [0., 3.14159265, 3.14159265, 0., 0., 3.14159265, 0., 0., 0., 1.57079633, 2.18627604, 2.52611294, 0.95531662, 1.57079633, 2.18627604, 0.61547971, 
                    0.95531662, 1.57079633, 1.57079633, 1.91063324, 2.18627604, 1.23095942, 1.57079633, 1.91063324, 0.95531662, 1.23095942, 1.57079633]
      z_ref      = [0., 0.35355339, 0.70710678, 0.35355339, 0.70710678, 1.06066017, 0.70710678, 1.06066017, 1.41421356, 0., 0.35355339,
                    0.70710678, 0.35355339, 0.70710678, 1.06066017, 0.70710678, 1.06066017, 1.41421356, 0., 0.35355339,
                    0.70710678, 0.35355339, 0.70710678, 1.06066017, 0.70710678, 1.06066017, 1.41421356]
    elif comm.size == 2:
      if comm.rank == 0 :
        radius_ref = [0., 0.5, 1., 0.5, 0., 0.5, 1., 0.5, 0., 0.70710678, 0.8660254, 1.22474487, 
                      0.8660254, 0.70710678, 0.8660254, 1.22474487, 0.8660254 , 0.70710678]
        theta_ref  = [0., 3.14159265, 3.14159265, 0., 0., 3.14159265, 0., 0., 0., 1.57079633, 2.18627604, 
                      2.52611294, 0.95531662, 1.57079633, 2.18627604, 0.61547971, 0.95531662, 1.57079633]
        z_ref      = [0., 0.35355339, 0.70710678, 0.35355339, 0.70710678, 1.06066017, 0.70710678, 1.06066017, 1.41421356, 0.,
                      0.35355339, 0.70710678, 0.35355339, 0.70710678, 1.06066017, 0.70710678, 1.06066017, 1.41421356]
      elif comm.rank == 1:
        radius_ref = [0.70710678, 0.8660254, 1.22474487, 0.8660254, 0.70710678, 0.8660254, 1.22474487, 0.8660254,
                      0.70710678, 1.41421356, 1.5, 1.73205081, 1.5, 1.41421356, 1.5, 1.73205081, 1.5, 1.41421356]
        theta_ref  = [1.57079633, 2.18627604, 2.52611294, 0.95531662, 1.57079633, 2.18627604, 0.61547971, 0.95531662, 1.57079633, 
                      1.57079633, 1.91063324, 2.18627604, 1.23095942, 1.57079633, 1.91063324, 0.95531662, 1.23095942, 1.57079633]
        z_ref      = [0., 0.35355339, 0.70710678, 0.35355339, 0.70710678, 1.06066017, 0.70710678, 1.06066017, 1.41421356, 0.,
                      0.35355339, 0.70710678, 0.35355339, 0.70710678, 1.06066017, 0.70710678, 1.06066017, 1.41421356]
      
    for zone in PT.get_all_Zone_t(part_tree):
      # Recover coordinates and fields in the new basis
      for container_name in ['GridCoordinates', 'FlowSolution', 'ZoneSubRegion']:
        container = PT.get_child_from_name(zone, container_name)
        val_r, val_theta, val_z = [PT.get_node_from_name(container, f'*{d}')[1] for d in ['R', 'Theta', 'Z']]

        assert np.allclose(radius_ref, val_r)
        assert np.allclose(theta_ref, val_theta)
        assert np.allclose(z_ref, val_z)

        if container_name != 'GridCoordinates':
          scalar = PT.get_child_from_name(container, 'Scalar')[1]
          assert np.allclose(scalar, np.ones(PT.Zone.n_vtx(zone)))
          vectorr = PT.get_child_from_name(container, 'VectorR')[1]
          vectort = PT.get_child_from_name(container, 'VectorTheta')[1]
          vectorz = PT.get_child_from_name(container, 'VectorZ')[1]
          assert np.allclose(vectorr, np.cos(theta_ref))
          assert np.allclose(vectort, -1*np.sin(theta_ref))
          assert np.allclose(vectorz, np.zeros(PT.Zone.n_vtx(zone)))
      
  def test_wrong_axis(self, revolution_axis, comm):
    dist_tree = maia.factory.generate_dist_block(3, 'Poly', comm)
    with pytest.raises(AssertionError):
      transform.cartesian_to_cylindrical(dist_tree, (0, 0, 0))
