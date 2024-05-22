import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT
from maia.factory.dcube_generator import dcube_generate, dcube_nodal_generate

from maia.algo.part.geometry import centers

@pytest_parallel.mark.parallel(1)
def test_compute_cell_center(comm):
  #Test U
  tree = dcube_generate(3, 1., [0,0,0], comm)
  zoneU = PT.get_all_Zone_t(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
      node = PT.get_child_from_name(elt_node, name)
      node[1] = node[1].astype(np.int32)

  cell_center = centers.compute_cell_center(zoneU)
  expected_cart = np.array([0.25, 0.25, 0.25, 
                            0.75, 0.25, 0.25, 
                            0.25, 0.75, 0.25, 
                            0.75, 0.75, 0.25, 
                            0.25, 0.25, 0.75, 
                            0.75, 0.25, 0.75, 
                            0.25, 0.75, 0.75, 
                            0.75, 0.75, 0.75])
  assert (cell_center == expected_cart).all()

  #Test S
  cx_s = PT.get_node_from_name(zoneU, 'CoordinateX')[1].reshape((3,3,3), order='F')
  cy_s = PT.get_node_from_name(zoneU, 'CoordinateY')[1].reshape((3,3,3), order='F')
  cz_s = PT.get_node_from_name(zoneU, 'CoordinateZ')[1].reshape((3,3,3), order='F')

  zoneS = PT.new_Zone(size=[[3,2,0], [3,2,0], [3,2,0]], type='Structured')
  grid_coords = PT.new_GridCoordinates(parent=zoneS)
  PT.new_DataArray('CoordinateX', cx_s, parent=grid_coords)
  PT.new_DataArray('CoordinateY', cy_s, parent=grid_coords)
  PT.new_DataArray('CoordinateZ', cz_s, parent=grid_coords)
  cell_center = centers.compute_cell_center(zoneS)
  assert (cell_center == expected_cart).all()

  #TestU cylindrical
  tree = dcube_generate(3, 1., [0,0,0], comm)
  zoneU = PT.get_all_Zone_t(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
      node = PT.get_child_from_name(elt_node, name)
      node[1] = node[1].astype(np.int32)
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  expected_cyl = np.array([0.35355339, 0.78539816, 0.25, 
                           0.79056942, 0.32175055, 0.25, 
                           0.79056942, 1.24904577, 0.25, 
                           1.06066017, 0.78539816, 0.25, 
                           0.35355339, 0.78539816, 0.75, 
                           0.79056942, 0.32175055, 0.75, 
                           0.79056942, 1.24904577, 0.75, 
                           1.06066017, 0.78539816, 0.75])
  assert np.allclose(centers.compute_cell_center(zoneU), expected_cyl)

  #TestS cylindrical
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  tree = maia.factory.partition_dist_tree(tree, comm)
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  zoneS = PT.get_all_Zone_t(tree)[0]
  expected = np.array([0.35355339, 0.78539816, 0.25,
                       0.79056942, 0.32175055, 0.25,
                       0.79056942, 1.24904577, 0.25,
                       1.06066017, 0.78539816, 0.25,
                       0.35355339, 0.78539816, 0.75,
                       0.79056942, 0.32175055, 0.75, 
                       0.79056942, 1.24904577, 0.75,
                       1.06066017, 0.78539816, 0.75])
  assert np.allclose(centers.compute_cell_center(zoneS), expected)

  #Test Elts
  tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  zoneU = PT.get_all_Zone_t(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity']:
      node = PT.get_child_from_name(elt_node, name)
      node[1] = node[1].astype(np.int32)
  cell_center = centers.compute_cell_center(zoneU)
  assert (cell_center == expected_cart).all()

  #Test Elts // Cyl
  tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  zoneU = PT.get_all_Zone_t(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity']:
      node = PT.get_child_from_name(elt_node, name)
      node[1] = node[1].astype(np.int32)
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  cell_center = centers.compute_cell_center(zoneU)
  assert np.allclose(cell_center, expected_cyl)

@pytest_parallel.mark.parallel(1)
def test_compute_face_center_3d(comm):
  tree = dcube_generate(3, 1., [0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]

  expected = np.array([
      0.25,0.25,0.  ,   0.75,0.25,0.  ,   0.25,0.75,0.  ,   0.75,0.75,0.  ,
      0.25,0.25,0.5 ,   0.75,0.25,0.5 ,   0.25,0.75,0.5 ,   0.75,0.75,0.5 ,
      0.25,0.25,1.  ,   0.75,0.25,1.  ,   0.25,0.75,1.  ,   0.75,0.75,1.  ,
      0.  ,0.25,0.25,   0.  ,0.75,0.25,   0.  ,0.25,0.75,   0.  ,0.75,0.75,
      0.5 ,0.25,0.25,   0.5 ,0.75,0.25,   0.5 ,0.25,0.75,   0.5 ,0.75,0.75,
      1.  ,0.25,0.25,   1.  ,0.75,0.25,   1.  ,0.25,0.75,   1.  ,.75 ,.75 ,
      0.25,0.  ,0.25,   0.25,0.  ,0.75,   0.75,0.  ,0.25,   0.75,0.  ,0.75,
      0.25,0.5 ,0.25,   0.25,0.5 ,0.75,   0.75,0.5 ,0.25,   0.75,0.5 ,0.75,
      0.25,1.  ,0.25,   0.25,1.  ,0.75,   0.75,1.  ,0.25,   0.75,1.  ,0.75,])

  assert np.array_equal(centers.compute_face_center(zone), expected)

  tree = maia.factory.generate_dist_block(3, 'Structured', comm)
  # Reput coords as partitioned tree
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  expected = np.array([
     0.  ,0.25,0.25, 0.5 ,0.25,0.25, 1.  ,0.25,0.25,
     0.  ,0.75,0.25, 0.5 ,0.75,0.25, 1.  ,0.75,0.25, 
     0.  ,0.25,0.75, 0.5 ,0.25,0.75, 1.  ,0.25,0.75,
     0.  ,0.75,0.75, 0.5 ,0.75,0.75, 1.  ,0.75,0.75, # End of IFaces
     0.25,0.  ,0.25, 0.75,0.  ,0.25, 0.25,0.5 ,0.25,
     0.75,0.5 ,0.25, 0.25,1.  ,0.25, 0.75,1.  ,0.25, 
     0.25,0.  ,0.75, 0.75,0.  ,0.75, 0.25,0.5 ,0.75,
     0.75,0.5 ,0.75, 0.25,1.  ,0.75, 0.75,1.  ,0.75, # End of JFaces
     0.25,0.25,0.  , 0.75,0.25,0.  , 0.25,0.75,0.  ,
     0.75,0.75,0.  , 0.25,0.25,0.5 , 0.75,0.25,0.5 ,
     0.25,0.75,0.5 , 0.75,0.75,0.5 , 0.25,0.25,1.  ,
     0.75,0.25,1.  , 0.25,0.75,1.  , 0.75,0.75,1.  ]) # End of KFaces
  assert np.array_equal(centers.compute_face_center(zone), expected)
  
  # Test structured in cylindrical coordinates
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  expected = np.array([0.25      , 0.78539816, 0.25, 0.60355339, 0.39269908, 0.25, 1.05901699, 0.2318238 , 0.25, 
                       0.75      , 1.57079633, 0.25, 0.91257038, 0.94627344, 0.25, 1.26612378, 0.62452289, 0.25, 
                       0.25      , 0.78539816, 0.75, 0.60355339, 0.39269908, 0.75, 1.05901699, 0.2318238 , 0.75, 
                       0.75      , 1.57079633, 0.75, 0.91257038, 0.94627344, 0.75, 1.26612378, 0.62452289, 0.75, # End of IFace
                       0.25      , 0.        , 0.25, 0.75      , 0.        , 0.25, 0.60355339, 1.17809725, 0.25, 
                       0.91257038, 0.62452289, 0.25, 1.05901699, 1.33897252, 0.25, 1.26612378, 0.94627344, 0.25, 
                       0.25      , 0.        , 0.75, 0.75      , 0.        , 0.75, 0.60355339, 1.17809725, 0.75,
                       0.91257038, 0.62452289, 0.75, 1.05901699, 1.33897252, 0.75, 1.26612378, 0.94627344, 0.75, # End of JFace
                       0.4267767 , 0.58904862, 0.  , 0.83128519, 0.31226144, 0.  , 0.83128519, 1.25853488, 0.  , 
                       1.08934708, 0.78539816, 0.  , 0.4267767 , 0.58904862, 0.5 , 0.83128519, 0.31226144, 0.5 ,
                       0.83128519, 1.25853488, 0.5 , 1.08934708, 0.78539816, 0.5 , 0.4267767 , 0.58904862, 1.  , 
                       0.83128519, 0.31226144, 1.  , 0.83128519, 1.25853488, 1.  , 1.08934708, 0.78539816, 1.  ]) # End of KFaceCenter
  assert np.allclose(centers.compute_face_center(zone), expected)

  # Test unstructured in cylindrical coordinates
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  tree = maia.factory.partition_dist_tree(tree, comm)
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  zone = PT.get_all_Zone_t(tree)[0]
  expected = [0.4267767 , 0.58904862, 0.  , 0.83128519, 0.31226144, 0.  , 0.83128519, 1.25853488, 0.  ,
              1.08934708, 0.78539816, 0.  , 0.4267767 , 0.58904862, 0.5 , 0.83128519, 0.31226144, 0.5 , 
              0.83128519, 1.25853488, 0.5 , 1.08934708, 0.78539816, 0.5 , 0.4267767 , 0.58904862, 1.  , 
              0.83128519, 0.31226144, 1.  , 0.83128519, 1.25853488, 1.  , 1.08934708, 0.78539816, 1.  , 
              0.25      , 0.78539816, 0.25, 0.75      , 1.57079633, 0.25, 0.25      , 0.78539816, 0.75,
              0.75      , 1.57079633, 0.75, 0.60355339, 0.39269908, 0.25, 0.91257038, 0.94627344, 0.25, 
              0.60355339, 0.39269908, 0.75, 0.91257038, 0.94627344, 0.75, 1.05901699, 0.2318238 , 0.25, 
              1.26612378, 0.62452289, 0.25, 1.05901699, 0.2318238 , 0.75, 1.26612378, 0.62452289, 0.75, 
              0.25      , 0.        , 0.25, 0.25      , 0.        , 0.75, 0.75      , 0.        , 0.25,
              0.75      , 0.        , 0.75, 0.60355339, 1.17809725, 0.25, 0.60355339, 1.17809725, 0.75,
              0.91257038, 0.62452289, 0.25, 0.91257038, 0.62452289, 0.75, 1.05901699, 1.33897252, 0.25,
              1.05901699, 1.33897252, 0.75, 1.26612378, 0.94627344, 0.25, 1.26612378, 0.94627344, 0.75]  
  assert np.allclose(centers.compute_face_center(zone), expected)

@pytest.mark.skipif(not maia.pdma_enabled, reason="Require ParaDiGMA")
@pytest_parallel.mark.parallel(1)
def test_compute_face_center_2d(comm):
  dslice_tree = maia.factory.generate_dist_block(4, "QUAD_4", comm)
  pslice_tree = maia.factory.partition_dist_tree(dslice_tree, comm)
  zone = PT.get_all_Zone_t(pslice_tree)[0]

  expected = np.array([0.16, 0.16, 0.,   0.83, 0.50, 0.,   0.50, 0.83, 0.,
                       0.16, 0.83, 0.,   0.50, 0.50, 0.,   0.50, 0.16, 0.,
                       0.16, 0.50, 0.,   0.83, 0.16, 0.,   0.83, 0.83, 0., ])

  assert np.allclose(centers.compute_face_center(zone), expected, atol=1e-2)

def test_compute_face_center_2d_S(comm):
  # With CZ   

  tree = maia.factory.generate_dist_block([3,3,1], 'Structured', comm)
  # As partitioned
  for dir in ['X', 'Y']:
    node = PT.get_node_from_name(tree, f'Coordinate{dir}')
    node[1] = node[1].reshape((3,3), order='F')
  # Change Z for test
  node = PT.get_node_from_name(tree, 'CoordinateZ')
  node[1] = np.array([[0,0.1,0], [0,0,0], [0,0.2,0.2]], order='F')
  zone = PT.get_all_Zone_t(tree)[0]
  expected = np.array([0.25,0.25,0.025  , 0.75 ,0.25,0.05,  0.25, 0.75, 0.025,   0.75, 0.75, 0.1])
  assert np.array_equal(centers.compute_face_center(zone), expected)

  # Without CZ   
  tree = maia.factory.generate_dist_block([3,3], 'Structured', comm, origin=[0., 0.])
  # As partitioned
  for dir in ['X', 'Y']:
    node = PT.get_node_from_name(tree, f'Coordinate{dir}')
    node[1] = node[1].reshape((3,3), order='F')
  zone = PT.get_all_Zone_t(tree)[0]
  expected = np.array([0.25,0.25,  0.75 ,0.25,  0.25, 0.75,  0.75, 0.75])
  assert np.array_equal(centers.compute_face_center(zone), expected)

@pytest.mark.skipif(not maia.pdm_has_ptscotch, reason="Require PTScotch")
@pytest_parallel.mark.parallel(1)
def test_compute_face_center_elmts_3d(comm):
  tree = dcube_nodal_generate(2, 1., [0,0,0], 'HEXA_8', comm)
  
  part_tree = maia.factory.partition_dist_tree(tree, comm, graph_part_tool='ptscotch')
  # Nb : metis is not robust if n_cell == 1
  zone = PT.get_all_Zone_t(part_tree)[0]

  expected = np.array([0.5,0.5,0., 0.5,0.5,1., 0.,0.5,0.5,
                       1.,0.5,0.5, 0.5,0.,0.5, 0.5,1.,0.5])
  assert np.allclose(centers.compute_face_center(zone), expected, atol=1e-2)

@pytest.mark.parametrize("elt_kind", ["QUAD_4" ,'NFACE_n'])
@pytest_parallel.mark.parallel(1)
def test_compute_edge_center_2d(elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  PT.rm_nodes_from_name(zone, ":CGNS#Distribution") # Fake part_zone (from test_connectivity_utils)

  if elt_kind=="QUAD_4":
    expected = np.array([0.25,0.,0., 0.75,0.,0., 0.25,1.,0., 0.75,1.,0.,
                         0.,0.25,0., 0.,0.75,0., 1.,0.25,0., 1.,0.75,0.])
    assert np.allclose(centers.compute_edge_center(zone), expected, atol=1e-2)

  elif elt_kind=="NFACE_n":
    with pytest.raises(NotImplementedError):
      centers.compute_edge_center(zone)
