import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT
from maia.factory.dcube_generator import dcube_generate, dcube_nodal_generate

from maia.algo.part.geometry import centers

def to_expected_cyl(expected_cart):
  x = expected_cart[0::3]
  y = expected_cart[1::3]
  z = expected_cart[2::3]
  expected_cyl = np.empty_like(expected_cart)
  expected_cyl[0::3] = np.sqrt(x**2 + y**2) #R
  expected_cyl[1::3] = np.arctan2(y,x)      #O
  expected_cyl[2::3] = z                    #Z
  return expected_cyl

#region Cell center ------------------------------------------------------------


@pytest_parallel.mark.parallel(1)
def test_compute_cell_center_u_ngon(comm):
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

@pytest_parallel.mark.parallel(1)
def test_compute_cell_center_u_elts(comm):
  #Test Elts
  tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  zoneU = PT.get_all_Zone_t(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity']:
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

@pytest_parallel.mark.parallel(1)
def test_compute_cell_center_s(comm):
  tree = dcube_generate(3, 1., [0,0,0], comm)
  zoneU = PT.get_all_Zone_t(tree)[0]
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

  expected_cart = np.array([0.25, 0.25, 0.25, 
                            0.75, 0.25, 0.25, 
                            0.25, 0.75, 0.25, 
                            0.75, 0.75, 0.25, 
                            0.25, 0.25, 0.75, 
                            0.75, 0.25, 0.75, 
                            0.25, 0.75, 0.75, 
                            0.75, 0.75, 0.75])
  assert (cell_center == expected_cart).all()

@pytest_parallel.mark.parallel(1)
def test_compute_cell_center_u_ngon_cyl(comm):
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

@pytest_parallel.mark.parallel(1)
def test_compute_cell_center_u_elts_cyl(comm):
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

  expected_cyl = np.array([0.35355339, 0.78539816, 0.25, 
                           0.79056942, 0.32175055, 0.25, 
                           0.79056942, 1.24904577, 0.25, 
                           1.06066017, 0.78539816, 0.25, 
                           0.35355339, 0.78539816, 0.75, 
                           0.79056942, 0.32175055, 0.75, 
                           0.79056942, 1.24904577, 0.75, 
                           1.06066017, 0.78539816, 0.75])
  assert np.allclose(cell_center, expected_cyl)

@pytest_parallel.mark.parallel(1)
def test_compute_cell_center_s_cyl(comm):
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

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("cell_indices,expec",[
  ([8,1],[0.75,0.75,0.75,0.25,0.25,0.25]),
  ([2,1],[[0.75,0.25,0.25,0.25,0.25,0.25]]),
  ([],[]),
])
def test_compute_cell_center_u_ngon_filtered(comm,cell_indices,expec):
  #Test U
  tree = dcube_generate(3, 1., [0,0,0], comm)
  zoneU = PT.get_all_Zone_t(tree)[0]
  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
      node = PT.get_child_from_name(elt_node, name)
      node[1] = node[1].astype(np.int32)

  cell_center = centers.compute_cell_center(zoneU,cell_indices)
  assert np.allclose(cell_center,np.array(expec))

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("cell_indices,expec",[
  ([8,1],[0.75,0.75,0.75,0.25,0.25,0.25]),
  ([2,1],[[0.75,0.25,0.25,0.25,0.25,0.25]]),
  ([],[]),
])
def test_compute_cell_center_u_elts_filtered(comm,cell_indices,expec):
  tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  zoneU = PT.get_all_Zone_t(tree)[0]

  #On partitions, element are supposed to be I4
  for elt_node in PT.iter_children_from_label(zoneU, 'Elements_t'):
    for name in ['ElementConnectivity']:
      node = PT.get_child_from_name(elt_node, name)
      node[1] = node[1].astype(np.int32)
  cell_center = centers.compute_cell_center(zoneU,cell_indices)
  assert np.allclose(cell_center,np.array(expec))

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("cell_indices,expec",[
  ([8,1],[0.75,0.75,0.75,0.25,0.25,0.25]),
  ([2,1],[[0.75,0.25,0.25,0.25,0.25,0.25]]),
  ([],[]),
])
def test_compute_cell_center_s_filtered(comm,cell_indices,expec):
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  tree = maia.factory.partition_dist_tree(tree, comm)
  zoneS = PT.get_all_Zone_t(tree)[0]
  cell_center = centers.compute_cell_center(zoneS,cell_indices)
  assert np.allclose(cell_center,np.array(expec))
  
#endregion Cell center ---------------------------------------------------------
#region Face center ------------------------------------------------------------

@pytest_parallel.mark.parallel(1)
def test_compute_face_center_3d_u_ngon(comm):
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

@pytest_parallel.mark.parallel(1)
def test_compute_face_center_3d_u_elts(comm):
  # Test unstructured in cylindrical coordinates
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  expected_cart = centers.compute_face_center(zone)
  assert np.allclose(centers.compute_face_center(zone), expected_cart)

@pytest.mark.skipif(not maia.pdm_has_ptscotch, reason="Require PTScotch")
@pytest_parallel.mark.parallel(1)
def test_compute_face_center_3d_u_elts_2(comm):
  tree = dcube_nodal_generate(2, 1., [0,0,0], 'HEXA_8', comm)
  
  part_tree = maia.factory.partition_dist_tree(tree, comm, graph_part_tool='ptscotch')
  # Nb : metis is not robust if n_cell == 1
  zone = PT.get_all_Zone_t(part_tree)[0]

  expected = np.array([0.5,0.5,0., 0.5,0.5,1., 0.,0.5,0.5,
                       1.,0.5,0.5, 0.5,0.,0.5, 0.5,1.,0.5])
  assert np.allclose(centers.compute_face_center(zone), expected)

@pytest_parallel.mark.parallel(1)
def test_compute_face_center_3d_s(comm):
  tree = maia.factory.generate_dist_block(3, 'Structured', comm)
  # Reput coords as partitioned tree
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  expected_cart = np.array([
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
  assert np.array_equal(centers.compute_face_center(zone), expected_cart)

@pytest_parallel.mark.parallel(1)
def test_compute_face_center_3d_u_ngon_cyl(comm):
  tree = dcube_generate(3, 1., [0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]

  expected_cart = np.array([
      0.25,0.25,0.  ,   0.75,0.25,0.  ,   0.25,0.75,0.  ,   0.75,0.75,0.  ,
      0.25,0.25,0.5 ,   0.75,0.25,0.5 ,   0.25,0.75,0.5 ,   0.75,0.75,0.5 ,
      0.25,0.25,1.  ,   0.75,0.25,1.  ,   0.25,0.75,1.  ,   0.75,0.75,1.  ,
      0.  ,0.25,0.25,   0.  ,0.75,0.25,   0.  ,0.25,0.75,   0.  ,0.75,0.75,
      0.5 ,0.25,0.25,   0.5 ,0.75,0.25,   0.5 ,0.25,0.75,   0.5 ,0.75,0.75,
      1.  ,0.25,0.25,   1.  ,0.75,0.25,   1.  ,0.25,0.75,   1.  ,.75 ,.75 ,
      0.25,0.  ,0.25,   0.25,0.  ,0.75,   0.75,0.  ,0.25,   0.75,0.  ,0.75,
      0.25,0.5 ,0.25,   0.25,0.5 ,0.75,   0.75,0.5 ,0.25,   0.75,0.5 ,0.75,
      0.25,1.  ,0.25,   0.25,1.  ,0.75,   0.75,1.  ,0.25,   0.75,1.  ,0.75,])
  expected_cyl = to_expected_cyl(expected_cart)
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  assert np.allclose(centers.compute_face_center(zone), expected_cyl)

@pytest_parallel.mark.parallel(1)
def test_compute_face_center_3d_u_elts_cyl(comm):
  # Test unstructured in cylindrical coordinates
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  expected_cart = centers.compute_face_center(zone)
  expected_cyl = to_expected_cyl(expected_cart)
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  assert np.allclose(centers.compute_face_center(zone), expected_cyl)

@pytest_parallel.mark.parallel(1)
def test_compute_face_center_3d_s_cyl(comm):
  tree = maia.factory.generate_dist_block(3, 'Structured', comm)
  # Reput coords as partitioned tree
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  # Test structured in cylindrical coordinates
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  expected_cart = np.array([
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
  expected_cyl = to_expected_cyl(expected_cart)
  assert np.allclose(centers.compute_face_center(zone), expected_cyl)

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("face_ind",[
  [1],
  [1,2,3],
  [1,3,5,7,9,11],
  [11,12,14]
])
def test_compute_face_center_3d_u_ngon_filtered(comm,face_ind):
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
  out = centers.compute_face_center(zone,face_ind)
  face_ind = np.asarray(face_ind)
  for d in range(3):
    assert np.array_equal(out[d::3], expected[d::3][face_ind-1])


@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("face_ind",[
  [1,2,3],
  [1,3,5,7,9,11],
  [11,12,14],
  [],
])
def test_compute_face_center_3d_s_filtered(comm,face_ind):
  tree = maia.factory.generate_dist_block(3, 'Structured', comm)
  # Reput coords as partitioned tree
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  expected_cart = np.array([
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
  out = centers.compute_face_center(zone,face_ind)
  face_ind = np.asarray(face_ind)
  if not len(face_ind):
    assert len(out) == 0
  else:
    for d in range(3):
      assert np.allclose(out[d::3], expected_cart[d::3][face_ind-1])


@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("face_ind",[
  [1],
  [1,2,3],
  [1,3,5,7,9,11],
  [11,12,14]
])
def test_compute_face_center_3d_u_elts_cyl_filtered(comm,face_ind):
  # Test unstructured in cylindrical coordinates
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  expected_cart = centers.compute_face_center(zone)
  expected_cyl = to_expected_cyl(expected_cart)
  maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  out = centers.compute_face_center(zone,face_ind)
  face_ind = np.asarray(face_ind)
  for d in range(3):
    assert np.allclose(out[d::3], expected_cyl[d::3][face_ind-1])



@pytest.mark.skipif(not maia.pdm_has_ptscotch, reason="Require PTScotch")
@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("face_ind",[
  [1],
  [1,2,3],
  [5,4,3],
  [5,2,1]
])
def test_compute_face_center_3d_u_elts_filtered(comm,face_ind):
  tree = dcube_nodal_generate(2, 1., [0,0,0], 'HEXA_8', comm)
  
  part_tree = maia.factory.partition_dist_tree(tree, comm, graph_part_tool='ptscotch')
  # Nb : metis is not robust if n_cell == 1
  zone = PT.get_all_Zone_t(part_tree)[0]

  expected = np.array([0.5,0.5,0., 0.5,0.5,1., 0.,0.5,0.5,
                       1.,0.5,0.5, 0.5,0.,0.5, 0.5,1.,0.5])
  out = centers.compute_face_center(zone,face_ind)
  face_ind = np.asarray(face_ind)
  for d in range(3):
    assert np.array_equal(out[d::3], expected[d::3][face_ind-1])

@pytest.mark.parametrize("phydim", [3,2])
@pytest_parallel.mark.parallel(1)
def test_compute_face_center_2d(phydim, comm):
  dslice_tree = maia.factory.generate_dist_block(4, "QUAD_4", comm)
  pslice_tree = maia.factory.partition_dist_tree(dslice_tree, comm)
  zone = PT.get_all_Zone_t(pslice_tree)[0]

  if phydim == 2:
    PT.rm_nodes_from_name(zone, 'CoordinateZ')
  expected = np.array([1/6, 1/6, 0.,   0.5, 1/6, 0.,   5/6, 1/6, 0.,
                       1/6, 0.5, 0.,   0.5, 0.5, 0.,   5/6, 0.5, 0.,
                       1/6, 5/6, 0.,   0.5, 5/6, 0.,   5/6, 5/6, 0.])
  assert np.allclose(centers.compute_face_center(zone), expected)

@pytest.mark.parametrize("phydim", [3,2])
@pytest.mark.parametrize("face_ind", [
  [1],
  [1,2,3],
  [1,3,5,7],
  [6,7,5]
])
@pytest_parallel.mark.parallel(1)
def test_compute_face_center_2d_filtered(phydim, comm, face_ind):
  dslice_tree = maia.factory.generate_dist_block(4, "QUAD_4", comm)
  pslice_tree = maia.factory.partition_dist_tree(dslice_tree, comm)
  zone = PT.get_all_Zone_t(pslice_tree)[0]

  if phydim == 2:
    PT.rm_nodes_from_name(zone, 'CoordinateZ')

  expected = np.array([1/6, 1/6, 0.,   0.5, 1/6, 0.,   5/6, 1/6, 0.,
                       1/6, 0.5, 0.,   0.5, 0.5, 0.,   5/6, 0.5, 0.,
                       1/6, 5/6, 0.,   0.5, 5/6, 0.,   5/6, 5/6, 0.])
  out = centers.compute_face_center(zone,face_ind)
  face_ind = np.asarray(face_ind)
  for d in range(3):
    assert np.allclose(out[d::3], expected[d::3][face_ind-1])

def test_compute_face_center_2d_s(comm):
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
  expected = np.array([0.25,0.25,0,  0.75 ,0.25,0,  0.25, 0.75,0,  0.75, 0.75,0])
  assert np.array_equal(centers.compute_face_center(zone), expected)

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("face_ind",[
  [1],
  [1,2,3],
  [3,1,2],
  [],
])
def test_compute_face_center_2d_s_filtered(comm,face_ind):
  tree = maia.factory.generate_dist_block([3,3,1], 'Structured', comm)
  # As partitioned
  for dir in ['X', 'Y']:
    node = PT.get_node_from_name(tree, f'Coordinate{dir}')
    node[1] = node[1].reshape((3,3), order='F')
  # Change Z for test
  node = PT.get_node_from_name(tree, 'CoordinateZ')
  node[1] = np.array([[0,0.1,0], [0,0,0], [0,0.2,0.2]], order='F')
  zone = PT.get_all_Zone_t(tree)[0]

  expected = np.array([0.25, 0.25, 0.025,  0.75 ,0.25,0.05, 
                       0.25, 0.75, 0.025,  0.75, 0.75, 0.1])
  out = centers.compute_face_center(zone,face_ind)
  face_ind = np.asarray(face_ind)
  if not len(face_ind):
    assert len(out) == 0
  else:
    for d in range(3):
      assert np.allclose(out[d::3], expected[d::3][face_ind-1])

def test_compute_face_center_2d_s_cyl(comm):
  # Without CZ, in cyl coords (move by hand, fct does not manage z == 0 ...)
  tree = maia.factory.generate_dist_block([3,3], 'Structured', comm, origin=[0., 0.])
  cx = PT.get_node_from_name(tree, 'CoordinateX')
  cy = PT.get_node_from_name(tree, 'CoordinateY')
  r     = np.sqrt(cx[1]**2  + cy[1]**2)
  theta = np.arctan2(cy[1], cx[1])
  cx[1] = r.reshape((3,3), order='F')
  cy[1] = theta.reshape((3,3), order='F')
  cx[0] = 'CoordinateR'
  cy[0] = 'CoordinateTheta'
  zone = PT.get_all_Zone_t(tree)[0]
  expected = np.array([0.35355339,0.78539816,0,  0.79056942,0.32175055,0,  0.79056942,1.24904577,0,  1.06066017,0.785398160,0])
  assert np.allclose(centers.compute_face_center(zone), expected, atol=1e-6)


#endregion Face center ---------------------------------------------------------
#region Edge center ------------------------------------------------------------

@pytest.mark.parametrize("elt_kind", ["QUAD_4" ,'NFACE_n'])
@pytest.mark.parametrize("cyl", [False, True])
@pytest.mark.parametrize("phydim", [3,2])
@pytest_parallel.mark.parallel(1)
def test_compute_edge_center_2d(elt_kind, cyl, phydim, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  if cyl:
    maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  zone = PT.get_all_Zone_t(tree)[0]
  PT.rm_nodes_from_name(zone, ":CGNS#Distribution") # Fake part_zone (from test_connectivity_utils)

  if phydim == 2:
    PT.rm_nodes_from_name(zone, 'CoordinateZ')

  if elt_kind=="QUAD_4":
    if cyl:
      expected = np.array([0.25,0.,0.,         0.75,0.,0.,        1.030776,1.325818,0.,  1.25,0.927295,0.,
                           0.25, 1.570796,0.,  0.75,1.570796,0.,  1.030776,0.244979,0.,  1.25,0.643501,0.])
    else:
      expected = np.array([0.25,0.,0., 0.75,0.,0., 0.25,1.,0., 0.75,1.,0.,
                           0.,0.25,0., 0.,0.75,0., 1.,0.25,0., 1.,0.75,0.])
    assert np.allclose(centers.compute_edge_center(zone), expected, atol=1e-4)

  elif elt_kind=="NFACE_n":
    with pytest.raises(NotImplementedError):
      centers.compute_edge_center(zone)

@pytest.mark.skip("Not implemented")
def test_compute_edge_center_u_ngon_filtered(comm,edge_indices,expec):
  raise NotImplementedError

@pytest.mark.parametrize("cyl", [False,True])
@pytest.mark.parametrize("phydim", [3,2])
@pytest.mark.parametrize("edge_indices,expec", [
  ([1],[0.25,0.,0.]),
  ([3,2,1],[0.25,1.,0., 0.75,0.,0., 0.25,0.,0.]),
  ([],[])
])
@pytest_parallel.mark.parallel(1)
def test_compute_edge_center_u_elts_filtered(comm,cyl,phydim,edge_indices,expec):
  tree = maia.factory.generate_dist_block(3, "QUAD_4", comm)
  if cyl:
    maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  zone = PT.get_all_Zone_t(tree)[0]
  PT.rm_nodes_from_name(zone, ":CGNS#Distribution") # Fake part_zone (from test_connectivity_utils)

  if phydim == 2:
    PT.rm_nodes_from_name(zone, 'CoordinateZ')
  out = centers.compute_edge_center(zone,edge_indices=edge_indices)
  expec = np.array(expec)
  if cyl:
    expec = to_expected_cyl(expec)

  assert np.allclose(out, expec)

@pytest.mark.skip("Not implemented")
def test_compute_edge_center_s_filtered(comm,edge_indices,expec):
  raise NotImplementedError

#endregion Edge center ---------------------------------------------------------