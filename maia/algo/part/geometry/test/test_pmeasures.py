import pytest
import pytest_parallel
import numpy as np
from mpi4py import MPI

import maia
import maia.pytree as PT

from maia.algo.part.geometry import measures

from maia.utils import test_utils as TU


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ['QUAD_4', 'Poly'])
def test_compute_edge_length2d(elt_kind, comm):
  tree = maia.factory.generate_dist_block(5, 'QUAD_4', comm)
  if elt_kind == 'Poly':
    maia.algo.dist.convert_elements_to_ngon(tree, comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]
  
  edge_length = measures.compute_edge_measure(zone)
  assert (edge_length == 0.25).all()

  if elt_kind == 'QUAD_4': 
    assert comm.allreduce(edge_length.sum(), MPI.SUM) == 4 # External edges only
  else:
    assert comm.allreduce(edge_length.sum(), MPI.SUM) == 10 + 1 # Internal, External & part interface edges

@pytest_parallel.mark.parallel(1)
def test_compute_edge_length_poly3D(comm):
  tree = maia.factory.generate_dist_block(3, 'NFACE_n', comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]
  
  # Create some edges
  edge_co = np.array([1,2,2,3, 9,18,18,27, 20,23,23,26], np.int32)
  edge = PT.new_Elements('EdgeElements', 'BAR_2', erange=[45,50], econn=edge_co, parent=zone)
  # We should create GlobalNumbering, but it is not required by function :-)

  edge_length = measures.compute_edge_measure(zone)

  assert (edge_length == 0.5).all()
  assert edge_length.size == PT.Element.Size(edge) # Only renseigned edges are computed

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ['Poly', 'HEXA_8', 'S'])
def test_compute_face_area3d(elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]
  
  face_area = measures.compute_face_measure(zone)
  assert (face_area == 0.25).all()

  if elt_kind == 'HEXA_8': 
    assert comm.allreduce(face_area.sum(), MPI.SUM) == 6 # External faces only
  else:
    assert comm.allreduce(face_area.sum(), MPI.SUM) == 6+3+1 # Internal, External & part interface faces

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ['TRI_3', 'Poly', 'S'])
def test_compute_face_area2d(elt_kind, comm):
  if elt_kind == 'S':
    tree = maia.factory.generate_dist_block([5,5,1], 'S', comm)
  else:
    tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
    if elt_kind == 'Poly':
      maia.algo.dist.convert_elements_to_ngon(tree, comm)

  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]

  face_area = measures._compute_elements_measure(zone, 2)
  if elt_kind == 'S':
    assert (face_area == 1./16).all()
  else:
    assert (face_area == 0.125).all()
  assert comm.allreduce(face_area.sum(), MPI.SUM) == 1

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ["S", "PENTA_6", "NFACE_n", "Poly"])
def test_compute_cell_volume(elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]

  maia.algo.transform_affine(zone, rotation_angle=[np.pi/4, np.pi/6, 0])
  cell_vol = measures.compute_cell_measure(zone)

  assert abs(comm.allreduce(cell_vol.sum(), MPI.SUM) -  1) < 1E-12
  if elt_kind in ['S']:
    assert np.allclose(cell_vol, 0.125)
  elif elt_kind in ['Poly', 'NFACE_n']:
    assert np.allclose(cell_vol, 0.125)
  elif elt_kind == 'PENTA_6':
    assert np.allclose(cell_vol, 0.0625)

@pytest_parallel.mark.parallel(1)
def test_compute_measure_indices(comm):
    # Elt mesh, 3D
    tree = maia.io.file_to_dist_tree(TU.mesh_dir / 'hex_2_prism_2.yaml', comm)
    ptree = maia.factory.partition_dist_tree(tree, comm)
    zone = PT.get_all_Zone_t(ptree)[0]

    mes = measures._compute_elements_measure(zone, 3, np.array([[17,15,17]], order='F'))
    assert np.array_equal(mes, [0.5, 1, 0.5])
    mes = measures._compute_elements_measure(zone, 2, np.array([[1,12,14,4,13]], order='F'))
    assert np.allclose(mes, [1, 1, 0.5, np.sqrt(2), 0.5])

    # S mesh, 3D
    tree = maia.factory.generate_dist_block(3, 'S', comm)
    ptree = maia.factory.partition_dist_tree(tree, comm)
    zone = PT.get_all_Zone_t(ptree)[0]

    mes = measures._compute_elements_measure(zone, 3, np.array([[1,2], [1,2], [1,2]], order='F'))
    assert np.array_equal(mes, [0.125, 0.125])

    mes = measures._compute_elements_measure(zone, 2, np.array([[2,2], [1,1], [1,2]], order='F'), 'IFaceCenter')
    assert np.array_equal(mes, [0.25, 0.25])

    # NGON mesh, 3D
    tree = maia.io.file_to_dist_tree(TU.mesh_dir / 'hex_2_prism_2.yaml', comm)
    maia.algo.dist.convert_elements_to_ngon(tree, comm)
    ptree = maia.factory.partition_dist_tree(tree, comm)
    zone = PT.get_all_Zone_t(ptree)[0]
    
    mes = measures._compute_elements_measure(zone, 3, np.array([[20,22,19]], order='F'))
    assert np.array_equal(mes, [1, 0.5, 1])
    mes = measures._compute_elements_measure(zone, 2, np.array([[1,18,11,8]], order='F'))
    assert np.allclose(mes, [0.5, 1, 1, np.sqrt(2)])


    # Prepare a 2D meshes having different cell size : 
    # 1sr column : 0.05, 2n column : 0.2, 3e column: 0.25)
    tree2d = maia.factory.generate_dist_block([4,3], 'S', comm)
    cx = PT.find_node_from_name(tree2d, 'CoordinateX')
    PT.set_value(cx, np.array([0, 0.1, 0.5, 1,  0, 0.1, 0.5, 1,  0, 0.1, 0.5, 1]))

    # Elt mesh, 2D
    tree = PT.deep_copy(tree2d)
    maia.algo.dist.convert_s_to_u(tree, 'Standard', comm)
    ptree = maia.factory.partition_dist_tree(tree, comm)
    zone = PT.get_all_Zone_t(ptree)[0]

    mes = measures._compute_elements_measure(zone, 2, np.array([[16,12,11]]))
    assert np.array_equal(mes, [0.25, 0.2, 0.05])
    mes = measures._compute_elements_measure(zone, 1, np.array([[10,9,8,1,2]]))
    assert np.array_equal(mes, [0.5, 0.4, 0.1, 0.5, 0.5])
        
    # S mesh, 2D
    tree = PT.deep_copy(tree2d)
    ptree = maia.factory.partition_dist_tree(tree, comm)
    zone = PT.get_all_Zone_t(ptree)[0]

    mes = measures._compute_elements_measure(zone, 2, np.array([[3,2,1], [1,1,2]], order='F'))
    assert np.array_equal(mes, [0.25, 0.2, 0.05])

    # NGON mesh, 2D
    tree = PT.deep_copy(tree2d)
    maia.algo.dist.convert_s_to_ngon(tree, comm)
    ptree = maia.factory.partition_dist_tree(tree, comm)
    zone = PT.get_all_Zone_t(ptree)[0]
    
    mes = measures._compute_elements_measure(zone, 2, np.array([[23,19,18]]))
    assert np.array_equal(mes, [0.25, 0.2, 0.05])
    mes = measures._compute_elements_measure(zone, 1, np.array([[14,13,12,6,7,8]]))
    assert np.array_equal(mes, [0.5, 0.4, 0.1, 0.5, 0.5, 0.5])

    # Elt mesh, 1D
    tree = maia.factory.generate_dist_block(5, 'BAR_2', comm)
    cx = PT.find_node_from_name(tree, 'CoordinateX')
    PT.set_value(cx, np.array([0, 0.1, 0.3, 0.6, 1.]))
    ptree = maia.factory.partition_dist_tree(tree, comm)
    zone = PT.get_all_Zone_t(ptree)[0]

    mes = measures._compute_elements_measure(zone, 1, np.array([[1,4,3,2]]))
    assert np.allclose(mes, [.1, .4, .3, .2])