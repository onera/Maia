import pytest
import pytest_parallel
import numpy as np
from mpi4py import MPI

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import test_utils as TU
from maia.utils import par_utils

from maia.algo.dist.geometry import measures as GEO

@pytest_parallel.mark.parallel(1)
def test_decompose_section_to_face_vtx(comm):
  tree = maia.io.file_to_dist_tree(TU.mesh_dir/'hex_2_prism_2.yaml', comm)
  # HEXA EC  : [1, 2, 5, 4, 6, 7, 10, 9,   6, 7, 10, 9, 11, 12, 15, 14]
  # PENTA EC : [2, 3, 5, 7, 8, 10,   7, 8, 10, 12, 13, 15]
  face_vtx = GEO._decompose_section_to_face_vtx(PT.get_node_from_name(tree, 'Hexas'))
  assert (face_vtx.counts == [4,4,4,4,4,4, 4,4,4,4,4,4]).all()
  assert (face_vtx.values[ 0:24]  == [1,4,5,2, 1,2,7,6, 2,5,10,7, 5,4,9,10, 1,6,9,4, 6,7,10,9]).all()
  assert (face_vtx.values[24:48]  == [6,9,10,7, 6,7,12,11, 7,10,15,12, 10,9,14,15, 6,11,14,9, 11,12,15,14]).all()
  face_vtx = GEO._decompose_section_to_face_vtx(PT.get_node_from_name(tree, 'Prisms'))
  assert (face_vtx.counts == [4,4,4,3,3, 4,4,4,3,3]).all()
  assert (face_vtx.values[ 0:18]  == [2,3,8,7, 3,5,10,8, 5,2,7,10, 2,5,3, 7,8,10]).all()
  assert (face_vtx.values[18:36]  == [7,8,13,12, 8,10,15,13, 10,7,12,15, 7,10,8, 12,13,15]).all()
  # With filter: 
  face_vtx = GEO._decompose_section_to_face_vtx(PT.get_node_from_name(tree, 'Prisms'), np.array([False, True]))
  assert (face_vtx.counts == [4,4,4,3,3]).all()
  assert (face_vtx.values == [7,8,13,12, 8,10,15,13, 10,7,12,15, 7,10,8, 12,13,15]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ['QUAD_4', 'Poly'])
def test_compute_edge_length2d(elt_kind, comm):
  tree = maia.factory.generate_dist_block(5, 'QUAD_4', comm)
  if elt_kind == 'Poly':
    maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  edge_length = GEO.compute_edge_measure(zone, comm)
  assert (edge_length == 0.25).all()

  if elt_kind == 'QUAD_4': 
    assert comm.allreduce(edge_length.sum(), MPI.SUM) == 4 # External edges only
  else:
    assert comm.allreduce(edge_length.sum(), MPI.SUM) == 10  # Internal & External edges

@pytest_parallel.mark.parallel(2)
def test_compute_edge_length_poly3D(comm):
  tree = maia.factory.generate_dist_block(3, 'NFACE_n', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  # Create some edges
  if comm.rank == 0:
    edge_co = np.array([1,2,2,3, 9,18], zone[1].dtype)
  elif comm.rank == 1:
    edge_co = np.array([18,27, 20,23,23,26], zone[1].dtype)
  edge = PT.new_Elements('EdgeElements', 'BAR_2', erange=[45,50], econn=edge_co, parent=zone)
  MT.new_Distribution({'Element' : par_utils.dn_to_distribution(3, comm)}, parent=edge)

  edge_length = GEO.compute_edge_measure(zone, comm)

  assert (edge_length == 0.5).all()
  assert comm.allreduce(edge_length.size, MPI.SUM) == PT.Element.Size(edge) # Only renseigned edges are computed


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("elt_kind", ['Poly', 'HEXA_8', 'S'])
def test_compute_face_area3d(elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  face_area = GEO.compute_face_measure(zone, comm)
  assert (face_area == 0.25).all()

  if elt_kind == 'HEXA_8': 
    assert comm.allreduce(face_area.sum(), MPI.SUM) == 6 # External faces only
  else:
    assert comm.allreduce(face_area.sum(), MPI.SUM) == 9 # Internal & External faces

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ['TRI_3', 'Poly'])
def test_compute_face_area2d(elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  if elt_kind == 'Poly':
    maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  face_area = GEO._compute_elements_measure(zone, 2, comm)
  assert (face_area == 0.125).all()
  assert comm.allreduce(face_area.sum(), MPI.SUM) == 1

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ["S", "PENTA_6", "NFACE_n", "Poly"])
def test_compute_cell_volume(elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  cell_vol = GEO.compute_cell_measure(zone, comm)

  assert comm.allreduce(cell_vol.sum(), MPI.SUM) == 1
  if elt_kind in ['S']:
    assert (cell_vol == 0.125).all()
  elif elt_kind in ['Poly', 'NFACE_n']:
    assert (cell_vol == 0.125).all()
  elif elt_kind == 'PENTA_6':
    assert (cell_vol == 0.0625).all()

@pytest_parallel.mark.parallel(2)
def test_compute_measure_indices(comm):
  # Elt mesh, 3D
  tree = maia.io.file_to_dist_tree(TU.mesh_dir / 'hex_2_prism_2.yaml', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  idx = [np.array([[17,15,17]]), np.empty((1,0), int)][comm.rank]
  mes = GEO._compute_elements_measure(zone, 3, comm, idx)
  expected = [[.5, 1, .5], []][comm.rank]
  assert np.array_equal(mes, expected)
  idx = [np.array([[1,13]]), np.array([[12,14,4]])][comm.rank]
  expected = [[1, .5], [1, .5, np.sqrt(2)]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 2, comm, idx)
  assert np.array_equal(mes, expected)

  # S mesh, 3D
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  idx = [np.array([[1,2], [1,2], [1,2]]), np.array([[1,2], [2,2], [2,2]])][comm.rank]
  mes = GEO._compute_elements_measure(zone, 3, comm, idx)
  assert np.array_equal(mes, [0.125, 0.125])

  idx = [np.array([[2,2], [1,1], [1,2]]), np.array([[3,2,1], [2,2,2], [2,2,2]])][comm.rank]
  expected = [[.25, .25], [.25, .25, .25]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 2, comm, idx, 'IFaceCenter')
  assert np.array_equal(mes, expected)

  # NGON mesh, 3D
  tree = maia.io.file_to_dist_tree(TU.mesh_dir / 'hex_2_prism_2.yaml', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  idx = [np.array([[20,22,19]]), np.array([[19]])][comm.rank]
  expected = [[1, .5, 1], [1]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 3, comm, idx)
  assert np.array_equal(mes, expected)
  
  idx = [np.array([[1,18]]), np.array([[11,8]])][comm.rank]
  expected = [[.5, 1], [1, np.sqrt(2)]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 2, comm, idx)
  assert np.array_equal(mes, expected)


  # Prepare a 2D meshes having different cell size : 
  # 1sr column : 0.05, 2n column : 0.2, 3e column: 0.25)
  tree2d = maia.factory.generate_dist_block([4,3], 'S', comm)
  vtx_distri = MT.Zone.vtx_distribution(PT.get_all_Zone_t(tree2d)[0])
  new_cx_val = np.array([0, 0.1, 0.5, 1,  0, 0.1, 0.5, 1,  0, 0.1, 0.5, 1])[vtx_distri[0]:vtx_distri[1]]
  cx = PT.find_node_from_name(tree2d, 'CoordinateX')
  PT.set_value(cx, new_cx_val)

  # Elt mesh, 2D
  tree = PT.deep_copy(tree2d)
  maia.algo.dist.convert_s_to_u(tree, 'Standard', comm)
  zone = PT.find_node_from_label(tree, 'Zone_t')

  idx = [np.array([[16,11]]), np.array([[12,13,15]])][comm.rank]
  expected = [[.25, .05], [.2, .25, .2]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 2, comm, idx)
  assert np.array_equal(mes, expected)
  idx = [np.array([[10,2,9]]), np.array([[1,8,5,4]])][comm.rank]
  expected = [[.5, .5, .4], [.5, .1, .1, .5]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 1, comm, idx)
  assert np.array_equal(mes, expected)

  # S mesh, 2D
  tree = PT.deep_copy(tree2d)
  zone = PT.get_all_Zone_t(tree)[0]

  idx = [np.array([[3,1],[2,1]]), np.array([[2,3,2],[1,1,2]])][comm.rank]
  expected = [[.25, .05], [.2, .25, .2]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 2, comm, idx)
  assert np.array_equal(mes, expected)

  # NGON mesh, 2D
  tree = PT.deep_copy(tree2d)
  maia.algo.dist.convert_s_to_ngon(tree, comm)
  maia.algo.edge_pe_to_ngon(tree, comm)
  zone = PT.find_node_from_label(tree, 'Zone_t')
  
  idx = [np.array([[23,18]]), np.array([[19,20,22]])][comm.rank]
  expected = [[.25, .05], [.2, .25, .2]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 2, comm, idx)
  assert np.array_equal(mes, expected)
  idx = [np.array([[17,4,16]]), np.array([[1,15,9,8]])][comm.rank]
  expected = [[.5, .5, .4], [.5, .1, .1, .5]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 1, comm, idx)
  assert np.array_equal(mes, expected)

  # Elt mesh, 1D
  tree = maia.factory.generate_dist_block(5, 'BAR_2', comm)
  zone = PT.find_node_from_label(tree, 'Zone_t')
  vtx_distri = MT.Zone.vtx_distribution(zone)
  cx = PT.find_node_from_name(zone, 'CoordinateX')
  PT.set_value(cx, np.array([0, 0.1, 0.3, 0.6, 1.][vtx_distri[0]:vtx_distri[1]]))

  idx = [np.array([[1,4]]), np.array([[2]])][comm.rank]
  expected = [[.1, .4], [.2]][comm.rank]
  mes = GEO._compute_elements_measure(zone, 1, comm, idx)
  assert np.allclose(mes, expected)
