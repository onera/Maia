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
def test_decompose_sections_to_face_vtx(comm):
  tree = maia.io.file_to_dist_tree(TU.mesh_dir/'hex_2_prism_2.yaml', comm)
  # HEXA EC  : [1, 2, 5, 4, 6, 7, 10, 9,   6, 7, 10, 9, 11, 12, 15, 14]
  # PENTA EC : [2, 3, 5, 7, 8, 10,   7, 8, 10, 12, 13, 15]
  zone = PT.get_all_Zone_t(tree)[0]
  face_vtx, cell_face_idx = GEO._decompose_sections_to_face_vtx(zone)
  assert (face_vtx.counts == [4,4,4,4,4,4, 4,4,4,4,4,4, 4,4,4,3,3, 4,4,4,3,3]).all()
  assert (face_vtx.displs == [0,4,8,12,16,20,24,28,32,36,40,44,48, 52,56,60,63,66,70,74,78,81,84]).all()
  assert (face_vtx.values[ 0:24]  == [1,4,5,2, 1,2,7,6, 2,5,10,7, 5,4,9,10, 1,6,9,4, 6,7,10,9]).all()
  assert (face_vtx.values[24:48]  == [6,9,10,7, 6,7,12,11, 7,10,15,12, 10,9,14,15, 6,11,14,9, 11,12,15,14]).all()
  assert (face_vtx.values[48:66]  == [2,3,8,7, 3,5,10,8, 5,2,7,10, 2,5,3, 7,8,10]).all()
  assert (face_vtx.values[66:84]  == [7,8,13,12, 8,10,15,13, 10,7,12,15, 7,10,8, 12,13,15]).all()
  assert (cell_face_idx == [0,6,12,17,22]).all()

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
