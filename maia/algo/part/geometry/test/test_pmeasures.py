import pytest
import pytest_parallel
import numpy as np
from mpi4py import MPI

import maia
import maia.pytree as PT

from maia.algo.part.geometry import measures


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

  face_area = measures._compute_zone_measures(zone, 2)
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

  if elt_kind in ['Poly', 'NFACE_n']:
    # !! For now, there is a bug in faces orientation of dcube (see PDM MR 77)
    # Replace this by -1 at next PDM update
    assert abs(comm.allreduce(cell_vol.sum(), MPI.SUM) - -1) < 1E-12
  else:
    assert abs(comm.allreduce(cell_vol.sum(), MPI.SUM) -  1) < 1E-12
  if elt_kind in ['S']:
    assert np.allclose(cell_vol, 0.125)
  elif elt_kind in ['Poly', 'NFACE_n']: # Same
    assert np.allclose(cell_vol, -0.125)
  elif elt_kind == 'PENTA_6':
    assert np.allclose(cell_vol, 0.0625)