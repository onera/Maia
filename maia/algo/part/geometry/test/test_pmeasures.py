import pytest
import pytest_parallel
import numpy as np
from mpi4py import MPI

import maia
import maia.pytree as PT

from maia.algo.part.geometry import measures

@pytest_parallel.mark.parallel(1)
def test_compute_face_circulation(comm):
  # 2D faces
  cx = np.array([0,.45,.55,1,0,1,0,.3,.7,1])
  cy = np.array([0,0,0,0,0.5,0.5,1,1,1,1])
  cz = 2*np.ones_like(cx) # Dont use 0 otherwise scalar product is null
  face_vtx_idx = np.array([0,3,6,12,15,18])
  face_vtx_n = np.array([3,3,6,3,3])
  face_vtx = np.array([1,2,5, 3,4,6, 2,3,6,9,8,5, 5,8,7, 6,10,9])
  circu = measures._compute_face_circulation([cx,cy,cz], face_vtx_idx, face_vtx_n, face_vtx)
  areas = np.array([.1125, .1125, 0, .075, .075])
  areas[2] = 1. - areas.sum()
  assert np.allclose(2*areas, circu) # Since cz==2, and normal is Oz axis, product xF.nF is 2

  # 3D faces
  cx = np.array([0.,1,0,1,0,1,0,1])
  cy = np.array([0.,0,1,1,0,0,1,1])
  cz = np.array([0.,0,0,0,1,1,1,1])
  face_vtx_idx = np.array([0,4])
  face_vtx_n = np.array([4])
  face_vtx = np.array([7,5,2,4])
  # Face center is (.5, .5, .5), unit normal (1,0,1) and face area 1 -> out is 1.
  circu = measures._compute_face_circulation([cx,cy,cz], face_vtx_idx, face_vtx_n, face_vtx)
  assert np.allclose(circu, [1.])

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ['QUAD_4', 'Poly'])
def test_compute_edge_lenght2d(elt_kind, comm):
  tree = maia.factory.generate_dist_block(5, 'QUAD_4', comm)
  if elt_kind == 'Poly':
    maia.algo.dist.convert_elements_to_ngon(tree, comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]
  
  edge_lenght = measures.compute_edge_measure(zone)
  assert (edge_lenght == 0.25).all()

  if elt_kind == 'QUAD_4': 
    assert comm.allreduce(edge_lenght.sum(), MPI.SUM) == 4 # External edges only
  else:
    assert comm.allreduce(edge_lenght.sum(), MPI.SUM) == 10 + 1 # Internal, External & part interface edges

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