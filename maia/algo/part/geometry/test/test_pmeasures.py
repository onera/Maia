import pytest
import pytest_parallel
from mpi4py import MPI

import maia
import maia.pytree as PT

from maia.algo.part.geometry import measures

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
@pytest.mark.parametrize("elt_kind", ['TRI_3', 'Poly'])
def test_compute_face_area2d(elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  if elt_kind == 'Poly':
    maia.algo.dist.convert_elements_to_ngon(tree, comm)

  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]

  face_area = measures._compute_zone_measures(zone, 2)
  assert (face_area == 0.125).all()
  assert comm.allreduce(face_area.sum(), MPI.SUM) == 1

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ["S", "PENTA_6", "NFACE_n", "Poly"])
def test_compute_cell_volume(elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]

  cell_vol = measures.compute_cell_measure(zone)

  if elt_kind in ['Poly', 'NFACE_n']:
    # !! For now, there is a bug in faces orientation of dcube (see PDM MR 77)
    # Replace this by -1 at next PDM update
    assert comm.allreduce(cell_vol.sum(), MPI.SUM) == -1
  else:
    assert comm.allreduce(cell_vol.sum(), MPI.SUM) == 1
  if elt_kind in ['S']:
    assert (cell_vol == 0.125).all()
  elif elt_kind in ['Poly', 'NFACE_n']: # Same
    assert (cell_vol == -0.125).all()
  elif elt_kind == 'PENTA_6':
    assert (cell_vol == 0.0625).all()