import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT

from maia.algo import geometry_utils as GU

@pytest_parallel.mark.parallel(1)
def test_compute_face_circulation(comm):
  # 2D faces
  cx = np.array([0,.45,.55,1,0,1,0,.3,.7,1])
  cy = np.array([0,0,0,0,0.5,0.5,1,1,1,1])
  cz = 2*np.ones_like(cx) # Dont use 0 otherwise scalar product is null
  face_vtx_idx = np.array([0,3,6,12,15,18])
  face_vtx_n = np.array([3,3,6,3,3])
  face_vtx = np.array([1,2,5, 3,4,6, 2,3,6,9,8,5, 5,8,7, 6,10,9])
  # Extend coords
  local_coords = [c[face_vtx-1] for c in [cx,cy,cz]]
  circu = GU.compute_face_circulation(local_coords, face_vtx_idx, face_vtx_n)
  areas = np.array([.1125, .1125, 0, .075, .075])
  areas[2] = 1. - areas.sum()
  assert np.allclose(2*areas, circu) # Since cz==2, and normal is Oz axis, product xF.nF is 2

  # 3D faces (coords already selected)
  cx = np.array([0.,0,1,1])
  cy = np.array([1.,0,0,1])
  cz = np.array([1.,1,0,0])
  face_vtx_idx = np.array([0,4])
  face_vtx_n = np.array([4])
  # Face center is (.5, .5, .5), unit normal (1,0,1) and face area 1 -> out is 1.
  circu = GU.compute_face_circulation([cx,cy,cz], face_vtx_idx, face_vtx_n)
  assert np.allclose(circu, [1.])

def test_update_container():
  zone = PT.new_Zone(type='Unstructured')
  fields = {'SolA' : np.ones(10)}

  cont = GU.update_container(zone, 'MyContainer', 'CellCenter', fields)
  assert PT.get_name(cont) == 'MyContainer' and PT.get_label(cont) == 'DiscreteData_t'
  assert PT.Subset.GridLocation(cont) == 'CellCenter'
  assert PT.get_child_from_name_and_label(cont, 'SolA', 'DataArray_t')[1].size == 10

  # Container should not be erased
  cont2 = GU.update_container(zone, 'MyContainer', 'CellCenter')
  assert PT.is_same_tree(cont, cont2)

  fields = {'SolA' : np.ones(15), 'SolB' : np.ones(15)}
  cont3 = GU.update_container(zone, 'MyContainer', 'CellCenter', fields)
  assert PT.get_child_from_name_and_label(cont3, 'SolA', 'DataArray_t')[1].size == 15
  assert PT.get_child_from_name_and_label(cont3, 'SolB', 'DataArray_t')[1].size == 15

  with pytest.raises(RuntimeError):
    cont4 = GU.update_container(zone, 'MyContainer', 'FaceCenter')

