import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia

from maia.algo.dist import geometry

def to_expected_cyl(expected_cart):
  x = expected_cart[0::3]
  y = expected_cart[1::3]
  z = expected_cart[2::3]
  expected_cyl = np.empty_like(expected_cart)
  expected_cyl[0::3] = np.sqrt(x**2 + y**2) #R
  expected_cyl[1::3] = np.arctan2(y,x)      #O
  expected_cyl[2::3] = z                    #Z
  return expected_cyl

def test_cell_vtx_connectivity_S():
  zone = PT.new_Zone(type='Structured', size=[[5,4,0],[3,2,0]])
  MT.new_distribution({'Cell': [2,6,8]}, zone)
  cell_vtx_idx, cell_vtx = geometry._cell_vtx_connectivity_S(zone, 2)
  assert (cell_vtx_idx ==[0,4,8,12,16]).all() 
  assert (cell_vtx == [3,4,9,8,  4,5,10,9,  6,7,12,11,  7,8,13,12]).all()

  MT.new_distribution({'Cell': [7,7,8]}, zone)
  cell_vtx_idx, cell_vtx = geometry._cell_vtx_connectivity_S(zone, 2)
  assert cell_vtx_idx == np.zeros(1, np.int32)
  assert cell_vtx.size == 0 and cell_vtx.dtype == zone[1].dtype

  zone = PT.new_Zone(type='Structured', size=[[5,4,0],[3,2,0],[2,1,0]])
  MT.new_distribution({'Cell': [2,6,8]}, zone)
  cell_vtx_idx, cell_vtx = geometry._cell_vtx_connectivity_S(zone, 3)
  assert (cell_vtx_idx ==[0,8,16,24,32]).all() 
  assert (cell_vtx == [3,4,9,8,18,19,24,23,  4,5,10,9,19,20,25,24,  6,7,12,11,21,22,27,26,  7,8,13,12,22,23,28,27]).all()

  zone = PT.new_Zone(type='Structured', size=[[5,4,0],[3,2,0],[4,3,0]])
  MT.new_distribution({'Cell': [15,17,24]}, zone)
  cell_vtx_idx, cell_vtx = geometry._cell_vtx_connectivity_S(zone, 3)
  assert (cell_vtx_idx ==[0,8,16]).all() 
  assert (cell_vtx == [24,25,30,29,39,40,45,44,  31,32,37,36,46,47,52,51]).all()

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("distri_global", [True, False])
def test_entity_vtx_connectivity_elt(distri_global, comm):
  ft = PT.yaml.to_cgns_tree("""
  Zone Zone_t [[12, 8, 0]]:
    ZoneType ZoneType_t "Unstructured":
    TRI Elements_t [5, 0]:
      ElementRange IndexRange_t [1, 3]:
      ElementConnectivity DataArray_t [1,2,3,  4,5,6,  7,8,9]:
    PYRA Elements_t [12, 0]:
      ElementRange IndexRange_t [8, 11]:
      ElementConnectivity DataArray_t [201,202,203,204,205,  206,207,208,209,210, 211,212,213,214,215, 216,217,218,219,220]:
    TETRA Elements_t [10, 0]:
      ElementRange IndexRange_t [4, 7]:
      ElementConnectivity DataArray_t [101,102,103,104,  105,106,107,108,  109,110,111,112,  113,114,115,116]:
  """)
  tree = maia.factory.full_to_dist_tree(ft, comm)
  zone = PT.get_node_from_label(tree, 'Zone_t')

  if distri_global:
    expected_cell_vtx_idx = [[0,4,8,12],   # Proc 0
                             [0,4,9,14],   # Proc 1
                             [0,5,10]      # Proc 2
                            ][comm.rank]
    expected_cell_vtx = [[101,102,103,104,  105,106,107,108,  109,110,111,112],
                         [113,114,115,116,  201,202,203,204,205,  206,207,208,209,210],
                         [211,212,213,214,215, 216,217,218,219,220]
                        ][comm.rank]
  else:
    expected_cell_vtx_idx = [[0,4,8,13,18],   # Proc 0
                             [0,4,9],         # Proc 1
                             [0,4,9]          # Proc 2
                            ][comm.rank]
    expected_cell_vtx = [[101,102,103,104,  105,106,107,108,  201,202,203,204,205,  206,207,208,209,210],
                         [109,110,111,112,  211,212,213,214,215],
                         [113,114,115,116,    216,217,218,219,220]
                        ][comm.rank]

  cell_vtx_idx, cell_vtx = geometry._entity_vtx_connectivity_elt(zone, comm, 3, distri_global)

  assert np.array_equal(cell_vtx_idx, expected_cell_vtx_idx)
  assert np.array_equal(cell_vtx, expected_cell_vtx)


  if not distri_global:
    expected_cell_vtx_idx = [0,3]
    expected_cell_vtx = [[1,2,3],
                         [4,5,6],
                         [7,8,9]
                        ][comm.rank]
    cell_vtx_idx, cell_vtx = geometry._entity_vtx_connectivity_elt(zone, comm, 2, False)
    assert np.array_equal(cell_vtx_idx, expected_cell_vtx_idx)
    assert np.array_equal(cell_vtx, expected_cell_vtx)


@pytest_parallel.mark.parallel(3)
def test_compute_face_normal3d(comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  face_normal = geometry.compute_face_normal(zone, comm)

  # All face area are 0.25
  if comm.Get_rank() == 0:
    expected_face_normal = 0.25 * np.array([0,0, 1, 0,0, 1, 0,0, 1, 0,0, 1,
                                            0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1,
                                            0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1])
  elif comm.Get_rank() == 1:
    expected_face_normal = 0.25 * np.array([ 1,0,0,  1,0,0,  1,0,0,  1,0,0,
                                            -1,0,0, -1,0,0, -1,0,0, -1,0,0,
                                            -1,0,0, -1,0,0, -1,0,0, -1,0,0])

  if comm.Get_rank() == 2:
    expected_face_normal = 0.25 * np.array([0, 1,0,  0, 1,0,  0, 1,0,  0, 1,0,
                                            0,-1,0,  0,-1,0,  0,-1,0,  0,-1,0,
                                            0,-1,0,  0,-1,0,  0,-1,0,  0,-1,0])

  assert (face_normal == expected_face_normal).all()

@pytest_parallel.mark.parallel(2)
def test_compute_face_normal2d(comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  face_normal = geometry.compute_face_normal(zone, comm)

  assert (face_normal == np.array([0.,0.,0.125,  0.,0.,0.125,   0.,0.,0.125, 0.,0.,0.125])).all()

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center3d(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  
  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = geometry.compute_face_center(zone, comm)

  if comm.Get_rank() == 0:
    expected_face_center = np.array([
        0.25, 0.25, 0. ,  0.75, 0.25, 0. ,  0.25, 0.75, 0. ,  0.75, 0.75, 0. , 
        0.25, 0.25, 0.5,  0.75, 0.25, 0.5,  0.25, 0.75, 0.5,  0.75, 0.75, 0.5, 
        0.25, 0.25, 1. ,  0.75, 0.25, 1. ,  0.25, 0.75, 1. ,  0.75, 0.75, 1. ,
    ])
  elif comm.Get_rank() == 1:
    expected_face_center = np.array([
        0. , 0.25, 0.25,  0. , 0.75, 0.25,  0.,  0.25, 0.75,  0. , 0.75, 0.75,
        0.5, 0.25, 0.25,  0.5, 0.75, 0.25,  0.5, 0.25, 0.75,  0.5, 0.75, 0.75,
        1. , 0.25, 0.25,  1. , 0.75, 0.25,  1.,  0.25, 0.75,  1. , 0.75, 0.75,
    ])
  if comm.Get_rank() == 2:
    expected_face_center = np.array([
        0.25, 0. , 0.25,  0.25, 0. , 0.75,  0.75, 0. , 0.25,  0.75, 0. , 0.75, 
        0.25, 0.5, 0.25,  0.25, 0.5, 0.75,  0.75, 0.5, 0.25,  0.75, 0.5, 0.75,
        0.25, 1. , 0.25,  0.25, 1. , 0.75,  0.75, 1. , 0.25,  0.75, 1. , 0.75,
    ])

  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center2d(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = geometry.compute_face_center(zone, comm)

  if cylindrical:
    if comm.Get_rank() == 0:
      assert np.allclose(face_center, np.array([0.23570226,0.78539816,0,  0.47140452,0.78539816,0,  0.68718427,0.24497866,0,  0.89752747,0.38050638,0]))
    if comm.Get_rank() == 1:
      assert np.allclose(face_center, np.array([0.68718427,1.32581766,0,  0.89752747,1.19028995,0,  0.94280904,0.78539816,0,  1.17851130,0.78539816,0]))
  else:
    if comm.Get_rank() == 0:
      assert (face_center == np.array([1.,1,0, 2,2,0, 4,1,0, 5,2,0]) / 6.).all()
    if comm.Get_rank() == 1:
      assert (face_center == np.array([1.,4,0, 2,5,0, 4,4,0, 5,5,0]) / 6.).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ["S", "NFACE_n", "Poly"])
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_cell_center(elt_kind, cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))
  
  cell_center = geometry.compute_cell_center(zone, comm)
  
  if cylindrical:
      from math import pi
      expt_cell_center = [
        np.array([0.35355339, pi/4, 0.25,  0.79056942,0.32175055,0.25,  0.79056942,1.24904577,0.25,  1.06066017,pi/4,0.25]),
        np.array([0.35355339, pi/4, 0.75,  0.79056942,0.32175055,0.75,  0.79056942,1.24904577,0.75,  1.06066017,pi/4,0.75])
      ][comm.Get_rank()]
  else:
      expt_cell_center = [
        np.array([0.25,0.25,0.25, 0.75,0.25,0.25, 0.25,0.75,0.25, 0.75,0.75,0.25]),
        np.array([0.25,0.25,0.75, 0.75,0.25,0.75, 0.25,0.75,0.75, 0.75,0.75,0.75])
      ][comm.Get_rank()]

  assert np.allclose(expt_cell_center, cell_center)
