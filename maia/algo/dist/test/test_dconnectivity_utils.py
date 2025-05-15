import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import vstride as vs

from maia.algo.dist import connectivity_utils as CU

@pytest_parallel.mark.parallel(2)
def test_combine_face_edge_and_edge_vtx(comm):
  edge_distri_full = np.array([0, 12, 24])
  if comm.rank == 0:
    face_edge_idx = np.array([0,4,8,12,16,20])
    face_edge = np.array([1,3,5,7,  -5,2,6,9,   -6,4,8,11,  -7,10,12,14, -12,-9,13,16])
    edge_vtx = np.array([1,2,2,3,4,1,3,4,2,6,3,7,6,5,4,8,7,6,9,5,8,7,6,10])
    expected_face_vtx = [1,2,6,5,  6,2,3,7,  7,3,4,8,  5,6,10,9,  10,6,7,11]
  else:
    face_edge_idx = np.array([20,24,28,32,36])
    face_edge = np.array([-13,-11,15,18,  -14,17,19,21,  -19,-16,20,23,  -20,-18,22,24])

    edge_vtx = np.array([7,11,10,9,8,12,11,10,13,9,12,11,10,14,11,15,14,13,12,16,15,14,16,15])
    expected_face_vtx = [11,7,8,12,  9,10,14,13,  14,10,11,15,  15,11,12,16]

  face_vtx = CU.combine_face_edge_and_edge_vtx(face_edge_idx, face_edge, edge_distri_full, edge_vtx, comm)

  assert np.array_equal(face_vtx, expected_face_vtx)

@pytest_parallel.mark.parallel(2)
def test_combine_dconnectivity(comm):
  if comm.rank == 0:
    cell_face = vs.array([[1, 2, 7, 8, 13, 14], [-14, 3, 4, 9, 10, 15]])
    face_vtx = vs.array([[1,5,7,3], [2,4,8,6], [5,9,11,7], [6,8,12,10], 
                         [9,13,15,11], [10,12,16,14], [1,2,6,5], [3,7,8,4]])

    expected_cell_vtx = vs.array([[1,5,7,3,2,4,8,6], [5,6,8,7,9,11,12,10]])
  elif comm.rank == 1:
    cell_face = vs.array([[-15, 5, 6, 11, 12, 16]])
    face_vtx = vs.array([[5,6,10,9], [7,11,12,8], [9,10,14,13], [11,15,16,12],
                         [1,3,4,2], [5,6,8,7], [9,10,12,11], [13,14,16,15]])

    expected_cell_vtx = vs.array([[9,10,12,11,13,15,16,14]])

  cell_distri = np.array([0,2,3])
  face_distri = np.array([0,8,16])

  cell_vtx = CU.combine_dconnectivity(cell_distri, face_distri, cell_face, face_vtx, False, comm)
  assert vs.array_equal(cell_vtx, expected_cell_vtx)

def test_cell_vtx_connectivity_S():
  zone = PT.new_Zone(type='Structured', size=[[51,50,0]])
  MT.new_distribution({'Cell': [12,16,50]}, zone)
  assert vs.array_equal(CU.cell_vtx_connectivity_S(zone, 1),
                        vs.array([[13,14], [14,15], [15,16], [16,17]]))

  zone = PT.new_Zone(type='Structured', size=[[5,4,0],[3,2,0]])
  MT.new_distribution({'Cell': [2,6,8]}, zone)
  assert vs.array_equal(CU.cell_vtx_connectivity_S(zone, 2),
                        vs.array([[3,4,9,8],  [4,5,10,9],  [6,7,12,11],  [7,8,13,12]]))

  MT.new_distribution({'Cell': [7,7,8]}, zone)
  cell_vtx = CU.cell_vtx_connectivity_S(zone, 2)
  assert len(cell_vtx) == 0 and cell_vtx.dtype == zone[1].dtype

  zone = PT.new_Zone(type='Structured', size=[[5,4,0],[3,2,0],[2,1,0]])
  MT.new_distribution({'Cell': [2,6,8]}, zone)
  assert vs.array_equal(CU.cell_vtx_connectivity_S(zone, 3),
                        vs.array([[3,4,9,8,18,19,24,23],  
                                  [4,5,10,9,19,20,25,24],  
                                  [6,7,12,11,21,22,27,26], 
                                  [7,8,13,12,22,23,28,27]]))

  zone = PT.new_Zone(type='Structured', size=[[5,4,0],[3,2,0],[4,3,0]])
  MT.new_distribution({'Cell': [15,17,24]}, zone)
  assert vs.array_equal(CU.cell_vtx_connectivity_S(zone, 3),
                        vs.array([[24,25,30,29,39,40,45,44], [31,32,37,36,46,47,52,51]]))


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

  cell_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 3, distri_global)
  assert vs.array_equal(cell_vtx, vs.from_displs(expected_cell_vtx_idx, expected_cell_vtx))


  if not distri_global:
    expected_cell_vtx_idx = [0,3]
    expected_cell_vtx = [[1,2,3],
                         [4,5,6],
                         [7,8,9]
                        ][comm.rank]
    cell_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 2, False)
    assert vs.array_equal(cell_vtx, vs.from_displs(expected_cell_vtx_idx, expected_cell_vtx))
