import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
dtype = 'I4' if pdm_gnum_dtype == np.int32 else 'I8'
from maia.factory.dcube_generator import dcube_generate
from  maia.pytree.yaml   import parse_yaml_cgns

from maia.algo.part import extract_boundary as EXB

def test_pr_to_face_pl():
  n_vtx = np.array([4,5,5], np.int32)
  pl = EXB._pr_to_face_pl(n_vtx, np.array([[1,1], [1,3], [1,3]], order='F'), 'Vertex')
  assert (pl == [[1,5,17,21]]).all()
  pl = EXB._pr_to_face_pl(n_vtx, np.array([[1,4], [1,5], [5,5]], order='F'), 'Vertex')
  assert (pl == [[173,174,175,176,177,178,179,180,181,182,183,184]]).all()
  pl = EXB._pr_to_face_pl(n_vtx, np.array([[1,3], [1,4], [5,5]], order='F'), 'FaceCenter')
  assert (pl == [[173,174,175,176,177,178,179,180,181,182,183,184]]).all()
  pl = EXB._pr_to_face_pl(n_vtx, np.array([[1,3], [1,4], [4,4]], order='F'), 'CellCenter')
  assert (pl == [[173,174,175,176,177,178,179,180,181,182,183,184]]).all()

def test_extract_sub_connectivity():
  face_vtx_idx = np.array([0, 4, 9, 12])
  face_vtx = np.array([104, 105, 115, 114,  105, 104, 103, 102, 101, 35,34,114])

  sub_face_vtx_idx, sub_face_vtx, vtx_ids = EXB._extract_sub_connectivity(face_vtx_idx, face_vtx, np.array([1,2,3]))
  assert (sub_face_vtx_idx == [0,4,9,12]).all()
  assert (sub_face_vtx     == [6,7,9,8,  7,6,5,4,3,  2,1,8]).all()
  assert (vtx_ids          == [34,35,101,102,103,104,105,114,115]).all()

  sub_face_vtx_idx, sub_face_vtx, vtx_ids = EXB._extract_sub_connectivity(face_vtx_idx, face_vtx, np.array([1,3]))
  assert (sub_face_vtx_idx == [0,4,7]).all()
  assert (sub_face_vtx     == [3,4,6,5,  2,1,5]).all()
  assert (vtx_ids          == [34,35,104,105,114,115]).all()

@pytest_parallel.mark.parallel(1)
def test_extract_faces_mesh(comm):
  # Test U
  tree = dcube_generate(3, 1., [0,0,0], comm)
  PT.rm_nodes_from_name(tree, ':CGNS#Distribution')
  zoneU = PT.get_all_Zone_t(tree)[0]

  cx, cy, cz, face_vtx_idx, face_vtx, vtx_ids = EXB.extract_faces_mesh(zoneU, np.array([21,22,23,24]))

  assert len(cx) == 9
  assert len(face_vtx_idx) == 4 + 1
  assert (cx == [1,1,1,1,1,1,1,1,1]).all()
  assert (cy == [0,.5,1,0,.5,1,0,.5,1]).all()
  assert (cz == [0,0,0,.5,.5,.5,1,1,1]).all()
  assert (face_vtx == [4,5,2,1,5,6,3,2,7,8,5,4,8,9,6,5]).all()
  assert (vtx_ids == [3,6,9,12,15,18,21,24,27]).all()

  # Test S
  cx_s = PT.get_node_from_name(zoneU, 'CoordinateX')[1].reshape((3,3,3), order='F')
  cy_s = PT.get_node_from_name(zoneU, 'CoordinateY')[1].reshape((3,3,3), order='F')
  cz_s = PT.get_node_from_name(zoneU, 'CoordinateZ')[1].reshape((3,3,3), order='F')

  zoneS = PT.new_Zone(size=[[3,2,0], [3,2,0], [3,2,0]], type='Structured')
  grid_coords = PT.new_GridCoordinates(parent=zoneS)
  PT.new_DataArray('CoordinateX', cx_s, parent=grid_coords)
  PT.new_DataArray('CoordinateY', cy_s, parent=grid_coords)
  PT.new_DataArray('CoordinateZ', cz_s, parent=grid_coords)

  # PDM does not use the same ordering for faces, in our structured convention xmax boundary
  # would be faces 3,6,9,12
  cx, cy, cz, face_vtx_idx, face_vtx, vtx_ids = EXB.extract_faces_mesh(zoneS, np.array([3,6,9,12]))
  assert (cx == [1,1,1,1,1,1,1,1,1]).all()
  assert (cy == [0,.5,1,0,.5,1,0,.5,1]).all()
  assert (cz == [0,0,0,.5,.5,.5,1,1,1]).all()
  assert (face_vtx == [1,2,5,4,2,3,6,5,4,5,8,7,5,6,9,8]).all()
  assert (vtx_ids == [3,6,9,12,15,18,21,24,27]).all()


@pytest_parallel.mark.parallel(2)
def test_extract_surf_from_bc(comm):
  #We dont put GlobalNumbering for BCs, since its not needed, but we should
  part_0 = f"""
  ZoneU Zone_t [[18,4,0]]:
    ZoneType ZoneType_t "Unstructured":
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1]:
      CoordinateY DataArray_t [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]:
      CoordinateZ DataArray_t [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
    NGon Elements_t [22,0]:
      ElementRange IndexRange_t [1,20]:
      ElementConnectivity DataArray_t:
         I4 : [ 2,  5,  4,  1,  3,  6,  5,  2,  5,  8,  7,  4,  6,  9,  8,  5, 10,
               13, 14, 11, 11, 14, 15, 12, 13, 16, 17, 14, 14, 17, 18, 15,  1,  4,
               13, 10,  4,  7, 16, 13, 11, 14,  5,  2, 14, 17,  8,  5, 12, 15,  6,
                3, 15, 18,  9,  6, 10, 11,  2,  1, 11, 12,  3,  2,  4,  5, 14, 13,
                5,  6, 15, 14,  7,  8, 17, 16,  8,  9, 18, 17]
      ElementStartOffset DataArray_t I4 [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80]:
      ParentElements DataArray_t:
        I4 : [[1, 0], [2, 0], [4, 0], [3, 0], [1, 0], [2, 0], [4, 0], [3, 0], [1, 0], [4, 0],
              [1, 2], [4, 3], [2, 0], [3, 0], [1, 0], [2, 0], [1, 4], [2, 3], [4, 0], [3, 0]]
      :CGNS#GlobalNumbering UserDefinedData_t:
        Element DataArray_t {dtype} [1,2,3,4,5,6,7,8,13,14,17,18,21,22,25,27,29,31,33,35]:
    ZoneBC ZoneBC_t:
      bcA BC_t "FamilySpecified":
        PointList IndexArray_t [[15,16]]:
        GridLocation GridLocation_t "FaceCenter":
        FamilyName FamilyName_t "WALL":
      bcB BC_t "FamilySpecified":
        PointList IndexArray_t [[1,3,4,2]]:
        GridLocation GridLocation_t "FaceCenter":
        FamilyName FamilyName_t "NOTWALL":
      otherBC BC_t "FamilySpecified":
        PointList IndexArray_t [[9,10]]:
        GridLocation GridLocation_t "FaceCenter":
        FamilyName FamilyName_t "WALL":
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {dtype} [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
      Cell DataArray_t {dtype} [1,2,3,4]:
  """
  part_1 = f"""
  ZoneU Zone_t [[18,4,0]]:
    ZoneType ZoneType_t "Unstructured":
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1]:
      CoordinateY DataArray_t [0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1]:
      CoordinateZ DataArray_t [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]:
    NGon Elements_t [22,0]:
      ElementRange IndexRange_t [1,20]:
      ElementConnectivity DataArray_t:
         I4 : [10, 11, 14, 13, 11, 12, 15, 14, 13, 14, 17, 16, 14, 15, 18, 17,  1,
                4,  5,  2,  2,  5,  6,  3,  4,  7,  8,  5,  5,  8,  9,  6, 10, 13,
                4,  1, 13, 16,  7,  4,  2,  5, 14, 11,  5,  8, 17, 14,  3,  6, 15,
               12,  6,  9, 18, 15,  1,  2, 11, 10,  2,  3, 12, 11, 13, 14,  5,  4,
               14, 15,  6,  5, 16, 17,  8,  7, 17, 18,  9,  8]
      ElementStartOffset DataArray_t I4 [0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80]:
      ParentElements DataArray_t:
        I4 : [[1, 0], [2, 0], [3, 0], [4, 0], [1, 0], [2, 0], [3, 0], [4, 0], [1, 0], [3, 0],
              [1, 2], [3, 4], [2, 0], [4, 0], [1, 0], [2, 0], [1, 3], [2, 4], [3, 0], [4, 0]]
      :CGNS#GlobalNumbering UserDefinedData_t:
        Element DataArray_t {dtype} [5,6,7,8,9,10,11,12,15,16,19,20,23,24,26,28,30,32,34,36]:
    ZoneBC ZoneBC_t:
      bcA BC_t "FamilySpecified":
        PointList IndexArray_t [[15,16]]:
        GridLocation GridLocation_t "FaceCenter":
        FamilyName FamilyName_t "WALL":
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {dtype} [19,20,21,22,23,24,25,26,27,10,11,12,13,14,15,16,17,18]:
      Cell DataArray_t {dtype} [5,6,7,8]:
  """
  if comm.Get_rank() == 0:
    part_zones = [parse_yaml_cgns.to_node(part_0)]
    bc_pl = np.array([15,16,9,10])
    expt_face_lngn = [3,5,1,2]
    expt_vtx_lngn = [1,2,3,4,5,6,7,8,9,10]
  elif comm.Get_rank() == 1:
    part_zones = [parse_yaml_cgns.to_node(part_1)]
    bc_pl = np.array([15,16])
    expt_face_lngn = [4,6]
    expt_vtx_lngn = [11,12,13,6,7,8]

  bc_predicate = lambda n: PT.get_value(PT.get_child_from_name(n, 'FamilyName')) == 'WALL'

  bc_face_vtx, bc_face_vtx_idx, bc_face_lngn, bc_coords, bc_vtx_lngn = \
  EXB.extract_surf_from_bc(part_zones, bc_predicate, comm)
  

  assert len(bc_face_vtx) == len(bc_face_vtx_idx) == len(bc_face_lngn) == len(bc_coords) == len(bc_vtx_lngn) == 1

  cx, cy, cz, expt_bc_face_vtx_idx, expt_bc_face_vtx, _ = EXB.extract_faces_mesh(part_zones[0], bc_pl)
  assert (bc_face_vtx_idx[0] == expt_bc_face_vtx_idx).all()
  assert (bc_face_vtx[0] == expt_bc_face_vtx).all()
  assert (bc_coords[0] == np.array([cx,cy,cz]).reshape(-1, order='F')).all()
  assert (bc_face_lngn[0] == expt_face_lngn).all()
  assert (bc_vtx_lngn[0] == expt_vtx_lngn).all()
