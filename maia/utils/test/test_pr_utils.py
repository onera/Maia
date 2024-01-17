import pytest

import numpy as np

from maia.utils import pr_utils

def test_normal_index_shift():
  nVtx = np.array([17,9,7])
  vtx_range_last   = np.array([[1,17], [9,9], [1,7]])
  vtx_range_first  = np.array([[1,17], [1,1], [1,7]])
  cell_range_last  = np.array([[1,16], [8,8], [1,6]])
  cell_range_first = np.array([[1,16], [1,1], [1,6]])
  facej_range_last  = np.array([[1,16], [9,9], [1,6]])
  facej_range_first = np.array([[1,16], [1,1], [1,6]])
  #Same location
  assert pr_utils.normal_index_shift(vtx_range_last, nVtx, 1, "Vertex", "Vertex") == 0
  assert pr_utils.normal_index_shift(cell_range_last, nVtx, 1, "CellCenter", "CellCenter") == 0
  #Vtx to cells
  assert pr_utils.normal_index_shift(vtx_range_last, nVtx, 1, "Vertex", "CellCenter") == -1
  assert pr_utils.normal_index_shift(vtx_range_first, nVtx, 1, "Vertex", "CellCenter") == 0
  #Cell to vtx
  assert pr_utils.normal_index_shift(cell_range_last, nVtx, 1, "CellCenter", "Vertex") == 1
  assert pr_utils.normal_index_shift(cell_range_first, nVtx, 1, "CellCenter", "Vertex") == 0
  #Face to cells
  assert pr_utils.normal_index_shift(facej_range_last, nVtx, 1, "FaceCenter", "CellCenter") == -1
  assert pr_utils.normal_index_shift(facej_range_first, nVtx, 1, "FaceCenter", "CellCenter") == 0
  #Cell to face
  assert pr_utils.normal_index_shift(cell_range_last, nVtx, 1, "CellCenter", "FaceCenter") == 1
  assert pr_utils.normal_index_shift(cell_range_first, nVtx, 1, "CellCenter", "FaceCenter") == 0



class Test_transform_bnd_pr_size():
  def test_non_ambiguous(self):
    vtx_range  = np.array([[1,17], [9,9], [1,7]])
    assert (pr_utils.transform_bnd_pr_size(vtx_range, 'Vertex', 'Vertex') == [17,1,7]).all()
    assert (pr_utils.transform_bnd_pr_size(vtx_range, 'Vertex', 'FaceCenter') == [16,1,6]).all()
    face_range  = np.array([[1,16], [9,9], [1,6]])
    assert (pr_utils.transform_bnd_pr_size(face_range, 'JFaceCenter', 'CellCenter') == [16,1,6]).all()
    assert (pr_utils.transform_bnd_pr_size(face_range, 'JFaceCenter', 'Vertex') == [17,1,7]).all()
    assert (pr_utils.transform_bnd_pr_size(face_range, 'FaceCenter', 'Vertex') == [17,1,7]).all()
    cell_range  = np.array([[1,16], [8,8], [1,6]])
    assert (pr_utils.transform_bnd_pr_size(cell_range, 'CellCenter', 'FaceCenter') == [16,1,6]).all()
    assert (pr_utils.transform_bnd_pr_size(cell_range, 'CellCenter', 'Vertex') == [17,1,7]).all()

  def test_ambiguous(self):
    face_range  = np.array([[1,16], [9,9], [6,6]])
    assert (pr_utils.transform_bnd_pr_size(face_range, 'JFaceCenter', 'Vertex') == [17,1,2]).all()
    assert (pr_utils.transform_bnd_pr_size(face_range, 'KFaceCenter', 'Vertex') == [17,2,1]).all()
    assert (pr_utils.transform_bnd_pr_size(face_range, 'FaceCenter', 'CellCenter') == [16,1,1]).all()
    with pytest.raises(ValueError):
      pr_utils.transform_bnd_pr_size(face_range, 'FaceCenter', 'Vertex')
    cell_range  = np.array([[1,16], [8,8], [5,5]])
    assert (pr_utils.transform_bnd_pr_size(cell_range, 'CellCenter', 'FaceCenter') == [16,1,1]).all()
    with pytest.raises(ValueError):
      pr_utils.transform_bnd_pr_size(cell_range, 'CellCenter', 'Vertex')