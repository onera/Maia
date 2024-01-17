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

class Test_compute_pointList_from_pointRanges():
  class Test_face():
    nVtx       = np.array([3, 3, 3], np.int32)
    loc        = "FaceCenter"
    # --------------------------------------------------------------------------- #
    def test_emptyRange(self):
      pointList  = pr_utils.compute_pointList_from_pointRanges([],self.nVtx,"I"+self.loc)
      assert (pointList.shape == (1,0))
      assert (pointList == np.empty((1,0), dtype=np.int32)).all()
    # --------------------------------------------------------------------------- #
    #FaceRange BC (Idir) : [[3,3], [1,2], [1,2]]
    def test_simple_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,2]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,"I"+self.loc)
      assert (pointList == [[3,6,9,12]]).all()
    # --------------------------------------------------------------------------- #
    def test_reversed_range(self):
      sub_ranges = [np.array([[3,3],[2,1],[1,2]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,"I"+self.loc)
      assert (pointList == [[6,3,12,9]]).all()
      sub_ranges = [np.array([[3,3],[2,1],[2,1]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,"I"+self.loc)
      assert (pointList == [[12,9,6,3]]).all()
    # --------------------------------------------------------------------------- #
    def test_multiple_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,1]]),
                    np.array([[3,3],[1,1],[2,2]]),
                    np.array([[3,3],[2,2],[2,2]])]
      pointList = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,"I"+self.loc)
      assert (pointList == [[3,6,9,12]]).all()
    # --------------------------------------------------------------------------- #
    #FaceRange BC (Jdir) : [[2,2], [1,1], [1,2]]
    def test_partial_bc(self):
      sub_ranges = [np.array([[2,2],[1,1],[1,2]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,"J"+self.loc)
      assert (pointList == [[14,20]]).all()
    # --------------------------------------------------------------------------- #
    def test_partial_bc_multiple_range(self):
      sub_ranges = [np.array([[2,2],[1,1],[1,1]]),
                    np.array([[2,2],[1,1],[2,2]])]
      pointList = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,"J"+self.loc)
      assert (pointList == [[14,20]]).all()

  class Test_vertex():
    nVtx       = np.array([3, 3, 3], np.int64)
    loc        = "Vertex"
    # --------------------------------------------------------------------------- #
    #VtxRange BC (IDir)= [[3,3], [1,3], [1,3]]
    def test_simple_range(self):
      sub_ranges = [np.array([[3,3],[1,3],[1,3]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc)
      assert (pointList == [[3,6,9,12,15,18,21,24,27]]).all()
    # --------------------------------------------------------------------------- #
    def test_multi_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,1]]),
                    np.array([[3,3],[3,3],[1,1]]),
                    np.array([[3,3],[1,1],[2,2]]),
                    np.array([[3,3],[2,3],[2,2]]),
                    np.array([[3,3],[1,2],[3,3]]),
                    np.array([[3,3],[3,3],[3,3]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc)
      assert (pointList == [[3,6,9,12,15,18,21,24,27]]).all()
    # --------------------------------------------------------------------------- #
    #VtxRange BC (KDir)= [[2,3], [1,2], [1,1]]
    def test_simple_partial(self):
      sub_ranges = [np.array([[2,3],[1,2],[1,1]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc)
      assert (pointList == [[2,3,5,6]]).all()

  class Test_cell():
    nVtx       = np.array([3, 3, 3])
    loc        = "CellCenter"
    # --------------------------------------------------------------------------- #
    def test_emptyRange(self):
      pointList  = pr_utils.compute_pointList_from_pointRanges([],self.nVtx,self.loc)
      assert (pointList.shape == (1,0))
      assert (pointList == np.empty((1,0), dtype=np.int32)).all()
    # --------------------------------------------------------------------------- #
    #CellRange BC (IDir)= [[2,2], [1,2], [1,2]]
    def test_simple_range(self):
      sub_ranges = [np.array([[2,2],[1,2],[1,2]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc)
      assert (pointList == [[2,4,6,8]]).all()
    # --------------------------------------------------------------------------- #
    def test_multiple_range(self):
      sub_ranges = [np.array([[2,2],[2,2],[1,1]]), #Assume we dont have first
                    np.array([[2,2],[1,2],[2,2]])]
      pointList  = pr_utils.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc)
      assert (pointList == [[4,6,8]]).all()