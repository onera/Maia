import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import mpi4py.MPI as MPI
import numpy as np
from maia.sids  import sids
from maia.utils import parse_yaml_cgns
from maia.transform.dist_tree import convert_s_to_u
import Converter.Internal as I


###############################################################################
def test_n_face_per_dir():
  nVtx = np.array([7,9,5])
  assert (convert_s_to_u.n_face_per_dir(nVtx, nVtx-1) == [7*8*4,6*9*4,6*8*5]).all()
def test_vtx_slab_to_n_face():
# --------------------------------------------------------------------------- #
  nVtx        = [3, 3, 3]
  #kmax
  assert (convert_s_to_u.vtx_slab_to_n_faces([[0, 3], [0, 3], [2, 3]], nVtx) == [0,0,4]).all()
  assert (convert_s_to_u.vtx_slab_to_n_faces([[1, 3], [1, 2], [2, 3]], nVtx) == [0,0,1]).all()
  assert (convert_s_to_u.vtx_slab_to_n_faces([[0, 3], [2, 3], [2, 3]], nVtx) == [0,0,0]).all()
  #random
  assert (convert_s_to_u.vtx_slab_to_n_faces([[0, 3], [0, 3], [0, 1]], nVtx) == [6,6,4]).all()
  assert (convert_s_to_u.vtx_slab_to_n_faces([[1, 2], [0, 1], [1, 2]], nVtx) == [1,1,1]).all()
  assert (convert_s_to_u.vtx_slab_to_n_faces([[0, 3], [1, 2], [1, 2]], nVtx) == [3,2,2]).all()
  assert (convert_s_to_u.vtx_slab_to_n_faces([[0, 2], [2, 3], [1, 2]], nVtx) == [0,2,0]).all()
  assert (convert_s_to_u.vtx_slab_to_n_faces([[2, 3], [2, 3], [1, 2]], nVtx) == [0,0,0]).all()
  assert (convert_s_to_u.vtx_slab_to_n_faces([[0, 3], [0, 1], [2, 3]], nVtx) == [0,0,2]).all()
  assert (convert_s_to_u.vtx_slab_to_n_faces([[0, 1], [1, 2], [2, 3]], nVtx) == [0,0,1]).all()
###############################################################################

###############################################################################
class Test_compute_pointList_from_pointRanges():
  class Test_face():
    nVtx       = [3, 3, 3]
    loc        = "FaceCenter"
    # --------------------------------------------------------------------------- #
    def test_emptyRange(self):
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges([],self.nVtx,self.loc,0)
      assert (pointList.shape == (1,0))
      assert (pointList == np.empty((1,0), dtype=np.int32)).all()
    # --------------------------------------------------------------------------- #
    #FaceRange BC (Idir) : [[3,3], [1,2], [1,2]]
    def test_simple_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,2]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[3,6,9,12]]).all()
    # --------------------------------------------------------------------------- #
    def test_reversed_range(self):
      sub_ranges = [np.array([[3,3],[2,1],[1,2]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[6,3,12,9]]).all()
      sub_ranges = [np.array([[3,3],[2,1],[2,1]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[12,9,6,3]]).all()
    # --------------------------------------------------------------------------- #
    def test_multiple_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,1]]),
                    np.array([[3,3],[1,1],[2,2]]),
                    np.array([[3,3],[2,2],[2,2]])]
      pointList = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[3,6,9,12]]).all()
    # --------------------------------------------------------------------------- #
    #FaceRange BC (Jdir) : [[2,2], [1,1], [1,2]]
    def test_partial_bc(self):
      sub_ranges = [np.array([[2,2],[1,1],[1,2]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,1)
      assert (pointList == [[14,20]]).all()
    # --------------------------------------------------------------------------- #
    def test_partial_bc_multiple_range(self):
      sub_ranges = [np.array([[2,2],[1,1],[1,1]]),
                    np.array([[2,2],[1,1],[2,2]])]
      pointList = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,1)
      assert (pointList == [[14,20]]).all()

  class Test_vertex():
    nVtx       = [3, 3, 3]
    loc        = "Vertex"
    # --------------------------------------------------------------------------- #
    #VtxRange BC (IDir)= [[3,3], [1,3], [1,3]]
    def test_simple_range(self):
      sub_ranges = [np.array([[3,3],[1,3],[1,3]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[3,6,9,12,15,18,21,24,27]]).all()
    # --------------------------------------------------------------------------- #
    def test_multi_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,1]]),
                    np.array([[3,3],[3,3],[1,1]]),
                    np.array([[3,3],[1,1],[2,2]]),
                    np.array([[3,3],[2,3],[2,2]]),
                    np.array([[3,3],[1,2],[3,3]]),
                    np.array([[3,3],[3,3],[3,3]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[3,6,9,12,15,18,21,24,27]]).all()
    # --------------------------------------------------------------------------- #
    #VtxRange BC (KDir)= [[2,3], [1,2], [1,1]]
    def test_simple_partial(self):
      sub_ranges = [np.array([[2,3],[1,2],[1,1]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,2)
      assert (pointList == [[2,3,5,6]]).all()

  class Test_cell():
    nVtx       = [3, 3, 3]
    loc        = "CellCenter"
    # --------------------------------------------------------------------------- #
    def test_emptyRange(self):
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges([],self.nVtx,self.loc,0)
      assert (pointList.shape == (1,0))
      assert (pointList == np.empty((1,0), dtype=np.int32)).all()
    # --------------------------------------------------------------------------- #
    #CellRange BC (IDir)= [[2,2], [1,2], [1,2]]
    def test_simple_range(self):
      sub_ranges = [np.array([[2,2],[1,2],[1,2]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[2,4,6,8]]).all()
    # --------------------------------------------------------------------------- #
    def test_multiple_range(self):
      sub_ranges = [np.array([[2,2],[2,2],[1,1]]), #Assume we dont have first
                    np.array([[2,2],[1,2],[2,2]])]
      pointList  = convert_s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[4,6,8]]).all()
###############################################################################

###############################################################################
class Test_cgns_transform_funcs():
  # ------------------------------------------------------------------------- #
  def test_isSameAxis(self):
    assert convert_s_to_u.is_same_axis( 1, 2) == 0
    assert convert_s_to_u.is_same_axis( 1, 1) == 1
    assert convert_s_to_u.is_same_axis(-2, 2) == 1
    assert convert_s_to_u.is_same_axis( 1,-1) == 1
  # ------------------------------------------------------------------------- #
  def test_compute_transformMatrix(self):
    transform = [1,2,3]
    expected_matrix = np.eye(3, dtype=np.int32)
    assert (convert_s_to_u.compute_transform_matrix(transform) == expected_matrix).all()

    transform = [-2,3,1]
    expected_matrix = np.zeros((3,3),dtype=np.int32,order='F')
    expected_matrix[0][2] =  1
    expected_matrix[1][0] = -1
    expected_matrix[2][1] =  1
    assert (convert_s_to_u.compute_transform_matrix(transform) == expected_matrix).all()
  # ------------------------------------------------------------------------- #
  def test_apply_transformation(self):
    t_matrix = np.array([[0,-1,0], [-1,0,0], [0,0,-1]])
    start_1 = np.array([17,3,1])
    start_2 = np.array([7,9,5])
    assert (convert_s_to_u.apply_transformation(start_1, start_1, start_2, t_matrix)\
           == start_2).all() #Start
    assert (convert_s_to_u.apply_transformation(np.array([17,6,3]), start_1, start_2, t_matrix)\
           == [4,9,3]).all() #Middle
    assert (convert_s_to_u.apply_transformation(np.array([17,9,5]), start_1, start_2, t_matrix)\
           == [1,9,1]).all() #End
###############################################################################

def test_guess_bnd_normal_index():
  #Unambiguous
  assert convert_s_to_u.guess_bnd_normal_index(np.array([[1,17], [9,9], [1,7]]),'Vertex') == 1
  assert convert_s_to_u.guess_bnd_normal_index(np.array([[1,17], [9,9], [1,7]]),'CellCenter') == 1
  assert convert_s_to_u.guess_bnd_normal_index(np.array([[1,17], [9,9], [1,7]]),'JFaceCenter') == 1
  #Ambiguous
  assert convert_s_to_u.guess_bnd_normal_index(np.array([[1,17], [9,9], [7,7]]),'JFaceCenter') == 1
  assert convert_s_to_u.guess_bnd_normal_index(np.array([[1,17], [9,9], [7,7]]),'KFaceCenter') == 2
  with pytest.raises(ValueError):
    convert_s_to_u.guess_bnd_normal_index(np.array([[1,17], [9,9], [7,7]]),'FaceCenter')
    convert_s_to_u.guess_bnd_normal_index(np.array([[1,17], [9,9], [7,7]]),'CellCenter')

def test_normal_index_shift():
  nVtx = np.array([17,9,7])
  vtx_range_last   = np.array([[1,17], [9,9], [1,7]])
  vtx_range_first  = np.array([[1,17], [1,1], [1,7]])
  cell_range_last  = np.array([[1,16], [8,8], [1,6]])
  cell_range_first = np.array([[1,16], [1,1], [1,6]])
  facej_range_last  = np.array([[1,16], [9,9], [1,6]])
  facej_range_first = np.array([[1,16], [1,1], [1,6]])
  #Same location
  assert convert_s_to_u.normal_index_shift(vtx_range_last, nVtx, 1, "Vertex", "Vertex") == 0
  assert convert_s_to_u.normal_index_shift(cell_range_last, nVtx, 1, "CellCenter", "CellCenter") == 0
  #Vtx to cells
  assert convert_s_to_u.normal_index_shift(vtx_range_last, nVtx, 1, "Vertex", "CellCenter") == -1
  assert convert_s_to_u.normal_index_shift(vtx_range_first, nVtx, 1, "Vertex", "CellCenter") == 0
  #Cell to vtx
  assert convert_s_to_u.normal_index_shift(cell_range_last, nVtx, 1, "CellCenter", "Vertex") == 1
  assert convert_s_to_u.normal_index_shift(cell_range_first, nVtx, 1, "CellCenter", "Vertex") == 0
  #Face to cells
  assert convert_s_to_u.normal_index_shift(facej_range_last, nVtx, 1, "FaceCenter", "CellCenter") == -1
  assert convert_s_to_u.normal_index_shift(facej_range_first, nVtx, 1, "FaceCenter", "CellCenter") == 0
  #Cell to face
  assert convert_s_to_u.normal_index_shift(cell_range_last, nVtx, 1, "CellCenter", "FaceCenter") == 1
  assert convert_s_to_u.normal_index_shift(cell_range_first, nVtx, 1, "CellCenter", "FaceCenter") == 0

class Test_transform_bnd_pr_size():
  def test_non_ambiguous(self):
    vtx_range  = np.array([[1,17], [9,9], [1,7]])
    assert (convert_s_to_u.transform_bnd_pr_size(vtx_range, 'Vertex', 'Vertex') == [17,1,7]).all()
    assert (convert_s_to_u.transform_bnd_pr_size(vtx_range, 'Vertex', 'FaceCenter') == [16,1,6]).all()
    face_range  = np.array([[1,16], [9,9], [1,6]])
    assert (convert_s_to_u.transform_bnd_pr_size(face_range, 'JFaceCenter', 'CellCenter') == [16,1,6]).all()
    assert (convert_s_to_u.transform_bnd_pr_size(face_range, 'JFaceCenter', 'Vertex') == [17,1,7]).all()
    assert (convert_s_to_u.transform_bnd_pr_size(face_range, 'FaceCenter', 'Vertex') == [17,1,7]).all()
    cell_range  = np.array([[1,16], [8,8], [1,6]])
    assert (convert_s_to_u.transform_bnd_pr_size(cell_range, 'CellCenter', 'FaceCenter') == [16,1,6]).all()
    assert (convert_s_to_u.transform_bnd_pr_size(cell_range, 'CellCenter', 'Vertex') == [17,1,7]).all()

  def test_ambiguous(self):
    face_range  = np.array([[1,16], [9,9], [6,6]])
    assert (convert_s_to_u.transform_bnd_pr_size(face_range, 'JFaceCenter', 'Vertex') == [17,1,2]).all()
    assert (convert_s_to_u.transform_bnd_pr_size(face_range, 'KFaceCenter', 'Vertex') == [17,2,1]).all()
    assert (convert_s_to_u.transform_bnd_pr_size(face_range, 'FaceCenter', 'CellCenter') == [16,1,1]).all()
    with pytest.raises(ValueError):
      convert_s_to_u.transform_bnd_pr_size(face_range, 'FaceCenter', 'Vertex')
    cell_range  = np.array([[1,16], [8,8], [5,5]])
    assert (convert_s_to_u.transform_bnd_pr_size(cell_range, 'CellCenter', 'FaceCenter') == [16,1,1]).all()
    with pytest.raises(ValueError):
      convert_s_to_u.transform_bnd_pr_size(cell_range, 'CellCenter', 'Vertex')

# --------------------------------------------------------------------------- #

def test_bc_s_to_bc_u():
  #We dont test value of PL here, this is carried out by Test_compute_pointList_from_pointRanges
  n_vtx = np.array([4,4,4])
  bc_s = I.newBC('MyBCName', btype='BCOutflow', pointRange=[[1,4], [1,4], [4,4]])
  bc_u = convert_s_to_u.bc_s_to_bc_u(bc_s, n_vtx, 'FaceCenter', 0, 1)
  assert I.getName(bc_u) == 'MyBCName'
  assert I.getValue(bc_u) == 'BCOutflow'
  assert sids.GridLocation(bc_u) == 'FaceCenter'
  assert I.getValue(I.getNodeFromName1(bc_u, 'PointList')).shape == (1,9)

def test_gc_s_to_gc_u():
  #https://cgns.github.io/CGNS_docs_current/sids/cnct.html
  #We dont test value of PL here, this is carried out by Test_compute_pointList_from_pointRanges
  n_vtx_A = np.array([17,9,7])
  n_vtx_B = np.array([7,9,5])
  gcA_s = I.newGridConnectivity1to1('matchA', 'Base/zoneB', pointRange=[[17,17], [3,9], [1,5]], \
      pointRangeDonor=[[7,1], [9,9], [5,1]], transform = [-2,-1,-3])
  gcB_s = I.newGridConnectivity1to1('matchB', 'zoneA', pointRange=[[7,1], [9,9], [5,1]], \
      pointRangeDonor=[[17,17], [3,9], [1,5]], transform = [-2,-1,-3])

  gcA_u = convert_s_to_u.gc_s_to_gc_u(gcA_s, 'Base/zoneA', n_vtx_A, n_vtx_B, 'FaceCenter', 0, 1)
  gcB_u = convert_s_to_u.gc_s_to_gc_u(gcB_s, 'Base/zoneB', n_vtx_B, n_vtx_A, 'FaceCenter', 0, 1)

  assert I.getName(gcA_u) == 'matchA'
  assert I.getName(gcB_u) == 'matchB'
  assert I.getValue(gcA_u) == 'Base/zoneB'
  assert I.getValue(gcB_u) == 'zoneA'
  assert sids.GridLocation(gcA_u) == 'FaceCenter'
  assert sids.GridLocation(gcB_u) == 'FaceCenter'
  assert I.getValue(I.getNodeFromType1(gcA_u, 'GridConnectivityType_t')) == 'Abutting1to1'
  assert I.getValue(I.getNodeFromType1(gcB_u, 'GridConnectivityType_t')) == 'Abutting1to1'
  assert I.getValue(I.getNodeFromName1(gcA_u, 'PointList')).shape == (1,24)
  assert (I.getNodeFromName1(gcA_u, 'PointList')[1]\
          == I.getNodeFromName1(gcB_u, 'PointListDonor')[1]).all()
  assert (I.getNodeFromName1(gcB_u, 'PointList')[1]\
          == I.getNodeFromName1(gcA_u, 'PointListDonor')[1]).all()

  gcA_s = I.newGridConnectivity1to1('matchB', 'Base/zoneA', pointRange=[[17,17], [3,9], [1,5]], \
      pointRangeDonor=[[7,1], [9,9], [5,1]], transform = [-2,-1,-3])
  gcB_s = I.newGridConnectivity1to1('matchA', 'zoneB', pointRange=[[7,1], [9,9], [5,1]], \
      pointRangeDonor=[[17,17], [3,9], [1,5]], transform = [-2,-1,-3])
  gcA_u = convert_s_to_u.gc_s_to_gc_u(gcA_s, 'Base/zoneB', n_vtx_B, n_vtx_A, 'FaceCenter', 0, 1)
  gcB_u = convert_s_to_u.gc_s_to_gc_u(gcB_s, 'Base/zoneA', n_vtx_A, n_vtx_B, 'FaceCenter', 0, 1)
  assert (I.getNodeFromName1(gcA_u, 'PointList')[1]\
          == I.getNodeFromName1(gcB_u, 'PointListDonor')[1]).all()
  assert (I.getNodeFromName1(gcB_u, 'PointList')[1]\
          == I.getNodeFromName1(gcA_u, 'PointListDonor')[1]).all()

@mark_mpi_test(2)
def test_zonedims_to_ngon(sub_comm):
  #We dont test value of faceVtx/ngon here, this is carried out by Test_compute_all_ngon_connectivity
  n_vtx_zone = np.array([3,2,4])
  ngon = convert_s_to_u.zonedims_to_ngon(n_vtx_zone, sub_comm)
  n_faces = I.getNodeFromName1(ngon, "ElementStartOffset")[1].shape[0] - 1
  if sub_comm.Get_rank() == 0:
    expected_n_faces = 15
    expected_eso     = 4*np.arange(0,15+1)
  elif sub_comm.Get_rank() == 1:
    expected_n_faces = 14
    expected_eso     = 4*np.arange(15, 15+14+1)
  assert n_faces == expected_n_faces
  assert (I.getNodeFromName1(ngon, 'ElementRange')[1] == [1, 29]).all()
  assert (I.getNodeFromName1(ngon, 'ElementStartOffset')[1] == expected_eso).all()
  assert I.getNodeFromPath(ngon, ':CGNS#Distribution/ElementConnectivity')[1][2] == 4*29
  assert I.getNodeFromName1(ngon, 'ParentElements')[1].shape == (expected_n_faces, 2)
  assert I.getNodeFromName1(ngon, 'ElementConnectivity')[1].shape == (4*expected_n_faces,)
###############################################################################
