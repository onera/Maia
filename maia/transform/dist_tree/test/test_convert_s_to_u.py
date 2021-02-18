import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import mpi4py.MPI as MPI
import numpy as np
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
class Test_compute_all_ngon_connectivity():
# --------------------------------------------------------------------------- #
  n_vtx        = np.array([3, 3, 3])
  def test_compute_ngon_monoslab_kmax(self):
    vtx_slabs = [[[0, 3], [0, 3], [2, 3]]]
    face_gnum, face_ngon, face_pe = \
        convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs, self.n_vtx)
    assert (face_gnum[:] == [33,35,34,36]                                    ).all()
    assert (face_ngon[:] == [19,20,23,22,22,23,26,25,20,21,24,23,23,24,27,26]).all()
    assert (face_pe[:,0] == [ 5, 7, 6, 8]                                    ).all()
    assert (face_pe[:,1] == [ 0, 0, 0, 0]                                    ).all()
# --------------------------------------------------------------------------- #
  def test_compute_ngon_monoslab_kmin(self):
    vtx_slabs = [[[0, 3], [0, 3], [0, 1]]]
    face_gnum, face_ngon, face_pe = \
        convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs, self.n_vtx)
    assert (face_gnum[:] == [1, 4, 2, 5, 3, 6, 13, 15, 17, 14, 16, 18, 25, 27, 26, 28]).all()
    assert (face_ngon[:] == [1, 10, 13,  4, 4, 13, 16,  7, 2,  5, 14, 11, 5,  8, 17, 14,
                             3,  6, 15, 12, 6,  9, 18, 15, 1,  2, 11, 10, 4, 13, 14,  5,
                             7, 16, 17,  8, 2,  3, 12, 11, 5, 14, 15,  6, 8, 17, 18,  9,
                             1,  4,  5,  2, 4,  7,  8,  5, 2,  5,  6,  3, 5,  8,  9,  6]).all()
    assert (face_pe[:,0] == [1, 3, 1, 3, 2, 4, 1, 1, 3, 2, 2, 4, 1, 3, 2, 4]).all()
    assert (face_pe[:,1] == [0, 0, 2, 4, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0]).all()
# --------------------------------------------------------------------------- #
  def test_compute_ngon_monoslab_random(self):
    vtx_slabs = [[[1, 2], [0, 1], [1, 2]]]
    face_gnum, face_ngon, face_pe = \
        convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs, self.n_vtx)
    assert (face_gnum[:] == [8, 20, 30]                          ).all()
    assert (face_ngon[:] == [11,14,23,20,11,12,21,20,11,12,15,14]).all()
    assert (face_pe[:,0] == [ 5, 6, 2]                           ).all()
    assert (face_pe[:,1] == [ 6, 0, 6]                           ).all()
# --------------------------------------------------------------------------- #
  def test_compute_n_face_multislabs1(self):
    vtx_slabs = [[[1, 3], [1, 2], [2, 3]], [[0, 3], [2, 3], [2, 3]]]
    face_gnum, face_ngon, face_pe = \
        convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs, self.n_vtx)
    assert (face_gnum[:] == [36]         ).all()
    assert (face_ngon[:] == [23,24,27,26]).all()
    assert (face_pe[:,0] == [ 8]         ).all()
    assert (face_pe[:,1] == [ 0]         ).all()
# --------------------------------------------------------------------------- #
  def test_compute_n_face_multislabs2(self):
    vtx_slabs = [[[0, 3], [1, 2], [1, 2]], [[0, 2], [2, 3], [1, 2]]]
    face_gnum, face_ngon, face_pe = \
        convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs, self.n_vtx)
    assert (face_gnum[:] == [10, 11, 12, 21, 22, 31, 32, 23, 24]            ).all()
    assert (face_ngon[:] == [13, 22, 25, 16, 14, 17, 26, 23, 15, 18, 27, 24,
                             13, 22, 23, 14, 14, 23, 24, 15, 13, 14, 17, 16,
                             14, 15, 18, 17, 16, 25, 26, 17, 17, 26, 27, 18]).all()
    assert (face_pe[:,0] == [7, 7, 8, 5, 6, 3, 4, 7, 8]                     ).all()
    assert (face_pe[:,1] == [0, 8, 0, 7, 8, 7, 8, 0, 0]                     ).all()
# --------------------------------------------------------------------------- #
  def test_compute_n_face_multislabs3(self):
    vtx_slabs = [[[2, 3], [2, 3], [1, 2]], [[0, 3], [0, 1], [2, 3]], [[0, 1], [1, 2], [2, 3]]]
    face_gnum, face_ngon, face_pe = \
        convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs, self.n_vtx)
    assert (face_gnum[:] == [33,34,35]                           ).all()
    assert (face_ngon[:] == [19,20,23,22,20,21,24,23,22,23,26,25]).all()
    assert (face_pe[:,0] == [ 5, 6, 7]                           ).all()
    assert (face_pe[:,1] == [ 0, 0, 0]                           ).all()
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
  vtx_range_last  = np.array([[1,17], [9,9], [1,7]])
  vtx_range_first = np.array([[1,17], [1,1], [1,7]])
  cell_range_last  = np.array([[1,16], [8,8], [1,6]])
  cell_range_first = np.array([[1,16], [1,1], [1,6]])
  #Same location
  assert convert_s_to_u.normal_index_shift(vtx_range_last, nVtx, 1, False, False) == 0
  assert convert_s_to_u.normal_index_shift(cell_range_last, nVtx, 1, True, True) == 0
  #Vtx to cells
  assert convert_s_to_u.normal_index_shift(vtx_range_last, nVtx, 1, False, True) == -1
  assert convert_s_to_u.normal_index_shift(vtx_range_first, nVtx, 1, False, True) == 0
  #Cell to vtx
  assert convert_s_to_u.normal_index_shift(cell_range_last, nVtx, 1, True, False) == 1
  assert convert_s_to_u.normal_index_shift(cell_range_first, nVtx, 1, True, False) == 0

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
  assert I.getValue(I.getNodeFromType1(bc_u, 'GridLocation_t')) == 'FaceCenter'
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
  assert I.getValue(I.getNodeFromType1(gcA_u, 'GridLocation_t')) == 'FaceCenter'
  assert I.getValue(I.getNodeFromType1(gcB_u, 'GridLocation_t')) == 'FaceCenter'
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
  assert I.getNodeFromName1(ngon, 'ElementConnectivity#Size')[1] == 4*29
  assert I.getNodeFromName1(ngon, 'ParentElements')[1].shape == (expected_n_faces, 2)
  assert I.getNodeFromName1(ngon, 'ElementConnectivity')[1].shape == (4*expected_n_faces,)
###############################################################################
