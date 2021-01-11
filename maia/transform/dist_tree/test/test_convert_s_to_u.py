import pytest
import mpi4py.MPI as MPI
import numpy as np
from maia.utils import parse_yaml_cgns
from maia.transform.dist_tree import convert_s_to_u
import Converter.Internal as I

# class Test_compare_pointrange():
#   def test_ok(self):
#     jn1 = I.newGridConnectivity1to1(pointRange=[[17,17],[3,9],[1,5]], pointRangeDonor=[[7,1],[9,9],[5,1]])
#     jn2 = I.newGridConnectivity1to1(pointRangeDonor=[[17,17],[3,9],[1,5]], pointRange=[[7,1],[9,9],[5,1]])
#     assert(add_joins_ordinal._compare_pointrange(jn1, jn2) == True)
#   def test_ko(self):
#     jn1 = I.newGridConnectivity1to1(pointRange=[[17,17],[3,9],[1,5]], pointRangeDonor=[[7,1],[9,9],[5,1]])
#     jn2 = I.newGridConnectivity1to1(pointRangeDonor=[[17,17],[3,9],[1,5]], pointRange=[[1,7],[9,9],[1,5]])
#     assert(add_joins_ordinal._compare_pointrange(jn1, jn2) == False)
# class Test_MPI():
#   @pytest.mark.mpi(min_size=3)
#   @pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
#   def test_mpi(self,sub_comm):
#     if(sub_comm == MPI.COMM_NULL):
#       return
#     nRank = sub_comm.Get_size()
#     iRank = sub_comm.Get_rank()

###############################################################################
class Test_vtx_slab_to_n_face():
# --------------------------------------------------------------------------- #
  nVtx        = [3, 3, 3]
  def test_vtx_slab_to_n_face_monoslab_imax_jmax_kmax(self):
    slabListVtx = [[0, 3], [0, 3], [2, 3]]
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx,self.nVtx) == 4
# --------------------------------------------------------------------------- #
  def test_vtx_slab_to_n_face_monoslab_imin_jmin_kmin(self):
    slabListVtx = [[0, 3], [0, 3], [0, 1]]
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx,self.nVtx) == 16
# --------------------------------------------------------------------------- #
  def test_vtx_slab_to_n_face_monoslab_random(self):
    slabListVtx = [[1, 2], [0, 1], [1, 2]]
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx,self.nVtx) == 3
# --------------------------------------------------------------------------- #
  def test_vtx_slab_to_n_face_multislabs1(self):
    slabListVtx = [[[1, 3], [1, 2], [2, 3]], [[0, 3], [2, 3], [2, 3]]]
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx[0],self.nVtx) == 1
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx[1],self.nVtx) == 0
# --------------------------------------------------------------------------- #
  def test_vtx_slab_to_n_face_multislabs2(self):
    slabListVtx = [[[0, 3], [1, 2], [1, 2]], [[0, 2], [2, 3], [1, 2]]]
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx[0],self.nVtx) == 7
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx[1],self.nVtx) == 2
# --------------------------------------------------------------------------- #
  def test_vtx_slab_to_n_face_multislabs3(self):
    slabListVtx = [[[2, 3], [2, 3], [1, 2]], [[0, 3], [0, 1], [2, 3]], [[0, 1], [1, 2], [2, 3]]]
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx[0],self.nVtx) == 0
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx[1],self.nVtx) == 2
    assert convert_s_to_u.vtx_slab_to_n_face(slabListVtx[2],self.nVtx) == 1
###############################################################################
  
###############################################################################
class Test_compute_all_ngon_connectivity():
# --------------------------------------------------------------------------- #
  n_vtx        = np.array([3, 3, 3])
  def test_compute_ngon_monoslab_kmax(self):
    vtx_slabs = [[[0, 3], [0, 3], [2, 3]]]
    n_face = 4
    face_gnum  = -np.ones(  n_face, dtype=np.int32)
    face_ngon  = -np.ones(4*n_face, dtype=np.int32)
    face_pe    = -np.ones((n_face,2), dtype=np.int32)
    convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs, self.n_vtx,
                                                 face_gnum, face_ngon, face_pe)
    assert (face_gnum[:] == [33,35,34,36]                                    ).all()
    assert (face_ngon[:] == [19,20,23,22,22,23,26,25,20,21,24,23,23,24,27,26]).all()
    assert (face_pe[:,0] == [ 5, 7, 6, 8]                                    ).all()
    assert (face_pe[:,1] == [ 0, 0, 0, 0]                                    ).all()
# --------------------------------------------------------------------------- #
  def test_compute_ngon_monoslab_kmin(self):
    vtx_slabs = [[[0, 3], [0, 3], [0, 1]]]
    n_face = 16
    face_gnum  = -np.ones(  n_face, dtype=np.int32)
    face_ngon  = -np.ones(4*n_face, dtype=np.int32)
    face_pe    = -np.ones((n_face,2), dtype=np.int32)
    convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs, self.n_vtx,
                                                 face_gnum ,face_ngon, face_pe)
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
    n_face = 3
    face_gnum  = -np.ones(  n_face, dtype=np.int32)
    face_ngon  = -np.ones(4*n_face, dtype=np.int32)
    face_pe    = -np.ones((n_face,2), dtype=np.int32)
    convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs,self.n_vtx,
                                                 face_gnum, face_ngon, face_pe)
    assert (face_gnum[:] == [8, 20, 30]                          ).all()
    assert (face_ngon[:] == [11,14,23,20,11,12,21,20,11,12,15,14]).all()
    assert (face_pe[:,0] == [ 5, 6, 2]                           ).all()
    assert (face_pe[:,1] == [ 6, 0, 6]                           ).all()
# --------------------------------------------------------------------------- #
  def test_compute_n_face_multislabs1(self):
    vtx_slabs = [[[1, 3], [1, 2], [2, 3]], [[0, 3], [2, 3], [2, 3]]]
    n_face = 1
    face_gnum  = -np.ones(  n_face, dtype=np.int32)
    face_ngon  = -np.ones(4*n_face, dtype=np.int32)
    face_pe    = -np.ones((n_face,2), dtype=np.int32)
    convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs,self.n_vtx,
                                                 face_gnum, face_ngon, face_pe)
    assert (face_gnum[:] == [36]         ).all()
    assert (face_ngon[:] == [23,24,27,26]).all()
    assert (face_pe[:,0] == [ 8]         ).all()
    assert (face_pe[:,1] == [ 0]         ).all()
# --------------------------------------------------------------------------- #
  def test_compute_n_face_multislabs2(self):
    vtx_slabs = [[[0, 3], [1, 2], [1, 2]], [[0, 2], [2, 3], [1, 2]]]
    n_face = 9
    face_gnum  = -np.ones(  n_face, dtype=np.int32)
    face_ngon  = -np.ones(4*n_face, dtype=np.int32)
    face_pe    = -np.ones((n_face,2), dtype=np.int32)
    convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs,self.n_vtx,
                                                 face_gnum, face_ngon, face_pe)
    assert (face_gnum[:] == [10, 11, 12, 21, 22, 31, 32, 23, 24]            ).all()
    assert (face_ngon[:] == [13, 22, 25, 16, 14, 17, 26, 23, 15, 18, 27, 24,
                             13, 22, 23, 14, 14, 23, 24, 15, 13, 14, 17, 16,
                             14, 15, 18, 17, 16, 25, 26, 17, 17, 26, 27, 18]).all()
    assert (face_pe[:,0] == [7, 7, 8, 5, 6, 3, 4, 7, 8]                     ).all()
    assert (face_pe[:,1] == [0, 8, 0, 7, 8, 7, 8, 0, 0]                     ).all()
# --------------------------------------------------------------------------- #
  def test_compute_n_face_multislabs3(self):
    vtx_slabs = [[[2, 3], [2, 3], [1, 2]], [[0, 3], [0, 1], [2, 3]], [[0, 1], [1, 2], [2, 3]]]
    n_face = 3
    face_gnum  = -np.ones(  n_face, dtype=np.int32)
    face_ngon  = -np.ones(4*n_face, dtype=np.int32)
    face_pe    = -np.ones((n_face,2), dtype=np.int32)
    convert_s_to_u.compute_all_ngon_connectivity(vtx_slabs,self.n_vtx,
                                                 face_gnum, face_ngon, face_pe)
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
def test_isSameAxis():
  assert convert_s_to_u.isSameAxis( 1, 2) == 0
  assert convert_s_to_u.isSameAxis( 1, 1) == 1
  assert convert_s_to_u.isSameAxis(-2, 2) == 1
  assert convert_s_to_u.isSameAxis( 1,-1) == 1
###############################################################################
  
###############################################################################
class Test_compute_transformMatrix():
# --------------------------------------------------------------------------- #
  def test_compute_transformMatrix(self):
    transform = [1,2,3]
    attendedTransformMatrix = np.eye(3, dtype=np.int32)
    assert (convert_s_to_u.compute_transformMatrix(transform) == attendedTransformMatrix).all()
# --------------------------------------------------------------------------- #
  def test_compute_transformMatrix(self):
    transform = [-2,3,1]
    attendedTransformMatrix = np.zeros((3,3),dtype=np.int32,order='F')
    attendedTransformMatrix[0][2] =  1
    attendedTransformMatrix[1][0] = -1
    attendedTransformMatrix[2][1] =  1
    assert (convert_s_to_u.compute_transformMatrix(transform) == attendedTransformMatrix).all()
###############################################################################
def test_guess_boundary_axis():
  #Unambiguous
  assert convert_s_to_u.guess_boundary_axis(np.array([[1,17], [9,9], [1,7]]),'Vertex') == 1
  assert convert_s_to_u.guess_boundary_axis(np.array([[1,17], [9,9], [1,7]]),'CellCenter') == 1
  assert convert_s_to_u.guess_boundary_axis(np.array([[1,17], [9,9], [1,7]]),'JFaceCenter') == 1
  #Ambiguous
  assert convert_s_to_u.guess_boundary_axis(np.array([[1,17], [9,9], [7,7]]),'JFaceCenter') == 1
  assert convert_s_to_u.guess_boundary_axis(np.array([[1,17], [9,9], [7,7]]),'KFaceCenter') == 2
  with pytest.raises(ValueError):
    convert_s_to_u.guess_boundary_axis(np.array([[1,17], [9,9], [7,7]]),'FaceCenter')
    convert_s_to_u.guess_boundary_axis(np.array([[1,17], [9,9], [7,7]]),'CellCenter')

def test_cst_axe_shift():
  nVtx = np.array([17,9,7])
  vtx_range_last  = np.array([[1,17], [9,9], [1,7]])
  vtx_range_first = np.array([[1,17], [1,1], [1,7]])
  cell_range_last  = np.array([[1,16], [8,8], [1,6]])
  cell_range_first = np.array([[1,16], [1,1], [1,6]])
  #Same location
  assert convert_s_to_u.cst_axe_shift(vtx_range_last, nVtx, 1, False, False) == 0
  assert convert_s_to_u.cst_axe_shift(cell_range_last, nVtx, 1, True, True) == 0
  #Vtx to cells
  assert convert_s_to_u.cst_axe_shift(vtx_range_last, nVtx, 1, False, True) == -1
  assert convert_s_to_u.cst_axe_shift(vtx_range_first, nVtx, 1, False, True) == 0
  #Cell to vtx
  assert convert_s_to_u.cst_axe_shift(cell_range_last, nVtx, 1, True, False) == 1
  assert convert_s_to_u.cst_axe_shift(cell_range_first, nVtx, 1, True, False) == 0

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
  # @pytest.mark.mpi(min_size=3)
  # @pytest.mark.parametrize("sub_comm", [1], indirect=['sub_comm'])
  # def test_mpi(self,sub_comm):
  #   if(sub_comm == MPI.COMM_NULL):
  #     return
  #   nRank = sub_comm.Get_size()
  #   iRank = sub_comm.Get_rank()
    
    
    
    

###############################################################################
