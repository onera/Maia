import pytest
import pytest_parallel
import mpi4py.MPI as MPI
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.pytree.yaml import parse_yaml_cgns

from maia.algo.dist  import s_to_u

###############################################################################
def test_n_face_per_dir():
  nVtx = np.array([7,9,5])
  assert (s_to_u.n_face_per_dir(nVtx, nVtx-1) == [7*8*4,6*9*4,6*8*5]).all()
###############################################################################

###############################################################################
class Test_compute_pointList_from_pointRanges():
  class Test_face():
    nVtx       = np.array([3, 3, 3], np.int32)
    loc        = "FaceCenter"
    # --------------------------------------------------------------------------- #
    def test_emptyRange(self):
      pointList  = s_to_u.compute_pointList_from_pointRanges([],self.nVtx,self.loc,0)
      assert (pointList.shape == (1,0))
      assert (pointList == np.empty((1,0), dtype=np.int32)).all()
    # --------------------------------------------------------------------------- #
    #FaceRange BC (Idir) : [[3,3], [1,2], [1,2]]
    def test_simple_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,2]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[3,6,9,12]]).all()
    # --------------------------------------------------------------------------- #
    def test_reversed_range(self):
      sub_ranges = [np.array([[3,3],[2,1],[1,2]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[6,3,12,9]]).all()
      sub_ranges = [np.array([[3,3],[2,1],[2,1]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[12,9,6,3]]).all()
    # --------------------------------------------------------------------------- #
    def test_multiple_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,1]]),
                    np.array([[3,3],[1,1],[2,2]]),
                    np.array([[3,3],[2,2],[2,2]])]
      pointList = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[3,6,9,12]]).all()
    # --------------------------------------------------------------------------- #
    #FaceRange BC (Jdir) : [[2,2], [1,1], [1,2]]
    def test_partial_bc(self):
      sub_ranges = [np.array([[2,2],[1,1],[1,2]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,1)
      assert (pointList == [[14,20]]).all()
    # --------------------------------------------------------------------------- #
    def test_partial_bc_multiple_range(self):
      sub_ranges = [np.array([[2,2],[1,1],[1,1]]),
                    np.array([[2,2],[1,1],[2,2]])]
      pointList = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,1)
      assert (pointList == [[14,20]]).all()

  class Test_vertex():
    nVtx       = np.array([3, 3, 3], np.int64)
    loc        = "Vertex"
    # --------------------------------------------------------------------------- #
    #VtxRange BC (IDir)= [[3,3], [1,3], [1,3]]
    def test_simple_range(self):
      sub_ranges = [np.array([[3,3],[1,3],[1,3]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[3,6,9,12,15,18,21,24,27]]).all()
    # --------------------------------------------------------------------------- #
    def test_multi_range(self):
      sub_ranges = [np.array([[3,3],[1,2],[1,1]]),
                    np.array([[3,3],[3,3],[1,1]]),
                    np.array([[3,3],[1,1],[2,2]]),
                    np.array([[3,3],[2,3],[2,2]]),
                    np.array([[3,3],[1,2],[3,3]]),
                    np.array([[3,3],[3,3],[3,3]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[3,6,9,12,15,18,21,24,27]]).all()
    # --------------------------------------------------------------------------- #
    #VtxRange BC (KDir)= [[2,3], [1,2], [1,1]]
    def test_simple_partial(self):
      sub_ranges = [np.array([[2,3],[1,2],[1,1]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,2)
      assert (pointList == [[2,3,5,6]]).all()

  class Test_cell():
    nVtx       = np.array([3, 3, 3])
    loc        = "CellCenter"
    # --------------------------------------------------------------------------- #
    def test_emptyRange(self):
      pointList  = s_to_u.compute_pointList_from_pointRanges([],self.nVtx,self.loc,0)
      assert (pointList.shape == (1,0))
      assert (pointList == np.empty((1,0), dtype=np.int32)).all()
    # --------------------------------------------------------------------------- #
    #CellRange BC (IDir)= [[2,2], [1,2], [1,2]]
    def test_simple_range(self):
      sub_ranges = [np.array([[2,2],[1,2],[1,2]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[2,4,6,8]]).all()
    # --------------------------------------------------------------------------- #
    def test_multiple_range(self):
      sub_ranges = [np.array([[2,2],[2,2],[1,1]]), #Assume we dont have first
                    np.array([[2,2],[1,2],[2,2]])]
      pointList  = s_to_u.compute_pointList_from_pointRanges(sub_ranges,self.nVtx,self.loc,0)
      assert (pointList == [[4,6,8]]).all()
###############################################################################

###############################################################################

# --------------------------------------------------------------------------- #

def test_bc_s_to_bc_u():
  #We dont test value of PL here, this is carried out by Test_compute_pointList_from_pointRanges
  n_vtx = np.array([4,4,4])
  bc_s = PT.new_BC('MyBCName', type='BCOutflow', point_range=[[1,4], [1,4], [4,4]])
  bc_u = s_to_u.bc_s_to_bc_u(bc_s, n_vtx, 'FaceCenter', 0, 1)
  assert PT.get_name(bc_u) == 'MyBCName'
  assert PT.get_value(bc_u) == 'BCOutflow'
  assert PT.Subset.GridLocation(bc_u) == 'FaceCenter'
  assert PT.get_value(PT.get_child_from_name(bc_u, 'PointList')).shape == (1,9)

def test_bcds_s_to_bcds_u():
  #We dont test value of PL here, this is carried out by Test_compute_pointList_from_pointRanges
  n_vtx = np.array([4,4,4])
  bc_s = PT.new_BC('MyBCName', type='BCOutflow', point_range=[[1,4], [1,4], [4,4]])
  MT.newDistribution({'Index' : [0,16,16]}, parent=bc_s)
  bcds = PT.new_child(bc_s, 'BCDataSet', 'BCDataSet_t')
  bcdata = PT.new_child(bcds, "DirichletData", "BCData_t")
  PT.new_DataArray("array", np.arange(16), parent=bcdata)
  bc_u = s_to_u.bc_s_to_bc_u(bc_s, n_vtx, 'FaceCenter', 0, 1)
  bcds = PT.get_node_from_name(bc_u, 'BCDataSet')
  assert bcds is not None
  assert PT.Subset.GridLocation(bcds) == 'Vertex'
  assert PT.get_node_from_name(bcds, 'PointList') is not None
  assert PT.get_node_from_name(bcds, 'array') is not None

def test_gc_s_to_gc_u():
  #https://cgns.github.io/CGNS_docs_current/sids/cnct.html
  #We dont test value of PL here, this is carried out by Test_compute_pointList_from_pointRanges
  n_vtx_A = np.array([17,9,7])
  n_vtx_B = np.array([7,9,5])
  gcA_s = PT.new_GridConnectivity1to1('matchA', 'Base/zoneB', point_range=[[17,17], [3,9], [1,5]], \
      point_range_donor=[[7,1], [9,9], [5,1]], transform = [-2,-1,-3])
  gcB_s = PT.new_GridConnectivity1to1('matchB', 'zoneA', point_range=[[7,1], [9,9], [5,1]], \
      point_range_donor=[[17,17], [3,9], [1,5]], transform = [-2,-1,-3])

  gcA_u = s_to_u.gc_s_to_gc_u(gcA_s, 'Base/zoneA', n_vtx_A, n_vtx_B, 'FaceCenter', 0, 1)
  gcB_u = s_to_u.gc_s_to_gc_u(gcB_s, 'Base/zoneB', n_vtx_B, n_vtx_A, 'FaceCenter', 0, 1)

  assert PT.get_name(gcA_u) == 'matchA'
  assert PT.get_name(gcB_u) == 'matchB'
  assert PT.get_value(gcA_u) == 'Base/zoneB'
  assert PT.get_value(gcB_u) == 'zoneA'
  assert PT.Subset.GridLocation(gcA_u) == 'FaceCenter'
  assert PT.Subset.GridLocation(gcB_u) == 'FaceCenter'
  assert PT.get_value(PT.get_child_from_label(gcA_u, 'GridConnectivityType_t')) == 'Abutting1to1'
  assert PT.get_value(PT.get_child_from_label(gcB_u, 'GridConnectivityType_t')) == 'Abutting1to1'
  assert PT.get_value(PT.get_child_from_name(gcA_u, 'PointList')).shape == (1,24)
  assert (PT.get_child_from_name(gcA_u, 'PointList')[1]\
          == PT.get_child_from_name(gcB_u, 'PointListDonor')[1]).all()
  assert (PT.get_child_from_name(gcB_u, 'PointList')[1]\
          == PT.get_child_from_name(gcA_u, 'PointListDonor')[1]).all()

  gcA_s = PT.new_GridConnectivity1to1('matchB', 'Base/zoneA', point_range=[[17,17], [3,9], [1,5]], \
      point_range_donor=[[7,1], [9,9], [5,1]], transform = [-2,-1,-3])
  gcB_s = PT.new_GridConnectivity1to1('matchA', 'zoneB', point_range=[[7,1], [9,9], [5,1]], \
      point_range_donor=[[17,17], [3,9], [1,5]], transform = [-2,-1,-3])
  gcA_u = s_to_u.gc_s_to_gc_u(gcA_s, 'Base/zoneB', n_vtx_B, n_vtx_A, 'FaceCenter', 0, 1)
  gcB_u = s_to_u.gc_s_to_gc_u(gcB_s, 'Base/zoneA', n_vtx_A, n_vtx_B, 'FaceCenter', 0, 1)
  assert (PT.get_child_from_name(gcA_u, 'PointList')[1]\
          == PT.get_child_from_name(gcB_u, 'PointListDonor')[1]).all()
  assert (PT.get_child_from_name(gcB_u, 'PointList')[1]\
          == PT.get_child_from_name(gcA_u, 'PointListDonor')[1]).all()

@pytest_parallel.mark.parallel(2)
def test_zonedims_to_ngon(comm):
  #We dont test value of faceVtx/ngon here, this is carried out by Test_compute_all_ngon_connectivity
  n_vtx_zone = np.array([3,2,4])
  ngon = s_to_u.zonedims_to_ngon(n_vtx_zone, comm)
  n_faces = PT.get_child_from_name(ngon, "ElementStartOffset")[1].shape[0] - 1
  if comm.Get_rank() == 0:
    expected_n_faces = 15
    expected_eso     = 4*np.arange(0,15+1)
  elif comm.Get_rank() == 1:
    expected_n_faces = 14
    expected_eso     = 4*np.arange(15, 15+14+1)
  assert n_faces == expected_n_faces
  assert (PT.get_child_from_name(ngon, 'ElementRange')[1] == [1, 29]).all()
  assert (PT.get_child_from_name(ngon, 'ElementStartOffset')[1] == expected_eso).all()
  assert PT.get_node_from_path(ngon, ':CGNS#Distribution/ElementConnectivity')[1][2] == 4*29
  assert PT.get_child_from_name(ngon, 'ParentElements')[1].shape == (expected_n_faces, 2)
  assert PT.get_child_from_name(ngon, 'ElementConnectivity')[1].shape == (4*expected_n_faces,)
###############################################################################
