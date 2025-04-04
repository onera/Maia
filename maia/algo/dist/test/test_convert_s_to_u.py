import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype
stype = 'I4' if pdm_dtype == np.int32 else 'I8'

import maia

from maia.algo.dist  import s_to_u

def test_generate_all_bnd_bcs():
  bcs = s_to_u.generate_all_bnd_bcs([42,8], np.int32)
  assert len(bcs) == 4
  assert all(PT.get_label(n) == 'BC_t' for n in bcs)
  assert all(PT.Subset.getPatch(n)[1].dtype == np.int32 for n in bcs)
  assert (PT.get_child_from_name(bcs[0], 'PointRange')[1] == [[1,1],[1,8]]).all()
  assert (PT.get_child_from_name(bcs[1], 'PointRange')[1] == [[42,42],[1,8]]).all()
  assert (PT.get_child_from_name(bcs[2], 'PointRange')[1] == [[1,42],[1,1]]).all()
  assert (PT.get_child_from_name(bcs[3], 'PointRange')[1] == [[1,42],[8,8]]).all()

  bcs = s_to_u.generate_all_bnd_bcs([42,8,24], np.int64)
  assert len(bcs) == 6
  assert all(PT.get_label(n) == 'BC_t' for n in bcs)
  assert all(PT.Subset.getPatch(n)[1].dtype == np.int64 for n in bcs)
  assert (PT.get_child_from_name(bcs[0], 'PointRange')[1] == [[1,1],[1,8],[1,24]]).all()
  assert (PT.get_child_from_name(bcs[1], 'PointRange')[1] == [[42,42],[1,8],[1,24]]).all()
  assert (PT.get_child_from_name(bcs[2], 'PointRange')[1] == [[1,42],[1,1],[1,24]]).all()
  assert (PT.get_child_from_name(bcs[3], 'PointRange')[1] == [[1,42],[8,8],[1,24]]).all()
  assert (PT.get_child_from_name(bcs[4], 'PointRange')[1] == [[1,42],[1,8],[1,1]]).all()
  assert (PT.get_child_from_name(bcs[5], 'PointRange')[1] == [[1,42],[1,8],[24,24]]).all()


###############################################################################
def test_n_face_per_dir():
  nVtx = np.array([7,9,5])
  assert s_to_u.n_face_per_dir(nVtx) == (7*8*4,6*9*4,6*8*5)

def test_n_edge_per_dir():
  nVtx = np.array([6,3])
  assert s_to_u.n_edge_per_dir(nVtx) == (12,15)
###############################################################################

def test_get_output_loc():
  assert s_to_u.get_output_loc({}, PT.new_BC(loc='CellCenter')) == ['CellCenter']
  assert s_to_u.get_output_loc({}, PT.new_BC(loc='JFaceCenter')) == ['FaceCenter']
  assert s_to_u.get_output_loc({'BC_t' : 'Vertex'}, PT.new_BC(loc='CellCenter')) == ['Vertex']
  assert s_to_u.get_output_loc({'BC_t' : 'EdgeCenter'}, PT.new_BC(loc='CellCenter')) == ['EdgeCenter']

def test_s_location():
  assert s_to_u._s_location('EdgeCenter', 1) == 'JEdgeCenter'
  assert s_to_u._s_location('Vertex', 1) == 'Vertex'
  assert s_to_u._s_location('FaceCenter', 2) == 'KFaceCenter'

###############################################################################

# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('output_loc', ['FaceCenter', 'Vertex'])
def test_bc_s_to_bc_u(output_loc):
  n_vtx = np.array([4,4,4])
  bc_s = PT.new_BC('MyBCName', type='BCOutflow', point_range=[[1,4], [1,4], [3,3]])
  bc_u = s_to_u.bc_s_to_bc_u(bc_s, n_vtx, output_loc, 0, 1)
  assert PT.get_name(bc_u) == 'MyBCName'
  assert PT.get_value(bc_u) == 'BCOutflow'
  assert PT.Subset.GridLocation(bc_u) == output_loc
  if output_loc == 'FaceCenter':
    assert PT.get_value(PT.get_child_from_name(bc_u, 'PointList')).shape == (1,9)
    assert np.array_equal(PT.get_child_from_name(bc_u, 'PointList')[1], [[91,92,93,94,95,96,97,98,99]])
  elif output_loc == 'Vertex':
    assert PT.get_value(PT.get_child_from_name(bc_u, 'PointList')).shape == (1,16)
    assert np.array_equal(PT.get_child_from_name(bc_u, 'PointList')[1], [[33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]])

@pytest.mark.parametrize('output_loc', ['Vertex', 'FaceCenter'])
def test_bcds_s_to_bcds_u(output_loc):
  n_vtx = np.array([4,4,4])
  bc_s = PT.new_BC('MyBCName', type='BCOutflow', point_range=[[1,4], [1,4], [4,4]])
  distri = MT.newDistribution({'Index' : [0,16,16]}, parent=bc_s)

  # A dataset w/o PointRange
  ds = PT.new_BCDataSet('RelatedDS', parent=bc_s)
  bcdata = PT.new_BCData("NeumannData", fields={'array' : np.ones(16)}, parent=ds)

  # A dataset still Vertex located, but with its PointRange
  ds = PT.new_BCDataSet('CustomVertexDS', point_range=[[2,3], [2,3], [4,4]], parent=bc_s)
  PT.maia.newDistribution({'Index' : [0,4,4]}, parent=ds)
  PT.new_BCData('NeumannData', fields={'array' : 2*np.ones(4)}, parent=ds)

  # A dataset already FaceCenter located
  ds = PT.new_BCDataSet('CustomFaceDS', point_range=[[1,3], [1,3], [4,4]], loc='KFaceCenter', parent=bc_s)
  PT.maia.newDistribution({'Index' : [0,9,9]}, parent=ds)
  PT.new_BCData('NeumannData', fields={'array' : 3*np.ones(9)}, parent=ds)

  bc_u = s_to_u.bc_s_to_bc_u(bc_s, n_vtx, output_loc, 0, 1)

  # If BC loc is preserved, related DS remains related. Otherwise, it stays Vertex and becomes unrelated
  ds = PT.get_child_from_name(bc_u, 'RelatedDS')
  if output_loc == 'FaceCenter':
    assert (l:=PT.get_child_from_name(ds, 'GridLocation')) is not None and PT.get_value(l) == 'Vertex'
    assert np.array_equal(PT.get_child_from_name(ds, 'PointList')[1], np.arange(49,64+1).reshape((1,-1)))
  elif output_loc == 'Vertex':
    assert PT.get_child_from_name(ds, 'GridLocation') is None
    assert PT.get_child_from_name(ds, 'PointList') is None
  assert PT.get_node_from_name(ds, 'array') is not None

  # Unrelated DS keeps its location and get a new PL
  ds = PT.get_child_from_name(bc_u, 'CustomVertexDS')
  assert PT.BCDataSet.GridLocation(ds, bc_u) == 'Vertex'
  assert np.array_equal(PT.get_child_from_name(ds, 'PointList')[1], [[54,55,58,59]])
  assert PT.get_node_from_name(ds, 'array') is not None

  ds = PT.get_child_from_name(bc_u, 'CustomFaceDS')
  assert PT.BCDataSet.GridLocation(ds, bc_u) == 'FaceCenter'
  assert np.array_equal(PT.get_child_from_name(ds, 'PointList')[1], [[100,101,102,103,104,105,106,107,108]])
  assert PT.get_node_from_name(ds, 'array') is not None



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
  ngon = s_to_u.zonedims_to_ngon(n_vtx_zone, comm, np.int32)
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

@pytest_parallel.mark.parallel(2)
def test_s_to_u_2d_elt(comm):
  
  tree = maia.factory.generate_dist_block([6,3], 'S', comm)
  PT.rm_nodes_from_name(tree, 'Xmin')
  PT.rm_nodes_from_name(tree, 'Ymax')

  maia.algo.dist.convert_s_to_u(tree, 'Standard', comm)

  if comm.rank == 0:
    expected_bar_ec = np.array([7, 1, 6, 12, 13, 7, 12, 18, 1, 2, 2, 3], pdm_dtype)
    expected_bar_distri = np.array([0,6,14], pdm_dtype)
    expected_quad_ec = np.array([1, 2, 8, 7, 2, 3, 9, 8, 3, 4, 10, 9, 4, 5, 11, 10, 5, 6, 12, 11], pdm_dtype)
    expected_quad_distri = np.array([0,5,10], pdm_dtype)
  elif comm.rank == 1:
    expected_bar_ec = np.array([3, 4, 4, 5, 5, 6, 14, 13, 15, 14, 16, 15, 17, 16, 18, 17], pdm_dtype)
    expected_bar_distri = np.array([6,14,14], pdm_dtype)
    expected_quad_ec = np.array([7, 8, 14, 13, 8, 9, 15, 14, 9, 10, 16, 15, 10, 11, 17, 16, 11, 12, 18, 17], pdm_dtype)
    expected_quad_distri = np.array([5,10,10], pdm_dtype)

  expected_bar = PT.new_Elements('BAR_2', 'BAR_2', erange=np.array([1,14], pdm_dtype), econn=expected_bar_ec)
  MT.new_distribution({'Element' : expected_bar_distri}, expected_bar)
  expected_quad = PT.new_Elements('QUAD_4', 'QUAD_4', erange=np.array([15,24], pdm_dtype), econn=expected_quad_ec)
  MT.new_distribution({'Element' : expected_quad_distri}, expected_quad)

  assert PT.is_same_tree(PT.get_node_from_name(tree, 'BAR_2'), expected_bar)
  assert PT.is_same_tree(PT.get_node_from_name(tree, 'QUAD_4'), expected_quad)

@pytest_parallel.mark.parallel(3)
def test_s_to_u_3d_elt(comm):
  tree = maia.factory.generate_dist_block([4,3,2], 'S', comm)
  maia.algo.dist.convert_s_to_u(tree, 'Standard', comm, subset_loc={'BC_t' : 'FaceCenter'})

  if comm.rank == 0:
    expected_quad_ec = np.array([1,13,17,5, 4,8,20,16, 5,17,21,9, 8,12,24,20, 1,2,14,13, 2,3,15,14], pdm_dtype)
    expected_quad_distri = np.array([0,6,22], pdm_dtype)
    expected_hexa_ec = np.array([1,2,6,5,13,14,18,17, 2,3,7,6,14,15,19,18], pdm_dtype)
    expected_hexa_distri = np.array([0,2,6], pdm_dtype)
    expected_zmax_pl = np.array([[17,18]], pdm_dtype)
    expected_zmax_distri = np.array([0,2,6], pdm_dtype)
  elif comm.rank == 1:
    expected_quad_ec = np.array([3,4,16,15, 10,9,22,21, 11,10,23,22, 12,11,24,23, 1,5,6,2, 2,6,7,3, 3,7,8,4], pdm_dtype)
    expected_quad_distri = np.array([6,13,22], pdm_dtype)
    expected_hexa_ec = np.array([3,4,8,7,15,16,20,19, 5,6,10,9,17,18,22,21], pdm_dtype)
    expected_hexa_distri = np.array([2,4,6], pdm_dtype)
    expected_zmax_pl = np.array([[19,20]], pdm_dtype)
    expected_zmax_distri = np.array([2,4,6], pdm_dtype)
  elif comm.rank == 2:
    expected_quad_ec = np.array([5,9,10,6, 6,10,11,7, 7,11,12,8, 13,14,18,17, 14,15,19,18,
                                15,16,20,19, 17,18,22,21, 18,19,23,22, 19,20,24,23], pdm_dtype)
    expected_quad_distri = np.array([13,22,22], pdm_dtype)
    expected_hexa_ec = np.array([6,7,11,10,18,19,23,22, 7,8,12,11,19,20,24,23], pdm_dtype)
    expected_hexa_distri = np.array([4,6,6], pdm_dtype)
    expected_zmax_pl = np.array([[21,22]], pdm_dtype)
    expected_zmax_distri = np.array([4,6,6], pdm_dtype)
  
  expected_bar = PT.new_Elements('QUAD_4', 'QUAD_4', erange=np.array([1,22], pdm_dtype), econn=expected_quad_ec)
  MT.new_distribution({'Element' : expected_quad_distri}, expected_bar)
  expected_quad = PT.new_Elements('HEXA_8', 'HEXA_8', erange=np.array([23,28], pdm_dtype), econn=expected_hexa_ec)
  MT.new_distribution({'Element' : expected_hexa_distri}, expected_quad)
  expected_zmax = PT.new_BC('Zmax', 'Null', loc='FaceCenter', point_list=expected_zmax_pl)
  MT.new_distribution({'Index' : expected_zmax_distri}, expected_zmax)

  assert PT.is_same_tree(PT.get_node_from_name(tree, 'QUAD_4'), expected_bar)
  assert PT.is_same_tree(PT.get_node_from_name(tree, 'HEXA_8'), expected_quad)
  assert PT.is_same_tree(PT.get_node_from_name(tree, 'Zmax'), expected_zmax)

@pytest_parallel.mark.parallel([1,3])
def test_s_to_u_2d(comm):
  tree = maia.factory.generate_dist_block([6,3], 'S', comm)
  maia.algo.dist.convert_s_to_ngon(tree, comm)
  edge = PT.maia.Zone.EdgeNode(PT.get_node_from_label(tree, 'Zone_t'))
  
  if comm.Get_size() == 1:
    expt_distri = [0,27,27]
  elif comm.Get_size() == 3:
    expt_distri = [[0,9,27], [9,18,27], [18,27,27]][comm.Get_rank()]

  expt_edge_vtx = np.array([7,1, 2,8, 3,9, 4,10, 5,11, 6,12,
                            13,7, 8,14, 9,15, 10,16, 11,17, 12,18,
                            1,2, 2,3, 3,4, 4,5, 5,6,
                            7,8, 8,9, 9,10, 10,11, 11,12,
                            14,13 ,15,14, 16,15, 17,16, 18,17])[2*expt_distri[0]:2*expt_distri[1]]

  expt_edge_face = np.array([[28, 0], [28,29], [29,30], [30,31], [31,32], [32, 0],
                             [33, 0], [33,34], [34,35], [35,36], [36,37], [37, 0],
                             [28, 0], [29, 0], [30, 0], [31, 0], [32, 0],
                             [33,28], [34,29], [35,30], [36,31], [37,32],
                             [33, 0], [34, 0], [35, 0], [36, 0], [37, 0]])[expt_distri[0]:expt_distri[1]]

  assert np.array_equal(PT.get_node_from_name(edge, 'Element')[1],              expt_distri)
  assert np.array_equal(PT.get_child_from_name(edge, 'ElementRange')[1],        np.array([1,27]))
  assert np.array_equal(PT.get_child_from_name(edge, 'ElementConnectivity')[1], expt_edge_vtx)
  assert np.array_equal(PT.get_child_from_name(edge, 'ParentElements')[1],      expt_edge_face)
  
@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize('bc_loc_edge', [False, True])
def test_s_to_u_2d_dataset(bc_loc_edge, comm):
  tree = maia.factory.generate_dist_block([6,3], 'S', comm)

  # NB : Ymax is defined by PointRange [[1,6], [3,3]]
  ymax = PT.get_node_from_name(tree, 'Ymax')

  # A dataset w/o PointRange
  ds = PT.new_BCDataSet('RelatedDS', parent=ymax)
  PT.new_BCData('NeumannData', fields={'Test1' : np.ones(6)}, parent=ds)

  # A dataset still Vertex located, but with its PointRange
  ds = PT.new_BCDataSet('CustomVertexDS', point_range=[[1,3], [3,3]], parent=ymax)
  PT.maia.newDistribution({'Index' : [0,3,3]}, parent=ds)
  PT.new_BCData('NeumannData', fields={'Test2' : 2*np.ones(3)}, parent=ds)

  # A dataset already EdgeCenter located
  ds = PT.new_BCDataSet('CustomEdgeDS', point_range=[[1,5], [3,3]], loc='JEdgeCenter', parent=ymax)
  PT.maia.newDistribution({'Index' : [0,5,5]}, parent=ds)
  PT.new_BCData('NeumannData', fields={'Test3' : 3*np.ones(5)}, parent=ds)

  zbc = PT.get_node_from_label(tree, 'ZoneBC_t')
  PT.keep_children_from_name(zbc, 'Ymax')
  
  loc = {'BC_t' : 'EdgeCenter'} if bc_loc_edge else {}
  maia.algo.dist.convert_s_to_u(tree, 'Poly', comm, loc)
  
  ymax = PT.get_node_from_name(tree, 'Ymax')
  if bc_loc_edge:
    assert PT.Subset.GridLocation(ymax) == 'EdgeCenter'
    assert PT.get_label(PT.Subset.getPatch(ymax)) == 'IndexArray_t'
    assert (PT.get_value(PT.Subset.getPatch(ymax)) == [[23,24,25,26,27]]).all()
  else:
    assert PT.Subset.GridLocation(ymax) == 'Vertex'
    assert PT.get_label(PT.Subset.getPatch(ymax)) == 'IndexArray_t'
    assert (PT.get_value(PT.Subset.getPatch(ymax)) == [[13,14,15,16,17,18]]).all()

  # If BC loc is preserved, related DS remains related. Otherwise, it stays Vertex and becomes unrelated
  ds = PT.get_child_from_name(ymax, 'RelatedDS')
  if bc_loc_edge:
    assert (l:=PT.get_child_from_name(ds, 'GridLocation')) is not None and PT.get_value(l) == 'Vertex'
    assert (PT.get_child_from_name(ds, 'PointList')[1] == [[13,14,15,16,17,18]]).all()
  else:
    assert PT.get_child_from_name(ds, 'GridLocation') is None
    assert PT.get_child_from_name(ds, 'PointList') is None

  # Unrelated DS keeps its location and get a new PL
  ds = PT.get_child_from_name(ymax, 'CustomVertexDS')
  assert PT.BCDataSet.GridLocation(ds, ymax) == 'Vertex'
  assert (PT.get_child_from_name(ds, 'PointList')[1] == [[13,14,15]]).all()

  ds = PT.get_child_from_name(ymax, 'CustomEdgeDS')
  assert PT.BCDataSet.GridLocation(ds, ymax) == 'EdgeCenter'
  assert (PT.get_child_from_name(ds, 'PointList')[1] == [[23,24,25,26,27]]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("jn_loc", ['Vertex', 'EdgeCenter'])
@pytest.mark.parametrize("connectivity", ['Poly', 'Standard'])
def test_s_to_u_2d_gc(connectivity, jn_loc, comm):

  treeA = maia.factory.generate_dist_block([6,3], 'S', comm, length=[5, 2])
  treeB = maia.factory.generate_dist_block([5,4], 'S', comm, length=[4, 3])
  maia.algo.transform_affine(treeB, rotation_angle=[0,0,np.pi/2])
  maia.algo.transform_affine(treeB, translation=[8,-1,0])
  zoneA = PT.get_node_from_label(treeA, 'Zone_t')
  PT.set_name(zoneA, 'Left')
  zoneB = PT.get_node_from_label(treeB, 'Zone_t')
  PT.set_name(zoneB, 'Right')

  pr_left = np.array([[6,6],[1,3]], pdm_dtype)
  pr_right = np.array([[2,4],[4,4]], pdm_dtype)
  gc = PT.new_GridConnectivity1to1('Xmax', 'Right', point_range=pr_left, point_range_donor=pr_right, transform=[-2,1])
  PT.new_node('ZoneGridConnectivity', 'ZoneGridConnectivity_t', children=[gc], parent=zoneA)

  gc = PT.new_GridConnectivity1to1('Xmin', 'Left', point_range=pr_right, point_range_donor=pr_left, transform=[2,-1])
  PT.new_node('ZoneGridConnectivity', 'ZoneGridConnectivity_t', children=[gc], parent=zoneB)

  tree = PT.union(treeA, treeB)
  # Only test GCs in this function
  PT.rm_nodes_from_label(tree, 'ZoneBC_t')

  maia.algo.dist.convert_s_to_u(tree, connectivity, comm, {'GC_t' : jn_loc})

  if jn_loc == 'Vertex':
    pl_left  = [[6,12]]  if comm.rank == 0 else [[18]]
    pl_right = [[17,18]] if comm.rank == 0 else [[19]]
    distri = [0,2,3]     if comm.rank == 0 else [2,3,3]
  elif jn_loc == 'EdgeCenter':
    if connectivity == 'Poly':
      pl_left  = [[6]]  if comm.rank == 0 else [[12]]
      pl_right = [[29]] if comm.rank == 0 else [[30]]
    elif connectivity == 'Standard':
      pl_left  = [[2]]  if comm.rank == 0 else [[4]]
      pl_right = [[12]] if comm.rank == 0 else [[13]]
    distri = [0,1,2]  if comm.rank == 0 else [1,2,2]

  zgc_left = PT.yaml.to_node(f"""
  ZoneGridConnectivity ZoneGridConnectivity_t:
    Xmax GridConnectivity_t "Right":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridConnectivityDonorName Descriptor_t "Xmin":
      GridLocation GridLocation_t "{jn_loc}":
      PointList IndexArray_t {stype} {pl_left}:
      PointListDonor IndexArray_t {stype} {pl_right}:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {stype} {distri}:
  """)
  zgc_right = PT.yaml.to_node(f"""
  ZoneGridConnectivity ZoneGridConnectivity_t:
    Xmin GridConnectivity_t "Left":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridConnectivityDonorName Descriptor_t "Xmax":
      GridLocation GridLocation_t "{jn_loc}":
      PointList IndexArray_t {stype} {pl_right}:
      PointListDonor IndexArray_t {stype} {pl_left}:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {stype} {distri}:
  """)

  zoneA = PT.get_node_from_name(tree, 'Left')
  zoneB = PT.get_node_from_name(tree, 'Right')
  assert PT.is_same_tree(PT.get_child_from_label(zoneA, 'ZoneGridConnectivity_t'), zgc_left)
  assert PT.is_same_tree(PT.get_child_from_label(zoneB, 'ZoneGridConnectivity_t'), zgc_right)
  

@pytest.mark.parametrize("connectivity", ['Poly', 'Standard'])
def test_cell_center_subset_shift(connectivity, comm):
  tree = maia.factory.generate_dist_block(4, 'S', comm)

  bc_s = PT.new_BC('BC', type='BCOutflow', loc='CellCenter', point_range=[[1,3], [2,3], [1,1]])
  PT.maia.newDistribution({'Index' : [0,6,6]}, parent=bc_s)
  zbc = PT.get_node_from_label(tree, 'ZoneBC_t')
  PT.set_children(zbc, [bc_s])

  maia.algo.dist.convert_s_to_u(tree, connectivity, comm)

  before_shift = np.array([[4,5,6, 7,8,9]])
  if connectivity == 'Poly':
    shift = 3*(4*3*3) # Shift will all internal faces
  else:
    shift = 6*(3*3) # Shift will only external faces

  bc = PT.get_node_from_label(tree, 'BC_t')
  assert PT.Subset.GridLocation(bc) == 'CellCenter'
  assert (PT.get_child_from_name(bc, 'PointList')[1] == before_shift + shift).all()