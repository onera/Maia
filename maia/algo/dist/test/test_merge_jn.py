import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia              import npy_pdm_gnum_dtype as pdm_dtype
from maia.factory      import dcube_generator
from maia.pytree.yaml  import parse_yaml_cgns
from maia.utils        import par_utils

from maia.algo.dist    import merge_jn as MJ

@pytest_parallel.mark.parallel(2)
def test_update_ngon(comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]
  PT.rm_nodes_from_label(tree, 'ZoneBC_t')

  ngon = PT.get_node_from_name(zone, 'NGonElements')
  vtx_distri_ini = PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex')[1]

  #from maia.transform.dist_tree.merge_ids import merge_distributed_ids
  #old_to_new_vtx  = merge_distributed_ids(vtx_distri_ini, np.array([12,21,15,24,18,27]), \
      # np.array([10,19,13,22,16,25]), comm) #Used to get old_to_new_vtx
  old_to_new_vtx_full = np.array([1,2,3,4,5,6,7,8,9,10,9,11,12,11,13,14,13,15,16,15,17,18,17,19,20,19,21])

  old_to_new_vtx = old_to_new_vtx_full[vtx_distri_ini[0]:vtx_distri_ini[1]]
  if comm.Get_rank() == 1:
    ref_faces = np.array([15,23], dtype=pdm_dtype)
    del_faces = np.array([16,24], dtype=pdm_dtype)
  else:
    ref_faces = np.array([], dtype=pdm_dtype)
    del_faces = np.array([], dtype=pdm_dtype)

  expected_pe_full = np.array([[35,0],[36,0],[37,0],[38,0],[35,39],[36,40],[37,41],[38,42],[39,0],[40,0],[41,0],[42,0],
                               [35,0],[37,0],[39,41],[35,36],[37,38],[39,40],[41,42],[36,0],[38,0],[40,42],[35,0],[39,0],
                               [36,0],[40,0],[35,37],[39,41],[36,38],[40,42],[37,0],[41,0],[38,0],[42,0]])
  expected_ec_full = np.array([ 2, 5, 4, 1,  3, 6, 5, 2,  5, 8, 7, 4,  6, 9, 8, 5, 10,12,11, 9,  9,11,13,11,
                               12,14,13,11, 11,13,15,13, 16,18,17,15, 15,17,19,17, 18,20,19,17, 17,19,21,19,
                                1, 4,12,10,  4, 7,14,12, 10,12,18,16,  9,11, 5, 2, 11,13, 8, 5, 15,17,11, 9,
                               17,19,13,11, 11,13, 6, 3, 13,15, 9, 6, 17,19,13,11, 10, 9, 2, 1, 16,15, 9,10,
                                9,11, 3, 2, 15,17,11, 9,  4, 5,11,12, 12,11,17,18,  5, 6,13,11, 11,13,19,17,
                                7, 8,13,14, 14,13,19,20,  8, 9,15,13, 13,15,21,19])
  expected_eso_f = np.arange(0, 4*34+1, 4)

  MJ._update_ngon(ngon, ref_faces, del_faces, vtx_distri_ini, old_to_new_vtx, comm)

  start, end     = PT.get_node_from_path(ngon, ':CGNS#Distribution/Element')[1][[0,1]]
  start_e, end_e = PT.get_node_from_path(ngon, ':CGNS#Distribution/ElementConnectivity')[1][[0,1]]
  assert (PT.get_node_from_name(ngon, 'ElementRange')[1]        == [1,34]                      ).all()
  assert (PT.get_node_from_name(ngon, 'ParentElements')[1]      == expected_pe_full[start:end]    ).all()
  assert (PT.get_node_from_name(ngon, 'ElementConnectivity')[1] == expected_ec_full[start_e:end_e]).all()
  assert (PT.get_node_from_name(ngon, 'ElementStartOffset')[1]  == expected_eso_f[start:end+1] ).all()

@pytest_parallel.mark.parallel(2)
def test_update_nface(comm):
  #Create nface node, not generated by dcube gen (cube 2*2*2 cells)
  nface_ec_full = [1, 5, 13, 17, 25, 29,  2,  6, -17, 21, 27, 31,  3,  7, 14, 18, -29, 33,  4,  8, -18, 22, -31, 35, \
                  -5, 9, 15, 19, 26, 30, -6, 10, -19, 23, 28, 32, -7, 11, 16, 20, -30, 34, -8, 12, -20, 24, -32, 36]
  eso_full      = np.arange(0, 6*8+1, 6)

  face_distri_ini = par_utils.uniform_distribution(36, comm)
  cell_distri_ini = par_utils.uniform_distribution((3-1)**3, comm)
  cell_distri_ini_e = par_utils.uniform_distribution(6*(3-1)**3, comm)
  nface = PT.new_NFaceElements(erange = [36+1,36+8],
                               eso    = eso_full[cell_distri_ini[0]:cell_distri_ini[1]+1],
                               ec     = nface_ec_full[cell_distri_ini_e[0]:cell_distri_ini_e[1]])
  MT.newDistribution({'Element' : cell_distri_ini, 'ElementConnectivity' : cell_distri_ini_e}, nface)

  #from maia.transform.dist_tree.merge_ids import merge_distributed_ids
  #old_to_new_face = merge_distributed_ids(face_distri_ini, np.array([23,24]), np.array([15,16]), comm, True)
  old_to_new_face_f = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,-15,-16,23,24,25,26,27,28,29,30,31,32,33,34]
  old_to_new_face = np.array(old_to_new_face_f[face_distri_ini[0]:face_distri_ini[1]])

  MJ._update_nface(nface, face_distri_ini, old_to_new_face, n_rmvd_face=2, comm=comm)

  expected_eso_f = np.arange(0, 4*34+1, 4)

  expected_ec_full = [1, 5, 13, 17, 23, 27, 2, 6, -17, 21, 25, 29, 3, 7, 14, 18, -27, 31, 4, 8, -18, 22, -29, 33, \
                  -5, 9, 15, 19, 24, 28, -6, 10, -19, -1*15, 26, 30, -7, 11, 16, 20, -28, 32, -8, 12, -20, -1*16, -30, 34]

  start_e, end_e = PT.get_node_from_path(nface, ':CGNS#Distribution/ElementConnectivity')[1][[0,1]]
  assert (PT.get_node_from_name(nface, 'ElementConnectivity')[1] == expected_ec_full[start_e:end_e]).all()
  assert (PT.get_node_from_name(nface, 'ElementStartOffset')[1]  == eso_full[cell_distri_ini[0]:cell_distri_ini[1]+1]).all()

@pytest_parallel.mark.parallel(2)
def test_update_subset(comm):
  bc = PT.new_BC('BC')
  bc_distri = par_utils.uniform_distribution(5, comm)
  this_rank = slice(bc_distri[0], bc_distri[1])
  bcds   = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
  bcdata = PT.new_child(bcds, 'BCData', 'BCData_t')

  data = np.array([10,20,30,20,50][this_rank], dtype=np.float64)
  PT.new_DataArray('ArrayA', data,   parent=bcdata)
  PT.new_DataArray('ArrayB', 2*data, parent=bcdata)

  pl_new = np.array([1,2,3,2,4][this_rank], pdm_dtype)
  MJ._update_subset(bc, pl_new, ['BCDataSet_t', 'BCData_t', 'DataArray_t'], comm)

  new_distri = par_utils.uniform_distribution(4, comm)
  this_rank = slice(new_distri[0], new_distri[1])
  assert (PT.get_node_from_path(bc, ':CGNS#Distribution/Index')[1] == new_distri).all()
  assert (PT.get_node_from_name(bc, 'PointList')[1] ==   [1,2,3,4][this_rank]).all()
  assert (PT.get_node_from_name(bc, 'ArrayA')[1] ==  [10,20,30,50][this_rank]).all()
  assert (PT.get_node_from_name(bc, 'ArrayB')[1] == [20,40,60,100][this_rank]).all()

@pytest_parallel.mark.parallel(2)
def test_update_cgns_subsets(comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]
  #Move some node to diversify the test
  zgc = PT.new_ZoneGridConnectivity('ZoneGC', parent=zone)
  for bcname in ['Ymin', 'Ymax']:
    bc = PT.get_node_from_name(zone, bcname)
    PT.set_label(bc, 'GridConnectivity_t')
    PT.add_child(zgc, bc)
    PT.rm_child(PT.get_child_from_label(zone, 'ZoneBC_t'), bc)
  bc = PT.get_node_from_name(zone, 'Zmin')
  PT.set_label(bc, 'ZoneSubRegion_t')
  PT.add_child(zone, bc)
  PT.rm_child(PT.get_child_from_label(zone, 'ZoneBC_t'), bc)
  PT.new_DataArray('SubSol', np.copy(PT.get_node_from_name(bc, 'PointList')[1][0]), parent=bc)
  bc = PT.get_node_from_name(zone, 'Zmax')
  PT.set_label(bc, 'FlowSolution_t')
  PT.add_child(zone, bc)
  PT.rm_child(PT.get_child_from_label(zone, 'ZoneBC_t'), bc)
  PT.new_DataArray('Sol', np.copy(PT.get_node_from_name(bc, 'PointList')[1][0]), parent=bc)

  face_distri_ini = PT.get_value(MT.getDistribution(PT.get_node_from_path(zone, 'NGonElements'), 'Element'))
  old_to_new_face_f = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,15,16,23,24,25,26,27,28,29,30,31,32,33,34]
  old_to_new_face = np.array(old_to_new_face_f[face_distri_ini[0]:face_distri_ini[1]])
  MJ._update_cgns_subsets(zone, 'FaceCenter', face_distri_ini, old_to_new_face, 'Base', comm)

  bc_distri = par_utils.uniform_distribution(4, comm)
  this_rank = slice(bc_distri[0], bc_distri[1])
  assert (PT.get_node_from_path(zone, 'Zmin/PointList')[1] == [[1,2,3,4][this_rank]]).all()
  assert (PT.get_node_from_path(zone, 'Zmax/PointList')[1] == [[9,10,11,12][this_rank]]).all()
  assert (PT.get_node_from_path(zone, 'ZoneBC/Xmin/PointList')[1] == [[13,14,15,16][this_rank]]).all()
  assert (PT.get_node_from_path(zone, 'ZoneBC/Xmax/PointList')[1] == [[15,16,21,22][this_rank]]).all()
  assert (PT.get_node_from_path(zone, 'ZoneGC/Ymin/PointList')[1] == [[23,24,25,26][this_rank]]).all()
  assert (PT.get_node_from_path(zone, 'ZoneGC/Ymax/PointList')[1] == [[31,32,33,34][this_rank]]).all()
  assert (PT.get_node_from_path(zone, 'Zmin/PointList')[1] == PT.get_node_from_path(zone, 'Zmin/SubSol')[1]).all()
  assert (PT.get_node_from_path(zone, 'Zmax/PointList')[1] == PT.get_node_from_path(zone, 'Zmax/Sol')[1]).all()

def test_shift_cgns_subsets():
  yt = """
  Zone Zone_t:
    ZBC ZoneBC_t:
      bc1 BC_t:
        PointList IndexArray_t [[10,14,12,16]]:
        GridLocation GridLocation_t "CellCenter":
      bc2 BC_t:
        PointList IndexArray_t [[1,3,5,7]]:
        GridLocation GridLocation_t "Vertex":
      bc3 BC_t:
        PointList IndexArray_t [[1,3,5,7]]:
        GridLocation GridLocation_t "Vertex":
        BCDS BCDataSet_t:
          PointList IndexArray_t [[50,100]]:
          GridLocation GridLocation_t "CellCenter":
    ZSR ZoneSubRegion_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[100]]:
  """
  zone = parse_yaml_cgns.to_node(yt)

  MJ._shift_cgns_subsets(zone, 'CellCenter', -4)

  assert (PT.get_node_from_path(zone, 'ZBC/bc1/PointList')[1]      == [[6,10,8,12]]).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc2/PointList')[1]      == [[1,3,5,7]]  ).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc3/PointList')[1]      == [[1,3,5,7]]  ).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc3/BCDS/PointList')[1] == [[46,96]]    ).all()
  assert (PT.get_node_from_path(zone, 'ZSR/PointList')[1]          == [[96]]       ).all()

@pytest_parallel.mark.parallel(2)
def test_update_vtx_data(comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]
  PT.rm_nodes_from_label(tree, 'ZoneBC_t')
  distri = PT.get_value(MT.getDistribution(zone, 'Vertex'))
  fs = PT.new_FlowSolution('FSol', loc='Vertex', parent=zone)
  sol = PT.new_DataArray('Sol', np.arange(27)[distri[0]:distri[1]]+1, parent=fs)

  if comm.Get_rank() == 0:
    vtx_to_remove = np.array([12,21,15,24])
    expected_distri = np.array([0,13,21])
    expected_cx = np.array([0.,0.5,1.,0.,0.5,1.,0.,0.5,1.,0.,0.5,0.,0.5])
    expected_sol = np.array([1,2,3,4,5,6,7,8,9,10,11,13,14])
  elif comm.Get_rank() == 1:
    vtx_to_remove = np.array([18,27])
    expected_distri = np.array([13,21,21])
    expected_cx = np.array([0.,0.5,0.,0.5,0.,0.5,0.,0.5])
    expected_sol = np.array([16,17,19,20,22,23,25,26])

  MJ._update_vtx_data(zone, vtx_to_remove, comm)

  assert (PT.get_value(MT.getDistribution(zone, 'Vertex')) == expected_distri).all()
  assert (PT.get_node_from_name(zone, 'CoordinateX')[1] == expected_cx).all()
  assert (PT.get_node_from_name(zone, 'Sol')[1] == expected_sol).all()

@pytest_parallel.mark.parallel(2)
def test_merge_intrazone_jn(comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], comm)
  zone = PT.get_all_Zone_t(tree)[0]
  PT.rm_children_from_label(zone, 'ZoneBC_t')
  #Create jns
  zgc = PT.new_ZoneGridConnectivity('ZoneGC', parent=zone)
  pl = np.array([[15,16]], pdm_dtype) if comm.Get_rank() == 0 else np.array([[]], pdm_dtype)
  pld = np.array([[23,24]], pdm_dtype) if comm.Get_rank() == 0 else np.array([[]], pdm_dtype)
  distri = np.array([0,2,2], pdm_dtype) if comm.Get_rank() == 0 else np.array([2,2,2], pdm_dtype)
  jn = PT.new_GridConnectivity('matchA', 'zone', type='Abutting1to1', loc='FaceCenter', parent=zgc)
  PT.new_IndexArray('PointList', pl, parent=jn)
  PT.new_IndexArray('PointListDonor', pld, parent=jn)
  MT.newDistribution({'Index' : distri}, jn)
  jn = PT.new_GridConnectivity('matchB', 'zone', 'Abutting1to1', loc='FaceCenter', parent=zgc)
  PT.new_IndexArray('PointList', pld, parent=jn)
  PT.new_IndexArray('PointListDonor', pl, parent=jn)
  MT.newDistribution({'Index' : distri}, jn)
  #Other jns to ensure they are not merge
  pl = np.array([[13,14]], pdm_dtype) if comm.Get_rank() == 1 else np.array([[]], pdm_dtype)
  pld = np.array([[21,22]], pdm_dtype) if comm.Get_rank() == 1 else np.array([[]], pdm_dtype)
  distri = np.array([0,2,2], pdm_dtype) if comm.Get_rank() == 1 else np.array([2,2,2], pdm_dtype)
  jn = PT.new_GridConnectivity('matchC', 'zone', 'Abutting1to1', loc='FaceCenter', parent=zgc)
  PT.new_IndexArray('PointList', pl, parent=jn)
  PT.new_IndexArray('PointListDonor', pld, parent=jn)
  MT.newDistribution({'Index' : distri}, jn)
  jn = PT.new_GridConnectivity('matchD', 'zone', 'Abutting1to1', loc='FaceCenter', parent=zgc)
  PT.new_IndexArray('PointList', pld, parent=jn)
  PT.new_IndexArray('PointListDonor', pl, parent=jn)
  MT.newDistribution({'Index' : distri}, jn)

  jn_pathes = ('Base/zone/ZoneGC/matchC', 'Base/zone/ZoneGC/matchD')
  MJ.merge_intrazone_jn(tree, jn_pathes, comm)

  assert PT.get_node_from_name(zone, 'matchA') is not None
  assert PT.get_node_from_name(zone, 'matchB') is not None
  assert PT.get_node_from_name(zone, 'matchC') is None
  assert PT.get_node_from_name(zone, 'matchD') is None

  assert (PT.get_node_from_path(zone, 'NGonElements/ElementRange')[1] == [1,34]).all()
  assert (PT.get_node_from_path(zone, 'ZoneGC/matchA/PointList')[1] ==
          PT.get_node_from_path(zone, 'ZoneGC/matchB/PointListDonor')[1]).all()
  assert (PT.get_node_from_path(zone, 'ZoneGC/matchB/PointList')[1] ==
          PT.get_node_from_path(zone, 'ZoneGC/matchA/PointListDonor')[1]).all()
