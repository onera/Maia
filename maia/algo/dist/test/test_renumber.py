import pytest
import pytest_parallel

import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils import par_utils
from maia.utils import test_utils as TU
from maia.utils import vstride as vs

from maia.algo.dist import renumber as RENUM

from maia import npy_pdm_gnum_dtype as pdm_dtype

def test_collect_shifted_pl_one():
  subset = PT.new_GridConnectivity(point_list=[[1,3,5,7]], point_range_donor=[[101,120]])
  MT.new_Distribution({'Index' : np.array([10,14,20])}, subset)
  assert (RENUM._collect_shifted_pl_one(subset, shift=-1) == [0,2,4,6]).all()
  assert (RENUM._collect_shifted_pl_one(subset, shift=-1, donor=True) == [110,111,112,113]).all()

  subset = PT.new_BC()
  with pytest.raises(RuntimeError):
    RENUM._collect_shifted_pl_one(subset)

def test_update_pl_one():
  v = 42 # Unused, since initial value will be erased
  subset = PT.new_GridConnectivity(point_list=[[v,v,v,v]], point_range_donor=[[v,v]])
  RENUM._update_pl_one(subset, np.array([[0,1,2,3]]), shift=1)
  assert (PT.get_child_from_name(subset, 'PointList')[1] == [[1,2,3,4]]).all()
  RENUM._update_pl_one(subset, np.array([[10,11,12,13]]), donor=True, shift=1)
  assert (PT.get_child_from_name(subset, 'PointListDonor')[1] == [[11,12,13,14]]).all()

def test_subdistri():
  out = RENUM.subdistri(np.array([0,10,25,40], np.int32), 0, 10)
  assert np.array_equal(out, [0,10,10,10]) and out.dtype == np.int32
  out = RENUM.subdistri(np.array([0,10,25,40], np.int64), 15, 30)
  assert np.array_equal(out, [0,0,10,15]) and out.dtype == np.int64
  assert (RENUM.subdistri(np.array([0,10]), 5, 8) == [0,3]).all()  # Seq
  assert (RENUM.subdistri(np.array([0,10,25,40]), 50, 100) == [0,0,0,0]).all()

def test_local_bounds():
  assert RENUM.local_bounds(np.array([25, 50, 100]), 0,0) == (0,0)
  assert RENUM.local_bounds(np.array([25, 50, 100]), 4,23) == (0,0)
  assert RENUM.local_bounds(np.array([25, 50, 100]), 50,54) == (25,25)
  assert RENUM.local_bounds(np.array([25, 50, 100]), 400,500) == (25,25)
  assert RENUM.local_bounds(np.array([25, 50, 100]), 0,100) == (0,25)
  assert RENUM.local_bounds(np.array([25, 50, 100]), 0,500) == (0,25)
  assert RENUM.local_bounds(np.array([25, 50, 100]), 20,40) == (0,15)
  assert RENUM.local_bounds(np.array([25, 50, 100]), 40,60) == (15,25)
  assert RENUM.local_bounds(np.array([25, 25, 100]), 10,40) == (0,0)

@pytest_parallel.mark.parallel(2)
def test_renumber_vertices(comm):
  tree1 = maia.factory.generate_dist_block([3,2], 'S', comm, origin=[0,0])
  tree2 = maia.factory.generate_dist_block([3,2], 'S', comm, origin=[1,0])
  zone1 = PT.get_all_Zone_t(tree1)[0]
  zone2 = PT.get_all_Zone_t(tree2)[0]
  ztype = PT.get_np_value(zone1).dtype
  PT.set_name(zone1, 'Left')
  PT.set_name(zone2, 'Right')
  PT.rm_nodes_from_name(zone1, 'Xmax')
  PT.rm_nodes_from_name(zone2, 'Xmin')
  # Add jns
  pr  = np.array([[3,3], [1,2]], ztype)
  prd = np.array([[1,1], [1,2]], ztype)
  distri = par_utils.uniform_distribution(1*2, comm)
  jn = PT.new_GridConnectivity1to1(donor_name='Right', point_range=pr, point_range_donor=prd,
                                   transform=[1,2],
                                   parent=PT.new_ZoneGridConnectivity(parent=zone1))
  MT.new_Distribution({'Index' : distri}, jn)
  jn = PT.new_GridConnectivity1to1(donor_name='Base/Left', point_range=prd, point_range_donor=pr,
                                   transform=[1,2],
                                   parent=PT.new_ZoneGridConnectivity(parent=zone2))
  MT.new_Distribution({'Index' : distri}, jn)
  tree = PT.union(tree1, tree2)

  maia.algo.dist.convert_s_to_u(tree, 'Standard', comm)
  
  # Add sol
  vtx_distri = MT.Zone.vtx_distribution(PT.find_node_from_name(tree, 'Left'))
  PT.new_DiscreteData(loc='Vertex',
                      fields={'Id' : np.array([1,2,3,4,5,6])[vtx_distri[0]:vtx_distri[1]]},
                      parent=PT.find_node_from_name(tree, 'Left'))

  new_vtx_id = np.array([5,4,3,2,1,0], ztype)[vtx_distri[0]:vtx_distri[1]] # Choose a new numbering
  RENUM.renumber_vertices(tree, 'Base/Left', new_vtx_id, comm)

  # For Right zone, only GC should be modified
  expt_zone2 = PT.deep_copy(PT.find_node_from_name(tree, 'Right'))
  gc = PT.find_node_from_name(expt_zone2, 'GC')
  distri = MT.Subset.distribution(gc)
  pld = PT.get_np_value(PT.find_child_from_name(gc, 'PointListDonor'))
  pld[0,:] = np.array([4,1])[distri[0]:distri[1]]

  # For Left zone, vertices should be reordered
  zt = 'I8' if PT.get_np_value(expt_zone2).dtype == np.int64 else 'I4'
  expt_zone1_f = PT.yaml.to_node(f"""
  Left Zone_t {zt} [[6, 2, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [1.0, 0.5, 0.0, 1.0, 0.5, 0.0]:
      CoordinateY DataArray_t R8 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]:
    ZoneBC ZoneBC_t:
      Xmin BC_t 'Null':
        GridLocation GridLocation_t 'Vertex':
        PointList IndexArray_t {zt} [[6, 3]]:
      Ymin BC_t 'Null':
        GridLocation GridLocation_t 'Vertex':
        PointList IndexArray_t {zt} [[6, 5, 4]]:
      Ymax BC_t 'Null':
        GridLocation GridLocation_t 'Vertex':
        PointList IndexArray_t {zt} [[3, 2, 1]]:
    ZoneGridConnectivity ZoneGridConnectivity_t:
      GC GridConnectivity_t 'Right':
        GridConnectivityType GridConnectivityType_t 'Abutting1to1':
        GridLocation GridLocation_t 'Vertex':
        GridConnectivityDonorName Descriptor_t 'GC':
        PointListDonor IndexArray_t {zt} [[1, 4]]:
        PointList IndexArray_t {zt} [[4, 1]]:
    QUAD_4 Elements_t I4 [7, 0]:
      ElementRange IndexRange_t {zt} [7, 8]:
      ElementConnectivity DataArray_t {zt} [6, 5, 2, 3, 5, 4, 1, 2]:
    BAR_2 Elements_t I4 [3, 0]:
      ElementRange IndexRange_t {zt} [1, 6]:
      ElementConnectivity DataArray_t {zt} [3, 6, 4, 1, 6, 5, 5, 4, 2, 3, 1, 2]:
    DiscreteData DiscreteData_t:
      GridLocation GridLocation_t "Vertex":
      Id DataArray_t I8 [6,5,4,3,2,1]:
  """)
  expt_zone1 = maia.factory.full_to_dist_tree(expt_zone1_f, comm)
  # RM Distri/ElementConnectivity for comparaison
  for elt in PT.get_nodes_from_label(expt_zone1, 'Elements_t'):
    PT.rm_node_from_path(elt, ':CGNS#Distribution/ElementConnectivity')

  assert PT.is_same_tree(PT.find_node_from_name(tree, 'Left'), expt_zone1)
  assert PT.is_same_tree(PT.find_node_from_name(tree, 'Right'), expt_zone2)

@pytest_parallel.mark.parallel(1)
def test_renumber_edges_1d(comm):
  tree = maia.factory.generate_dist_block(5, 'BAR_2', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  ztype = PT.get_np_value(zone).dtype
  PT.new_FlowSolution(loc='CellCenter', fields={'ID': [1,2,3,4]}, parent=zone)

  new_edge_id = np.array([4,3,2,1], ztype) - 1
  RENUM.renumber_edges(tree, 'Base/Line', new_edge_id, comm)

  assert (PT.find_node_from_name(zone, 'ElementConnectivity')[1] == [4,5, 3,4, 2,3, 1,2]).all()
  assert (PT.find_node_from_name(zone, 'ID')[1] == [4,3,2,1]).all()


@pytest_parallel.mark.parallel(2)
def test_renumber_edges_2d_elt(comm):
  tree = maia.factory.generate_dist_block(4, 'QUAD_4', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  ztype = PT.get_np_value(zone).dtype
  # Use several edges sections in this test
  PT.rm_nodes_from_predicate(zone, PT.pred.is_element_of_type('BAR_2'))
  er_start = 10
  # Ini order: Ymin, Ymax, Xmin, Xmax
  bar_ecs = [[1,2, 2,3, 3,4], [14,13, 15,14, 16,15], [5,1, 9,5, 13,9, 4,8, 8,12, 12,16]] 
  distris = [[0,3,3], [0,0,3], [0,2,6]]
  for i, (ec,distri) in enumerate(zip(bar_ecs, distris)):
    n_bar_tot = len(ec) // 2
    _distri = par_utils.full_to_partial_distribution(np.array(distri, pdm_dtype), comm)
    elt = PT.new_Elements(f'BAR_{i}', 'BAR_2',
                          erange=np.array([er_start, er_start+n_bar_tot-1], ztype),
                          econn=np.array(ec, ztype)[2*_distri[0]:2*_distri[1]],
                          parent=zone)
    MT.new_Distribution({'Element' : _distri}, elt)
    er_start += n_bar_tot

  # Reoder : Xmin, Xmax, Ymin, Ymax
  new_edge_id = np.array([7,8,9, 10,11,12, 1,2,3,4,5,6], ztype) - 1
  # Distrib it:
  if comm.rank == 0:
    new_edge_id = new_edge_id[0:8]
  else:
    new_edge_id = new_edge_id[8:12]

  RENUM.renumber_edges(tree, 'Base/zone', new_edge_id, comm)
  
  if comm.rank == 0:
    expt_ec = np.array([5,1, 9,5, 13,9, 4,8, 8,12, 12,16])
    expt_distri = np.array([0, 6, 12])
  else:
    expt_ec = np.array([1,2, 2,3, 3,4, 14,13, 15,14, 16,15])
    expt_distri = np.array([6, 12, 12])

  bar = PT.find_node_from_name(tree, 'BAR_2')
  assert (PT.Element.Range(bar) == [10,21]).all()
  assert (PT.find_child_from_name(bar, 'ElementConnectivity')[1] == expt_ec).all()
  assert (MT.Element.distribution(bar) == expt_distri).all()



@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("remove", ['', 'ParentElements', 'NGonElements'])
def test_renumber_faces_2d_poly(remove, comm):
  tree = maia.factory.generate_dist_block(4, 'QUAD_4', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  ztype = PT.get_np_value(zone).dtype
  
  PT.rm_nodes_from_name(tree, remove)

  # Array poorly distributed
  new_id = np.array([1,2,3,7,8,9,4,5,6], ztype) - 1 if comm.rank == 0 else np.empty(0, ztype)
  RENUM.renumber_faces(tree, 'Base/zone', new_id, comm)

  if comm.rank == 0:
    expt_face_vtx = vs.array([[1,2,6,5], [6,2,3,7], [7,3,4,8], [9,10,14,13], [14,10,11,15]])
    expt_edge_face = [[25, 0], [26, 0], [25, 0], [27, 0], [25,26], [26,27], [25,31],
                      [27, 0], [26,32], [31, 0], [27,33], [31,32]]
  else:
    expt_face_vtx = vs.array([[15,11,12,16], [5,6,10,9], [10,6,7,11], [11,7,8,12]])
    expt_edge_face = [[32,33], [31,28], [33, 0], [32,29], [28, 0], [33,30], [28,29],
                      [29,30], [28, 0], [30, 0], [29, 0], [30, 0]]

  if remove != 'NGonElements':
    ng = PT.Zone.NGonNode(zone)
    face_vtx = MT.Element.connectivity(ng)
    assert vs.array_equal(face_vtx, expt_face_vtx)
  if remove != 'ParentElements':
    pe = PT.find_node_from_name(zone, 'ParentElements')
    assert (PT.get_np_value(pe) == expt_edge_face).all()

@pytest_parallel.mark.parallel(2)
def test_renumber_faces_3d_poly(comm):
  tree = maia.factory.generate_dist_block([3,2,2], 'Poly', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  maia.algo.pe_to_nface(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  ztype = PT.get_np_value(zone).dtype

  # Just permute Xmin (1) and Xmax (3) faces
  new_id = np.array([3,2,1,4,5,6,7,8,9,10,11], ztype) - 1
  new_id = new_id[:6] if comm.rank == 0 else new_id[6:]

  expt_zone = PT.deep_copy(zone)
  ng = PT.Zone.NGonNode(zone)
  nf = PT.Zone.NFaceNode(zone)
  ng_ec = PT.get_np_value(PT.find_child_from_name(ng, 'ElementConnectivity'))
  ng_pe = PT.get_np_value(PT.find_child_from_name(ng, 'ParentElements'))
  nf_ec = PT.get_np_value(PT.find_child_from_name(nf, 'ElementConnectivity'))
  xm_pl = PT.get_np_value(PT.get_node_from_names(zone, ['Xmin', 'PointList']))
  xM_pl = PT.get_np_value(PT.get_node_from_names(zone, ['Xmax', 'PointList']))
  if comm.rank == 0:
    ng_ec[0:4]  = [3,6,12,9]
    ng_ec[8:12] = [1,7,10,4] # Swap faces 1 & 3
    ng_pe[0] = [13, 0]
    ng_pe[2] = [12, 0] # Swap faces 1 & 3
    nf_ec[0] = 3 # was face 1 before
    xm_pl[0][0] = 3
    xM_pl[0][0] = 1
  if comm.rank == 1:
    nf_ec[1] = 1 # was face 3 before
 
  RENUM.renumber_faces(tree, 'Base/zone', new_id, comm)

  assert PT.is_same_tree(zone, expt_zone)


@pytest_parallel.mark.parallel(1)
def test_renumber_faces_3d_elt(comm):
  tree = maia.io.file_to_dist_tree(TU.mesh_dir / 'hex_prism_pyra_tet.yaml', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  new_id = np.array([1,2,3,6,5,4, 7,8,9,12,11,10]) - 1
  RENUM.renumber_faces(tree, 'Base/Zone', new_id, comm)

  tri = PT.get_node_from_name(zone, 'TRI_3')
  quad = PT.get_node_from_name(zone, 'QUAD_4')
  assert (PT.find_child_from_name(tri, 'ElementConnectivity')[1] == 
          [6,11,9, 8,10,11, 6,7,11, 9,11,10, 2,5,3, 7,8,11]).all()
  assert (PT.find_child_from_name(quad, 'ElementConnectivity')[1] == 
           [1,6,9,4, 3,5,10,8, 1,2,7,6, 1,4,5,2, 4,9,10,5, 2,3,8,7]).all()

  assert (PT.get_node_from_path(zone, 'ZoneBC/Ymin/PointList')[1] ==
          [[3,6,9,12]]).all()
  assert (PT.get_node_from_path(zone, 'ZoneBC/Zmax/PointList')[1] ==
          [[10]]).all()


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('pe_only', [False, True])
def test_renumber_cells_poly(pe_only, comm):
  tree = maia.factory.generate_dist_block([4,2,2], 'Poly', comm)
  zone = PT.find_node_from_label(tree, 'Zone_t')
  ztype = PT.get_np_value(zone).dtype

  if comm.rank == 0:
    PT.new_FlowSolution('Sol', loc='CellCenter', fields={'Id': [1,2]}, parent=zone)
  else:
    PT.new_FlowSolution('Sol', loc='CellCenter', fields={'Id': [3]}, parent=zone)

  if not pe_only:
    maia.algo.pe_to_nface(tree, comm)

  new_id = np.array([3,2,1], ztype) - 1 if comm.rank == 1 else np.empty(0, ztype)
  RENUM.renumber_cells(tree, 'Base/zone', new_id, comm)

  if comm.rank == 0:
    expt_cell_face = vs.array([[-3,4,7,10,13,16], [-2,3,6,9,12,15]])
    expt_face_cell = [[19, 0], [19,18], [18,17], [17, 0], [19, 0], [18, 0], [17, 0], [19, 0]]
    expt_sol = [3,2]
  else:
    expt_cell_face = vs.array([[1,2,5,8,11,14]])
    expt_face_cell = [[18, 0], [17, 0], [19, 0], [18, 0], [17, 0], [19, 0], [18, 0], [17, 0]]
    expt_sol = [1]

  ng = PT.Zone.NGonNode(zone)
  assert (PT.find_child_from_name(ng, 'ParentElements')[1] == expt_face_cell).all()
  if not pe_only:
    nf = PT.Zone.NFaceNode(zone)
    assert vs.array_equal(MT.Element.connectivity(nf), expt_cell_face)
  assert (PT.find_node_from_name(zone, 'Id')[1] == expt_sol).all()


@pytest_parallel.mark.parallel(2)
def test_renumber_cells_elt(comm):
  tree = maia.io.file_to_dist_tree(TU.mesh_dir / 'hex_2_prism_2.yaml', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if comm.rank == 0:
    PT.new_FlowSolution('Sol', loc='CellCenter', fields={'Id': [1,2]}, parent=zone)
    new_id = np.array([2,1]) - 1
  else:
    PT.new_FlowSolution('Sol', loc='CellCenter', fields={'Id': [3,4]}, parent=zone)
    new_id = np.array([4,3]) - 1

  RENUM.renumber_cells(tree, 'Base/Zone', new_id, comm)

  if comm.rank == 0:
    expt_hexa = [6,7,10,9,11,12,15,14]
    expt_prism = [7,8,10,12,13,15]
    expt_sol = [2,1]
  else:
    expt_hexa = [1,2,5,4,6,7,10,9]
    expt_prism = [2,3,5,7,8,10]
    expt_sol = [4,3]

  hexa  = PT.find_node_from_name(zone, 'HEXA_8')
  prism = PT.find_node_from_name(zone, 'PENTA_6')
  assert (PT.find_child_from_name(hexa, 'ElementConnectivity')[1] == expt_hexa).all()
  assert (PT.find_child_from_name(prism, 'ElementConnectivity')[1] == expt_prism).all()
  assert (PT.find_node_from_name(zone, 'Id')[1] == expt_sol).all()

@pytest_parallel.mark.parallel(2)
def test_renumber_cells_fail(comm):
  tree = maia.io.file_to_dist_tree(TU.mesh_dir / 'hex_2_prism_2.yaml', comm)

  with pytest.raises(ValueError):
    # Wrong because this numbering mixes HEXA and PRISM sections
    new_id = np.array([1,4,3,2]) - 1
    new_id = new_id[:3] if comm.rank == 0 else new_id[3:]

    RENUM.renumber_cells(tree, 'Base/Zone', new_id, comm)

  with pytest.raises(ValueError):
    # Technically this should be OK because elements are not mixed
    # (order of sections is permuted), but with currrent implem this
    # is an error. This test is here to remember we may allow this one day
    new_id = np.array([3,4,1,2]) - 1
    new_id = new_id[:3] if comm.rank == 0 else new_id[3:]

    RENUM.renumber_cells(tree, 'Base/Zone', new_id, comm)