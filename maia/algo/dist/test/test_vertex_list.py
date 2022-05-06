import numpy as np
import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia           import npy_pdm_gnum_dtype as pdm_dtype
from maia.factory   import dcube_generator
from maia.algo.dist import vertex_list as VL
from maia.utils     import par_utils

# TODO replace dcube_generate by handwritten mesh (4_cubes?)

@mark_mpi_test([1,3])
def test_face_ids_to_vtx_ids(sub_comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], sub_comm)
  ngon = I.getNodeFromName(tree, "NGonElements")

  offset, face_vtx   = VL.face_ids_to_vtx_ids(np.array([3,6,2]), ngon, sub_comm)
  assert (offset == np.arange(0,(3+1)*4,4)).all()
  assert (face_vtx == [5,8,7,4, 11,14,15,12, 3,6,5,2]).all()

  offset, face_vtx_d = VL.face_ids_to_vtx_ids(np.array([1,4,5]), ngon, sub_comm)
  assert (offset == np.arange(0,(3+1)*4,4)).all()
  assert (face_vtx_d == [2,5,4,1, 6,9,8,5, 10,13,14,11]).all()

@mark_mpi_test(2)
def test_filter_vtx_coordinates(sub_comm):
  empty = np.empty(0, int)
  tree = dcube_generator.dcube_generate(5,1.,[0,0,0], sub_comm)
  vtx_coords = I.getNodeFromType(tree, 'GridCoordinates_t')
  vtx_distri   = I.getVal(MT.getDistribution(I.getZones(tree)[0], 'Vertex'))
  if sub_comm.Get_rank() == 1:
    requested_vtx_ids = [2,6,7,106,3,103,107,102]
    expected_vtx_coords = np.array([[0.25, 0., 0.], [0., 0.25, 0.], [0.25, 0.25, 0.], [0., 0.25, 1.],
                                    [0.5, 0., 0.], [0.5, 0., 1.], [0.25, 0.25, 1.], [0.25, 0., 1.]])
  else:
    requested_vtx_ids = empty
    expected_vtx_coords = np.empty((0,3), dtype=np.float64)

  received_coords = VL.filter_vtx_coordinates(vtx_coords, vtx_distri, requested_vtx_ids,  sub_comm)
  assert (received_coords == expected_vtx_coords).all()

@mark_mpi_test(2)
def test_get_extended_pl(sub_comm):
  tree = dcube_generator.dcube_generate(3,1.,[0,0,0], sub_comm)
  ngon = I.getNodeFromName(tree, "NGonElements")
  if sub_comm.Get_rank() == 0:
    pl   = np.array([1,2,3])
    pl_d = np.array([9,10,11])
    pl_idx = np.array([0,4,8,12])
    pl_vtx = np.array([2,5,4,1, 3,6,5,2, 5,8,7,4])
    skip_f = np.array([True, True, True])
  else:
    pl   = np.array([4])
    pl_d = np.array([12])
    pl_idx = np.array([0,4])
    pl_vtx = np.array([6,9,8,5])
    skip_f = np.array([False])
  ext_pl, ext_pl_d = VL.get_extended_pl(pl, pl_d, pl_idx, pl_vtx, sub_comm)
  assert (ext_pl   == [1,2,3,4]).all()
  assert (ext_pl_d == [9,10,11,12]).all()

  ext_pl, ext_pl_d = VL.get_extended_pl(pl, pl_d, pl_idx, pl_vtx, sub_comm, skip_f)
  if sub_comm.Get_rank() == 0:
    assert len(ext_pl) == len(ext_pl_d) == 0
  else:
    assert (ext_pl   == [1,2,3,4]).all()
    assert (ext_pl_d == [9,10,11,12]).all()

def test_search_by_intersection():
  empty = np.empty(0, int)
  plv, plv_opp, face_is_treated = VL._search_by_intersection(np.array([0]), empty, empty)
  assert (plv == empty).all()
  assert (plv_opp == empty).all()
  assert (face_is_treated == empty).all()

  #Some examples from cube3, top/down jns
  #Single face can not be treated
  plv, plv_opp, face_is_treated = \
    VL._search_by_intersection( \
      [0          ,4], \
      [ 5, 8, 7, 4], \
      [22,25,26,23] \
    )
  assert (plv == [4,5,7,8]).all()
  assert (plv_opp == [0,0,0,0]).all()
  assert (face_is_treated == [False]).all()

  #Example from Cube4 : solo face will not be treated
  plv, plv_opp, face_is_treated = \
    VL._search_by_intersection( \
      [0          , 4          , 8           , 12], \
      [ 6,10, 9, 5, 12,16,15,11,  3, 7, 6, 2], \
      [37,41,42,38, 43,47,48,44, 34,38,39,35] \
    )

  assert (plv     == [ 2, 3, 5, 6, 7, 9,10,11,12,15,16]).all()
  assert (plv_opp == [34,35,37,38,39,41,42, 0, 0, 0, 0]).all()
  assert (face_is_treated == [True, False, True]).all()

  #Here is a crazy exemple where third face can not be treated on first pass,
  # but is the treated thanks to already treated faces
  plv, plv_opp, face_is_treated = \
    VL._search_by_intersection( \
      [0          , 4             , 9              ,14], \
      [ 2, 4, 3, 1, 10, 9, 6, 8, 7,  5, 6, 3, 4, 7], \
      [12,11,13,14, 19,20,17,18,16, 13,16,15,17,14] \
    )
  assert (plv == np.arange(1,10+1)).all()
  assert (plv_opp == np.arange(11,20+1)).all()
  assert (face_is_treated == [True, True, True]).all()

@mark_mpi_test(3)
def test_search_with_geometry(sub_comm):
  empty = np.empty(0, int)
  tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
  zone = I.getZones(tree)[0]
  grid_prop = I.newGridConnectivityProperty()
  perio = I.newPeriodic(rotationCenter=[0.,0.,0.], rotationAngle=[0.,0.,0.], translation=[0.,0.,1.], parent=grid_prop)

  if sub_comm.Get_rank() == 0:
    pl_face_vtx_idx = [0,4]
    pl_face_vtx = [2,6,7,3]
    pld_face_vtx = [51,55,54,50]
  elif sub_comm.Get_rank() == 2:
    pl_face_vtx_idx = [0,4,8]
    pl_face_vtx = [11,15,16,12, 13,14,10,9]
    pld_face_vtx = [64,63,59,60, 61,57,58,62]
  else:
    pl_face_vtx_idx = [0]
    pl_face_vtx = empty
    pld_face_vtx = empty

  plv, plvd = VL._search_with_geometry(zone, zone, grid_prop, pl_face_vtx_idx, pl_face_vtx, pld_face_vtx, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert (plv  == [2,3,6,7]).all()
    assert (plvd == [50,51,54,55]).all()
  elif sub_comm.Get_rank() == 2:
    assert (plv  == [9,10,11,12,13,14,15,16]).all()
    assert (plvd == [57,58,59,60,61,62,63,64]).all()
  else:
    assert (plv.size == plvd.size == 0)

class Test_generate_jn_vertex_list():
  @mark_mpi_test([1,2,4])
  def test_single_zone_topo(self, sub_comm):
    #With this configuration, we have locally (per rank) isolated faces when np=4
    tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
    zone = I.getZones(tree)[0]
    I._rmNodesByType(zone, 'ZoneBC_t')
    #Create fake jn
    zgc = I.newZoneGridConnectivity(parent=zone)
    gcA = I.newGridConnectivity('matchA', I.getName(zone), 'Abutting1to1', zgc)
    full_pl     = np.array([1,2,3,4,5,6,7,8,9])
    full_pl_opp = np.array([28,29,30,31,32,33,34,35,36])
    distri_pl   = par_utils.uniform_distribution(9, sub_comm)
    I.newGridLocation('FaceCenter', gcA)
    I.newPointList('PointList'     , full_pl    [distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
    I.newPointList('PointListDonor', full_pl_opp[distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
    MT.newDistribution({'Index' : distri_pl}, gcA)

    gc_path = "Base/zone/ZoneGridConnectivity/matchA"
    pl_vtx, pl_vtx_opp, distri_jn_vtx = VL.generate_jn_vertex_list(tree, gc_path, sub_comm)

    expected_full_pl_vtx     = np.arange(1,16+1)
    expected_full_pl_vtx_opp = np.arange(49,64+1)
    assert (distri_jn_vtx == par_utils.uniform_distribution(16, sub_comm)).all()
    assert (pl_vtx     == expected_full_pl_vtx    [distri_jn_vtx[0]:distri_jn_vtx[1]]).all()
    assert (pl_vtx_opp == expected_full_pl_vtx_opp[distri_jn_vtx[0]:distri_jn_vtx[1]]).all()

  @mark_mpi_test([2,4])
  def test_multi_zone_topo(self, sub_comm):
    tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
    zone = I.getZones(tree)[0]
    zone[0] = 'zoneA'
    I._rmNodesByType(zone, 'ZoneBC_t')

    #Create other zone
    tree2 = dcube_generator.dcube_generate(4,1.,[1,0,0], sub_comm)
    zoneB = I.getZones(tree2)[0]
    zoneB[0] = 'zoneB'
    I._rmNodesByType(zoneB, 'ZoneBC_t')
    I._addChild(I.getNodeFromName(tree, 'Base'), zoneB)

    #Create fake jn
    zgc = I.newZoneGridConnectivity(parent=zone)
    gcA = I.newGridConnectivity('matchA', 'zoneB', 'Abutting1to1', zgc)
    full_pl     = np.array([64,65,66,67,68,69,70,71,72]) #xmax
    full_pl_opp = np.array([37,38,39,40,41,42,43,44,45]) #xmin
    distri_pl   = par_utils.uniform_distribution(9, sub_comm)
    I.newGridLocation('FaceCenter', gcA)
    I.newPointList('PointList'     , full_pl    [distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
    I.newPointList('PointListDonor', full_pl_opp[distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
    MT.newDistribution({'Index' : distri_pl}, gcA)

    gc_path = "Base/zoneA/ZoneGridConnectivity/matchA"
    pl_vtx, pl_vtx_opp, distri_jn_vtx = VL.generate_jn_vertex_list(tree, gc_path, sub_comm)

    expected_full_pl_vtx     = np.arange(4, 64+1, 4)
    expected_full_pl_vtx_opp = np.arange(1, 64+1, 4)
    assert (distri_jn_vtx == par_utils.uniform_distribution(16, sub_comm)).all()
    assert (pl_vtx     == expected_full_pl_vtx    [distri_jn_vtx[0]:distri_jn_vtx[1]]).all()
    assert (pl_vtx_opp == expected_full_pl_vtx_opp[distri_jn_vtx[0]:distri_jn_vtx[1]]).all()

  @mark_mpi_test(3)
  def test_single_zone_all(self, sub_comm):
    #Faces that should be treated during phase 1,2 and 3 for each proc :
    # P0 : (2,0,1)   P1 : (0,1,0)   P2 : (0,1,1)
    tree = dcube_generator.dcube_generate(5,1.,[0,0,0], sub_comm)
    zone = I.getZones(tree)[0]
    I._rmNodesByType(zone, 'ZoneBC_t')
    zgc = I.newZoneGridConnectivity(parent=zone)
    gcA = I.newGridConnectivity('matchA', I.getName(zone), 'Abutting1to1', zgc)
    I.newGridLocation('FaceCenter', gcA)
    if sub_comm.Get_rank() == 0:
      pl_distri = [0,3,6]
      expt_jn_distri = [0, 7, 21]
    elif sub_comm.Get_rank() == 1:
      pl_distri = [3,4,6]
      expt_jn_distri = [7, 14, 21]
    elif sub_comm.Get_rank() == 2:
      pl_distri = [4,6,6]
      expt_jn_distri = [14, 21, 21]

    I.newPointList('PointList'     , (np.array([1,2,4,11,13,16])   [pl_distri[0]:pl_distri[1]]).reshape(1,-1), gcA)
    I.newPointList('PointListDonor', (np.array([65,66,68,75,77,80])[pl_distri[0]:pl_distri[1]]).reshape(1,-1), gcA)
    MT.newDistribution({'Index' : pl_distri}, gcA)

    gc_path = "Base/zone/ZoneGridConnectivity/matchA"
    pl_vtx, pld_vtx, distri_jn_vtx = VL.generate_jn_vertex_list(tree, gc_path, sub_comm)

    expected_full_pl_vtx  = np.array([1,2,3,4,5,6,7,8,9,10,13,14,16,17,18,19,20,21,22,24,25])
    expected_full_pld_vtx = expected_full_pl_vtx + 100
    assert (distri_jn_vtx == expt_jn_distri).all()
    assert (pl_vtx  == expected_full_pl_vtx [distri_jn_vtx[0]:distri_jn_vtx[1]]).all()
    assert (pld_vtx == expected_full_pld_vtx[distri_jn_vtx[0]:distri_jn_vtx[1]]).all()

  @mark_mpi_test(2)
  def test_multi_zone_geo(self, sub_comm):
    tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
    zone = I.getZones(tree)[0]
    zone[0] = 'zoneA'
    I._rmNodesByType(zone, 'ZoneBC_t')

    #Create other zone
    tree2 = dcube_generator.dcube_generate(4,1.,[1,0,0], sub_comm)
    zoneB = I.getZones(tree2)[0]
    zoneB[0] = 'zoneB'
    I._rmNodesByType(zoneB, 'ZoneBC_t')
    I._addChild(I.getNodeFromName(tree, 'Base'), zoneB)

    #Create fake jn such that we have only isolated faces
    zgc = I.newZoneGridConnectivity(parent=zone)
    gcA = I.newGridConnectivity('matchA', 'zoneB', 'Abutting1to1', zgc)
    full_pl     = np.array([64,66,71]) #xmax
    full_pl_opp = np.array([37,39,44]) #xmin
    distri_pl   = par_utils.uniform_distribution(3, sub_comm)
    I.newGridLocation('FaceCenter', gcA)
    I.newPointList('PointList'     , full_pl    [distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
    I.newPointList('PointListDonor', full_pl_opp[distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
    MT.newDistribution({'Index' : distri_pl}, gcA)

    gc_path = "Base/zoneA/ZoneGridConnectivity/matchA"
    pl_vtx, pl_vtx_opp, distri_jn_vtx = VL.generate_jn_vertex_list(tree, gc_path, sub_comm)

    expected_dist = [0,6,12] if sub_comm.Get_rank() == 0 else [6,12,12]
    expected_full_pl_vtx     = [4,8,12,16,20,24,28,32,40,44,56,60]
    expected_full_pl_vtx_opp = [1,5,9,13,17,21,25,29,37,41,53,57]
    assert (distri_jn_vtx == expected_dist).all()
    assert (pl_vtx     == expected_full_pl_vtx    [distri_jn_vtx[0]:distri_jn_vtx[1]]).all()
    assert (pl_vtx_opp == expected_full_pl_vtx_opp[distri_jn_vtx[0]:distri_jn_vtx[1]]).all()

@mark_mpi_test(3)
@pytest.mark.parametrize("have_isolated_faces", [False, True])
def test_generate_jns_vertex_list(sub_comm, have_isolated_faces):
  #For this test, we reuse previous test case, but with 2 jns,
  # and do not assert on values
  tree = dcube_generator.dcube_generate(4,1.,[0,0,0], sub_comm)
  zoneA = I.getZones(tree)[0]
  zoneA[0] = 'zoneA'
  I._rmNodesByType(zoneA, 'ZoneBC_t')

  #Create other zone
  tree2 = dcube_generator.dcube_generate(4,1.,[1,0,0], sub_comm)
  zoneB = I.getZones(tree2)[0]
  zoneB[0] = 'zoneB'
  I._rmNodesByType(zoneB, 'ZoneBC_t')
  I._addChild(I.getNodeFromName(tree, 'Base'), zoneB)

  #Create fake jns
  zgc = I.newZoneGridConnectivity(parent=zoneA)
  gcA = I.newGridConnectivity('matchA', 'zoneB', 'Abutting1to1', zgc)
  full_pl     = np.array([64,65,66,67,68,69,70,71,72], pdm_dtype) #xmax
  full_pl_opp = np.array([37,38,39,40,41,42,43,44,45], pdm_dtype) #xmin
  distri_pl   = par_utils.uniform_distribution(9, sub_comm)
  I.newGridLocation('FaceCenter', gcA)
  I.newPointList('PointList'     , full_pl    [distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
  I.newPointList('PointListDonor', full_pl_opp[distri_pl[0]:distri_pl[1]].reshape(1,-1), gcA)
  MT.newDistribution({'Index' : distri_pl}, gcA)

  zgc = I.newZoneGridConnectivity(parent=zoneB)
  gcB = I.newGridConnectivity('matchB', 'Base/zoneA', 'Abutting1to1', zgc)
  full_pl     = np.array([64,65,66,67,68,69,70,71,72], pdm_dtype) #xmax
  full_pl_opp = np.array([37,38,39,40,41,42,43,44,45], pdm_dtype) #xmin
  distri_pl   = par_utils.uniform_distribution(9, sub_comm)
  I.newGridLocation('FaceCenter', gcB)
  I.newPointList('PointListDonor', full_pl    [distri_pl[0]:distri_pl[1]].reshape(1,-1), gcB)
  I.newPointList('PointList'     , full_pl_opp[distri_pl[0]:distri_pl[1]].reshape(1,-1), gcB)
  MT.newDistribution({'Index' : distri_pl}, gcB)

  VL.generate_jns_vertex_list(tree, sub_comm, have_isolated_faces=have_isolated_faces)

  assert len(I.getNodesFromName(tree, "ZoneGridConnectivity#Vtx")) == 2
  assert I.getNodeFromName(tree, "matchA#Vtx") is not None
  jn_vtx = I.getNodeFromName(tree, "matchB#Vtx")
  assert jn_vtx is not None and PT.Subset.GridLocation(jn_vtx) == 'Vertex'
  assert I.getType(jn_vtx) == 'GridConnectivity_t' and I.getValue(jn_vtx) == I.getValue(gcB)
