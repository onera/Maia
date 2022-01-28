import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import maia.sids.Internal_ext as IE

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.sids import sids
from maia.utils         import parse_yaml_cgns
from maia.generate.dcube_generator import dcube_generate
from maia.transform.dist_tree import add_joins_ordinal as AJO

from maia.transform import merge

def test_find_connected_zones():
  yt = """
  BaseA CGNSBase_t:
    Zone1 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone2 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone4":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone3 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match1 GridConnectivity_t "BaseA/Zone1":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
        match2 GridConnectivity_t "BaseB/Zone6":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
    Zone4 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "Zone2":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  BaseB CGNSBase_t:
    Zone5 Zone_t:
    Zone6 Zone_t:
      ZGC ZoneGridConnectivity_t:
        match GridConnectivity_t "BaseA/Zone3":
          GridConnectivityType GridConnectivityType_t "Abutting1to1":
  """
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  connected_path = merge._find_connected_zones(tree)
  assert len(connected_path) == 3
  for zones in connected_path:
    if len(zones) == 1:
      assert zones == ['BaseB/Zone5']
    if len(zones) == 2:
      assert sorted(zones) == ['BaseA/Zone2', 'BaseA/Zone4']
    if len(zones) == 3:
      assert sorted(zones) == ['BaseA/Zone1', 'BaseA/Zone3', 'BaseB/Zone6']

@mark_mpi_test([1,3])
@pytest.mark.parametrize("merge_bc_from_name", [True, False])
def test_merge_zones_L(sub_comm, merge_bc_from_name):
  # Setup : create 3 2*2*2 cubes and make them connected in L
  n_vtx = 3
  dcubes = [dcube_generate(n_vtx, 1., [0,0,0], sub_comm), 
            dcube_generate(n_vtx, 1., [1,0,0], sub_comm),
            dcube_generate(n_vtx, 1., [1,1,0], sub_comm)]
  zones = [I.getZones(dcube)[0] for dcube in dcubes]
  tree = I.newCGNSTree()
  base = I.newCGNSBase(parent=tree)
  # After PDM, each boundary has a different distribution which makes difficult to
  # convert it to joins. Re distribution is done below
  for zone in zones:
    I._addChild(base, zone)
    for bc in I.getNodesFromType(zone, 'BC_t'):
      pl = I.getNodeFromName1(bc, 'PointList')
      distri = IE.getDistribution(bc, 'Index')
      data = {'PointList' : pl[1][0]}
      distri_new, data_new = merge._equilibrate_data(data, sub_comm, distri=distri[1])
      I.setValue(distri, distri_new)
      I.setValue(pl, data_new['PointList'].reshape((1,-1), order='F'))
      
  #Setup connections
  jn_cur = [['dcube_bnd_3'], ['dcube_bnd_2', 'dcube_bnd_5'], ['dcube_bnd_4']] #To copy to create jn
  jn_opp = [['dcube_bnd_2'], ['dcube_bnd_3', 'dcube_bnd_4'], ['dcube_bnd_5']] #To copy to create pld
  zone_opp = [['zone2'], ['zone1', 'zone3'], ['zone2']]
  for izone, zone in enumerate(zones):
    zone[0] = f'zone{izone+1}'
    for j,bc_n in enumerate(jn_cur[izone]):
      bc = I.getNodeFromName(zone, bc_n)
      I._rmNodesByName(zone, bc_n)
      zgc = I.newZoneGridConnectivity(parent=zone)
      gc = I.newGridConnectivity(f'match{j}', f'{zone_opp[izone][j]}', 'Abutting1to1', zgc)
      for name in [':CGNS#Distribution', 'GridLocation', 'PointList']:
        I._addChild(gc, I.getNodeFromName1(bc, name))
      ref_bc = I.getNodeFromName(tree, f'{jn_opp[izone][j]}')
      I.newIndexArray('PointListDonor', I.getNodeFromName1(ref_bc, 'PointList')[1].copy(), parent=gc)


  # If we use private func, we need to add ordinals
  AJO.add_joins_ordinal(tree, sub_comm)
  subset_merge = "name" if merge_bc_from_name else "none"
  merged_zone = merge._merge_zones(tree, sub_comm, subset_merge_strategy=subset_merge)

  assert sids.Zone.n_cell(merged_zone) == 3*((n_vtx-1)**3)
  assert sids.Zone.n_vtx(merged_zone) == 3*(n_vtx**3) - 2*(n_vtx**2)
  assert sids.Zone.n_face(merged_zone) == 3*(3*n_vtx*(n_vtx-1)**2) - 2*(n_vtx-1)**2
  assert I.getNodeFromType(merged_zone, 'ZoneGridConnectivity_t') is None
  if merge_bc_from_name:
    assert len(I.getNodesFromType(merged_zone, 'BC_t')) == 6 #BC merged by name
  else:
    assert len(I.getNodesFromType(merged_zone, 'BC_t')) == 3*6 - 4 #BC not merged

@pytest.mark.parametrize("merge_only_two", [False, True])
@mark_mpi_test(1)
def test_merge_zones_I(sub_comm, merge_only_two):
  """ A setup with 3 zones I-shaped connected by match
  jns (2) + 1 periodic between first and last zone.
  We request to merge only the two first zones
  """
  # Setup : create 3 2*2*2 cubes and make them connected in I
  n_vtx = 3
  dcubes = [dcube_generate(n_vtx, 1., [0,0,0], sub_comm), 
            dcube_generate(n_vtx, 1., [1,0,0], sub_comm),
            dcube_generate(n_vtx, 1., [2,0,0], sub_comm)]
  zones = [I.getZones(dcube)[0] for dcube in dcubes]
  tree = I.newCGNSTree()
  base = I.newCGNSBase(parent=tree)
  # After PDM, each boundary has a different distribution which makes difficult to
  # convert it to joins. Re distribution is done below
  for zone in zones:
    I._addChild(base, zone)
    for bc in I.getNodesFromType(zone, 'BC_t'):
      pl = I.getNodeFromName1(bc, 'PointList')
      distri = IE.getDistribution(bc, 'Index')
      data = {'PointList' : pl[1][0]}
      distri_new, data_new = merge._equilibrate_data(data, sub_comm, distri=distri[1])
      I.setValue(distri, distri_new)
      I.setValue(pl, data_new['PointList'].reshape((1,-1), order='F'))
      
  #Setup connections
  jn_cur = [['dcube_bnd_3'], ['dcube_bnd_2', 'dcube_bnd_3'], ['dcube_bnd_2']] #To copy to create jn
  jn_opp = [['dcube_bnd_2'], ['dcube_bnd_3', 'dcube_bnd_2'], ['dcube_bnd_3']] #To copy to create pld
  zone_opp = [['zone2'], ['zone1', 'zone3'], ['zone2']]
  for izone, zone in enumerate(zones):
    zone[0] = f'zone{izone+1}'
    for j,bc_n in enumerate(jn_cur[izone]):
      bc = I.getNodeFromName(zone, bc_n)
      I._rmNodesByName(zone, bc_n)
      zgc = I.newZoneGridConnectivity(parent=zone)
      gc = I.newGridConnectivity(f'match{j}', f'{zone_opp[izone][j]}', 'Abutting1to1', zgc)
      for name in [':CGNS#Distribution', 'GridLocation', 'PointList']:
        I._addChild(gc, I.getNodeFromName1(bc, name))
      ref_bc = I.getNodeFromName(tree, f'{jn_opp[izone][j]}')
      I.newIndexArray('PointListDonor', I.getNodeFromName1(ref_bc, 'PointList')[1].copy(), parent=gc)
  # Add periodic between first and last
  jn_cur = ['dcube_bnd_2', 'dcube_bnd_3'] #To copy to create jn
  jn_opp = ['dcube_bnd_3', 'dcube_bnd_2'] #To copy to create pld
  zone_opp = ['zone3', 'zone1']
  for izone, zone in zip(range(2), [zones[0], zones[-1]]):
    bc = I.getNodeFromName(zone, jn_cur[izone])
    zgc = I.getNodeFromType(zone, 'ZoneGridConnectivity_t')
    gc = I.newGridConnectivity('perio', f'{zone_opp[izone]}', 'Abutting1to1', zgc)
    for name in [':CGNS#Distribution', 'GridLocation', 'PointList']:
      I._addChild(gc, I.getNodeFromName1(bc, name))
    ref_bc = I.getNodeFromName(tree, f'{jn_opp[izone]}')
    I.newIndexArray('PointListDonor', I.getNodeFromName1(ref_bc, 'PointList')[1].copy(), parent=gc)
    gcp = I.newGridConnectivityProperty(parent=gc)
    sign = 1 if izone == 0 else -1
    I.newPeriodic(rotationCenter=[0.,0.,0.], rotationAngle=[0.,0.,0.], translation=[sign*3.,0.,0.], parent=gcp)
  for izone, zone in zip(range(2), [zones[0], zones[-1]]):
    I._rmNodesByName(zone, jn_cur[izone])


  if merge_only_two:
    n_merged = 2
    merge.merge_zones(tree, ['Base/zone1', 'Base/zone2'], sub_comm, output_path='MergedBase/MergedZone')
    assert len(I.getBases(tree)) == len(I.getBases(tree)) == 2
    merged_zone = I.getNodeFromPath(tree, 'MergedBase/MergedZone')
    assert len(I.getNodesFromType(merged_zone, 'GridConnectivity_t')) == 2
    assert len(I.getNodesFromType(merged_zone, 'Periodic_t')) == 1
  else:
    n_merged = 3
    merge.merge_connected_zones(tree, sub_comm)
    assert len(I.getZones(tree)) == 1
    merged_zone = I.getZones(tree)[0]

    assert len(I.getNodesFromType(merged_zone, 'GridConnectivity_t')) == 2
    for gc in I.getNodesFromType(merged_zone, 'GridConnectivity_t'):
      assert I.getValue(gc) not in ['zone1', 'zone2', 'zone3']
      assert I.getNodeFromType(gc, 'Periodic_t') is not None

  assert len(I.getNodesFromType(merged_zone, 'BC_t')) == 4
  assert sids.Zone.n_cell(merged_zone) == n_merged*((n_vtx-1)**3)
  assert sids.Zone.n_vtx(merged_zone) == n_merged*(n_vtx**3) - (n_merged-1)*(n_vtx**2)
  assert sids.Zone.n_face(merged_zone) == n_merged*(3*n_vtx*(n_vtx-1)**2) - (n_merged-1)*(n_vtx-1)**2

@mark_mpi_test(3)
def test_equilibrate_data(sub_comm):

  rank = sub_comm.Get_rank()
  data = {'rank' : rank * np.ones(10*rank, np.int32),
          'range': np.arange(10*rank).astype(float)}  #unequilibrated data

  current_distri_f = np.array([0, 0, 10, 30], pdm_dtype)
  current_distri = current_distri_f[[rank, rank+1, sub_comm.Get_size()]]

  expected_distri = np.array([0, 10, 20, 30])[[rank, rank+1, sub_comm.Get_size()]]
  expected_rank_f = np.concatenate([np.ones(10, np.int32), 2*np.ones(20, np.int32)])
  expected_range_f = np.concatenate([np.arange(10), np.arange(20)]).astype(float)

  distri, data_eq = merge._equilibrate_data(data, sub_comm)
  assert (distri == expected_distri).all()
  assert (data_eq['rank'] == expected_rank_f[distri[0]:distri[1]]).all()
  assert (data_eq['range'] == expected_range_f[distri[0]:distri[1]]).all()

  distri, data_eq = merge._equilibrate_data(data, sub_comm, distri=current_distri)
  assert (distri == expected_distri).all()
  assert (data_eq['rank'] == expected_rank_f[distri[0]:distri[1]]).all()
  assert (data_eq['range'] == expected_range_f[distri[0]:distri[1]]).all()

  distri, data_eq = merge._equilibrate_data(data, sub_comm, distri_full=current_distri_f)
  assert (distri == expected_distri).all()
  assert (data_eq['rank'] == expected_rank_f[distri[0]:distri[1]]).all()
  assert (data_eq['range'] == expected_range_f[distri[0]:distri[1]]).all()
