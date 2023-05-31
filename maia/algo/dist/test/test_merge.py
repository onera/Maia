import pytest
import pytest_parallel
from   mpi4py import MPI
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia             import npy_pdm_gnum_dtype as pdm_dtype
from maia.pytree.yaml import parse_yaml_cgns
from maia.algo.dist   import matching_jns_tools as MJT
from maia.factory     import full_to_dist as F2D
from maia.factory.dcube_generator import dcube_generate

from maia.algo.dist import merge

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("merge_bc_from_name", [True, False])   #       __
def test_merge_zones_L(comm, merge_bc_from_name):               #      |  |
  # Setup : create 3 2*2*2 cubes and make them connected in L   #    __|__|
  n_vtx = 3                                                     #   |  |  |
  dcubes = [dcube_generate(n_vtx, 1., [0,0,0], comm),           #   |__|__|
            dcube_generate(n_vtx, 1., [1,0,0], comm),           # 
            dcube_generate(n_vtx, 1., [1,1,0], comm)]
  zones = [PT.get_all_Zone_t(dcube)[0] for dcube in dcubes]
  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  # After PDM, each boundary has a different distribution which makes difficult to
  # convert it to joins. Re distribution is done below
  for izone, zone in enumerate(zones):
    zone[0] = f'zone{izone+1}'
    PT.add_child(base, zone)
    for bc in PT.iter_nodes_from_label(zone, 'BC_t'):
      pl = PT.get_child_from_name(bc, 'PointList')
      distri = MT.getDistribution(bc, 'Index')
      data = {'PointList' : pl[1][0]}
      distri_new, data_new = merge._equilibrate_data(data, comm, distri=distri[1])
      PT.set_value(distri, distri_new)
      PT.set_value(pl, data_new['PointList'].reshape((1,-1), order='F'))
      
  #Setup connections
  jn_cur = [['Xmax'], ['Xmin', 'Ymax'], ['Ymin']] #To copy to create jn
  jn_opp = [['Xmin'], ['Xmax', 'Ymin'], ['Ymax']] #To copy to create pld
  zone_opp = [['zone2'], ['zone1', 'zone3'], ['zone2']]
  for izone, zone in enumerate(zones):
    for j,bc_n in enumerate(jn_cur[izone]):
      bc = PT.get_node_from_name(zone, bc_n)
      PT.rm_nodes_from_name(zone, bc_n)
      zgc = PT.update_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
      gc = PT.new_GridConnectivity(f'match{j}', f'{zone_opp[izone][j]}', 'Abutting1to1', parent=zgc)
      for name in [':CGNS#Distribution', 'GridLocation', 'PointList']:
        PT.add_child(gc, PT.get_child_from_name(bc, name))
      ref_bc = PT.get_node_from_name(tree, f'{jn_opp[izone][j]}')
      PT.new_PointList('PointListDonor', PT.get_child_from_name(ref_bc, 'PointList')[1].copy(), parent=gc)

  #Setup some data
  for i_zone, zone in enumerate(zones):
    sol = PT.new_FlowSolution('FlowSolution', loc='Vertex', parent=zone)
    PT.new_DataArray('DomId', (i_zone+1)*np.ones(n_vtx**3, int), parent=sol)
    pl_sol_full = PT.new_FlowSolution('PartialSol', loc='CellCenter')
    PT.new_PointList('PointList', value=np.array([[8]], pdm_dtype), parent=pl_sol_full)
    PT.new_DataArray('SpecificSol', np.array([3.14]), parent=pl_sol_full)
    PT.add_child(zone, F2D.distribute_pl_node(pl_sol_full, comm))

  # If we use private func, we need to add ordinals
  MJT.add_joins_donor_name(tree, comm)
  subset_merge = "name" if merge_bc_from_name else "none"
  merged_zone = merge._merge_zones(tree, comm, subset_merge_strategy=subset_merge)

  assert PT.Zone.n_cell(merged_zone) == 3*((n_vtx-1)**3)
  assert PT.Zone.n_vtx(merged_zone) == 3*(n_vtx**3) - 2*(n_vtx**2)
  assert PT.Zone.n_face(merged_zone) == 3*(3*n_vtx*(n_vtx-1)**2) - 2*(n_vtx-1)**2
  assert PT.get_node_from_label(merged_zone, 'ZoneGridConnectivity_t') is None
  if merge_bc_from_name:
    assert len(PT.get_nodes_from_label(merged_zone, 'BC_t')) == 6 #BC merged by name
  else:
    assert len(PT.get_nodes_from_label(merged_zone, 'BC_t')) == 3*6 - 4 #BC not merged
  assert comm.allreduce(PT.get_node_from_name(merged_zone, 'DomId')[1].size, MPI.SUM) == PT.Zone.n_vtx(merged_zone)

  expected_partial_sol_size = 3 if merge_bc_from_name else 1
  assert comm.allreduce(PT.get_node_from_name(merged_zone, 'SpecificSol')[1].size, MPI.SUM) == expected_partial_sol_size
  if merge_bc_from_name:
    partial_pl = PT.get_node_from_path(merged_zone, 'PartialSol/PointList')
    assert (np.concatenate(comm.allgather(partial_pl[1][0])) == [8,16,24]).all()

@pytest.mark.parametrize("merge_only_two", [False, True])
@pytest_parallel.mark.parallel(1)
def test_merge_zones_I(comm, merge_only_two):
  """ A setup with 3 zones in I direction connected by match
  jns (2) + 1 periodic between first and last zone.
  We request to merge only the two first zones
  """
  # Setup : create 3 2*2*2 cubes and make them connected in I
  n_vtx = 3
  dcubes = [dcube_generate(n_vtx, 1., [0,0,0], comm), 
            dcube_generate(n_vtx, 1., [1,0,0], comm),
            dcube_generate(n_vtx, 1., [2,0,0], comm)]
  zones = [PT.get_all_Zone_t(dcube)[0] for dcube in dcubes]
  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  # After PDM, each boundary has a different distribution which makes difficult to
  # convert it to joins. Re distribution is done below
  for izone, zone in enumerate(zones):
    zone[0] = f'zone{izone+1}'
    PT.add_child(base, zone)
    for bc in PT.iter_nodes_from_label(zone, 'BC_t'):
      pl = PT.get_child_from_name(bc, 'PointList')
      distri = MT.getDistribution(bc, 'Index')
      data = {'PointList' : pl[1][0]}
      distri_new, data_new = merge._equilibrate_data(data, comm, distri=distri[1])
      PT.set_value(distri, distri_new)
      PT.set_value(pl, data_new['PointList'].reshape((1,-1), order='F'))
      
  #Setup connections
  jn_cur = [['Xmax'], ['Xmin', 'Xmax'], ['Xmin']] #To copy to create jn
  jn_opp = [['Xmin'], ['Xmax', 'Xmin'], ['Xmax']] #To copy to create pld
  zone_opp = [['zone2'], ['zone1', 'zone3'], ['zone2']]
  for izone, zone in enumerate(zones):
    for j,bc_n in enumerate(jn_cur[izone]):
      bc = PT.get_node_from_name(zone, bc_n)
      PT.rm_nodes_from_name(zone, bc_n)
      zgc = PT.update_child(zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
      gc = PT.new_GridConnectivity(f'match{j}', f'{zone_opp[izone][j]}', 'Abutting1to1', parent=zgc)
      for name in [':CGNS#Distribution', 'GridLocation', 'PointList']:
        PT.add_child(gc, PT.get_child_from_name(bc, name))
      ref_bc = PT.get_node_from_name(tree, f'{jn_opp[izone][j]}')
      PT.new_PointList('PointListDonor', PT.get_child_from_name(ref_bc, 'PointList')[1].copy(), parent=gc)
  # Add periodic between first and last
  jn_cur = ['Xmin', 'Xmax'] #To copy to create jn
  jn_opp = ['Xmax', 'Xmin'] #To copy to create pld
  zone_opp = ['zone3', 'zone1']
  for izone, zone in zip(range(2), [zones[0], zones[-1]]):
    bc = PT.get_node_from_name(zone, jn_cur[izone])
    zgc = PT.get_node_from_label(zone, 'ZoneGridConnectivity_t')
    gc = PT.new_GridConnectivity('perio', f'{zone_opp[izone]}', 'Abutting1to1', parent=zgc)
    for name in [':CGNS#Distribution', 'GridLocation', 'PointList']:
      PT.add_child(gc, PT.get_child_from_name(bc, name))
    ref_bc = PT.get_node_from_name(tree, f'{jn_opp[izone]}')
    PT.new_PointList('PointListDonor', PT.get_child_from_name(ref_bc, 'PointList')[1].copy(), parent=gc)
    sign = 1 if izone == 0 else -1
    PT.new_GridConnectivityProperty(periodic={'translation' : [sign*3., 0, 0]}, parent=gc)
  for izone, zone in zip(range(2), [zones[0], zones[-1]]):
    PT.rm_nodes_from_name(zone, jn_cur[izone])

  #Setup some data (Only one rank so next lines are OK)
  zsr_full = PT.new_ZoneSubRegion('SubRegion', bc_name='Ymin', loc='FaceCenter', parent=zones[1])
  old_id = PT.get_node_from_path(zones[1], 'ZoneBC/Ymin/PointList')[1][0].copy()
  PT.new_DataArray('OldId', old_id, parent=zsr_full)
  

  if merge_only_two:
    n_merged = 2
    merge.merge_zones(tree, ['Base/zone1', 'Base/zone2'], comm, output_path='MergedBase/MergedZone')
    assert len(PT.get_all_CGNSBase_t(tree)) == len(PT.get_all_CGNSBase_t(tree)) == 2
    merged_zone = PT.get_node_from_path(tree, 'MergedBase/MergedZone')
    assert len(PT.get_nodes_from_label(merged_zone, 'GridConnectivity_t')) == 2
    assert len(PT.get_nodes_from_label(merged_zone, 'Periodic_t')) == 1
  else:
    n_merged = 3
    merge.merge_connected_zones(tree, comm)
    assert len(PT.get_all_Zone_t(tree)) == 1
    merged_zone = PT.get_all_Zone_t(tree)[0]

    assert len(PT.get_nodes_from_label(merged_zone, 'GridConnectivity_t')) == 2
    for gc in PT.iter_nodes_from_label(merged_zone, 'GridConnectivity_t'):
      assert PT.get_value(gc) not in ['zone1', 'zone2', 'zone3']
      assert PT.get_node_from_label(gc, 'Periodic_t') is not None

  assert len(PT.get_nodes_from_label(merged_zone, 'BC_t')) == 4
  assert PT.Zone.n_cell(merged_zone) == n_merged*((n_vtx-1)**3)
  assert PT.Zone.n_vtx(merged_zone) == n_merged*(n_vtx**3) - (n_merged-1)*(n_vtx**2)
  assert PT.Zone.n_face(merged_zone) == n_merged*(3*n_vtx*(n_vtx-1)**2) - (n_merged-1)*(n_vtx-1)**2
  assert PT.get_node_from_path(merged_zone, 'SubRegion/GridLocation') is not None
  assert PT.get_node_from_path(merged_zone, 'SubRegion/BCRegionName') is None
  assert (PT.get_node_from_path(merged_zone, 'SubRegion/OldId')[1] == old_id).all()
  assert not (PT.get_node_from_path(merged_zone, 'SubRegion/PointList')[1] == old_id).all()

@pytest_parallel.mark.parallel(3)
def test_equilibrate_data(comm):

  rank = comm.Get_rank()
  data = {'rank' : rank * np.ones(10*rank, np.int32),
          'range': np.arange(10*rank).astype(float)}  #unequilibrated data

  current_distri_f = np.array([0, 0, 10, 30], pdm_dtype)
  current_distri = current_distri_f[[rank, rank+1, comm.Get_size()]]

  expected_distri = np.array([0, 10, 20, 30])[[rank, rank+1, comm.Get_size()]]
  expected_rank_f = np.concatenate([np.ones(10, np.int32), 2*np.ones(20, np.int32)])
  expected_range_f = np.concatenate([np.arange(10), np.arange(20)]).astype(float)

  distri, data_eq = merge._equilibrate_data(data, comm)
  assert (distri == expected_distri).all()
  assert (data_eq['rank'] == expected_rank_f[distri[0]:distri[1]]).all()
  assert (data_eq['range'] == expected_range_f[distri[0]:distri[1]]).all()

  distri, data_eq = merge._equilibrate_data(data, comm, distri=current_distri)
  assert (distri == expected_distri).all()
  assert (data_eq['rank'] == expected_rank_f[distri[0]:distri[1]]).all()
  assert (data_eq['range'] == expected_range_f[distri[0]:distri[1]]).all()

  distri, data_eq = merge._equilibrate_data(data, comm, distri_full=current_distri_f)
  assert (distri == expected_distri).all()
  assert (data_eq['rank'] == expected_rank_f[distri[0]:distri[1]]).all()
  assert (data_eq['range'] == expected_range_f[distri[0]:distri[1]]).all()
