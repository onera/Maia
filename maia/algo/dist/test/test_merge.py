import pytest
import pytest_parallel
from   mpi4py import MPI
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from maia             import npy_pdm_gnum_dtype as pdm_dtype
from maia.algo.dist   import matching_jns_tools as MJT
from maia.factory     import full_to_dist as F2D
from maia.utils       import par_utils
from maia.utils       import logging as mlog
from maia.factory.dcube_generator import dcube_generate

from maia.algo.dist import merge

class log_capture:
  def __init__(self):
    self.logs = ''
  def log(self, msg):
    self.logs += msg


@pytest_parallel.mark.parallel(1)
def test_pre_merge_families(comm):
  ftree = PT.yaml.to_cgns_tree("""
  Zone Zone_t [[16, 9, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZoneBC ZoneBC_t:
      left BC_t: # No FamilyName
        PointList IndexArray_t [[1,5,9,13]]:
      right BC_t: # Only one node
        FamilyName FamilyName_t "AUBE":
        PointList IndexArray_t [[4,8,12,16]]:
      bottom BC_t "BCWall":
        FamilyName FamilyName_t "AMONT":
        PointList IndexArray_t [[1,2,3,4]]:
        BCDataSetFull BCDataSet_t 'Null':
          DirichletData BCData_t:
            Data DataArray_t [10., 20., 30., 40.]:
        BCDataSetPart BCDataSet_t 'Null':
          PointList IndexArray_t [[1,2]]:
          DirichletData BCData_t:
            Data DataArray_t [10., 20.]:
      top BC_t "BCWall":
        FamilyName FamilyName_t "AMONT":
        PointList IndexArray_t [[16,15,14,13]]:
        BCDataSetFull BCDataSet_t 'Null':
          DirichletData BCData_t:
            Data DataArray_t [-10., -20., -30., -40.]:
        BCDataSetPart BCDataSet_t 'Null':
          PointList IndexArray_t [[15,16]]:
          DirichletData BCData_t:
            Data DataArray_t [-10., -20.]:
  """)
  tree = maia.factory.full_to_dist_tree(ftree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  zone_bck = PT.deep_copy(zone)
  merge.pre_merge_families_per_zone(zone, 'ZoneBC_t/BC_t', comm)
  assert PT.is_same_tree(PT.get_node_from_name(zone,     'left'),
                         PT.get_node_from_name(zone_bck, 'left'))
  assert PT.is_same_tree(PT.get_node_from_name(zone,     'right'),
                         PT.get_node_from_name(zone_bck, 'right'))
  
  merged_expt_f = PT.new_BC('mergedAMONT', 'BCWall', point_list=[[1,2,3,4, 16,15,14,13]], family='AMONT')
  ds = PT.new_BCDataSet('BCDataSetFull', parent=merged_expt_f)
  PT.new_BCData('DirichletData', {'Data' : [10.,20,30,40, -10,-20,-30,-40]}, parent=ds)
  ds = PT.new_BCDataSet('BCDataSetPart', point_list=[[1,2,15,16]], parent=merged_expt_f)
  PT.new_BCData('DirichletData', {'Data' : [10.,20,-10,-20]}, parent=ds)

  from maia.factory.full_to_dist import distribute_pl_node
  merged_expt = distribute_pl_node(merged_expt_f, comm)
  assert PT.is_same_tree(PT.get_node_from_name(zone,     'mergedAMONT'),
                         merged_expt)


def test_gather_subsets():
  to_names = lambda d: {key : [n[0] if n else None for n in val] for key,val in d.items()}
  zones = PT.yaml.to_nodes("""
  Zone1 Zone_t:
    ZoneBC ZoneBC_t:
      aval BC_t:
        DiData BCDataSet_t:
        NeData BCDataSet_t:
      amontA BC_t:
        FamilyName FamilyName_t "AMONT":
      aube BC_t:
        FamilyName FamilyName_t "AUBE":
  Zone2 Zone_t:
    ZoneBC ZoneBC_t:
      aval BC_t:
        DiData BCDataSet_t:
      amontB BC_t:
        FamilyName FamilyName_t "AMONT":
        DiData BCDataSet_t:
  """)
  gathered = merge.gather_subsets(zones, 'ZoneBC/BC_t', 'None')
  assert to_names(gathered) == {'ZoneBC/aval.0'   : ['aval', None],
                                'ZoneBC/aval.1'   : [None, 'aval'],
                                'ZoneBC/amontA.0' : ['amontA', None],
                                'ZoneBC/amontB.1' : [None, 'amontB'],
                                'ZoneBC/aube.0'   : ['aube', None]}

  gathered = merge.gather_subsets(zones, 'ZoneBC/BC_t', 'name')
  assert to_names(gathered) == {'ZoneBC/aval'   : ['aval', 'aval'],
                                'ZoneBC/amontA' : ['amontA', None],
                                'ZoneBC/amontB' : [None, 'amontB'],
                                'ZoneBC/aube'   : ['aube', None]}

  gathered = merge.gather_subsets(zones, 'ZoneBC/BC_t', 'family')
  assert to_names(gathered) == {'ZoneBC/aval'   : ['aval', 'aval'],
                                'ZoneBC/AMONT'  : ['amontA', 'amontB'],
                                'ZoneBC/AUBE'   : ['aube', None]}

  # For BCDS, distinction must be done at BC level
  gathered = merge.gather_subsets(zones, ['ZoneBC_t','BC_t','BCDataSet_t'], 'None', True)
  assert to_names(gathered) == {'ZoneBC/aval.0/DiData'   : ['DiData', None],
                                'ZoneBC/aval.0/NeData'   : ['NeData', None],
                                'ZoneBC/aval.1/DiData'   : [None, 'DiData'],
                                'ZoneBC/amontB.1/DiData' : [None, 'DiData']}

  gathered = merge.gather_subsets(zones, ['ZoneBC_t','BC_t','BCDataSet_t'], 'name', True)
  assert to_names(gathered) == {'ZoneBC/aval/DiData'   : ['DiData', 'DiData'],
                                'ZoneBC/aval/NeData'   : ['NeData', None],
                                'ZoneBC/amontB/DiData' : [None, 'DiData']}
  gathered = merge.gather_subsets(zones, ['ZoneBC_t','BC_t','BCDataSet_t'], 'family', True)
  assert to_names(gathered) == {'ZoneBC/aval/DiData'   : ['DiData', 'DiData'],
                                'ZoneBC/aval/NeData'   : ['NeData', None],
                                'ZoneBC/AMONT/DiData' : [None, 'DiData']}


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
      PT.new_IndexArray('PointListDonor', PT.get_child_from_name(ref_bc, 'PointList')[1].copy(), parent=gc)

  #Setup some data
  for i_zone, zone in enumerate(zones):
    sol = PT.new_FlowSolution('FlowSolution', loc='Vertex', parent=zone)
    PT.new_DataArray('DomId', (i_zone+1)*np.ones(n_vtx**3, int), parent=sol)
    pl_sol_full = PT.new_FlowSolution('PartialSol', loc='CellCenter')
    PT.new_IndexArray('PointList', value=np.array([[8]], pdm_dtype), parent=pl_sol_full)
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
      PT.new_IndexArray('PointListDonor', PT.get_child_from_name(ref_bc, 'PointList')[1].copy(), parent=gc)
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
    PT.new_IndexArray('PointListDonor', PT.get_child_from_name(ref_bc, 'PointList')[1].copy(), parent=gc)
    sign = 1 if izone == 0 else -1
    PT.new_GridConnectivityProperty(periodic={'translation' : [sign*3., 0, 0]}, parent=gc)
  for izone, zone in zip(range(2), [zones[0], zones[-1]]):
    PT.rm_nodes_from_name(zone, jn_cur[izone])

    
  # Mimic a non 1to1 jn
  zbc = PT.get_child_from_label(zones[1], 'ZoneBC_t')
  zgc = PT.get_child_from_label(zones[1], 'ZoneGridConnectivity_t')
  jn = PT.get_child_from_name(zbc, 'Zmin')
  PT.set_label(jn, 'GridConnectivity_t')
  PT.new_child(jn, 'GridConnectivityType', 'GridConnectivityType_t', 'Abutting')
  PT.set_value(jn, PT.get_name(zones[1]))
  PT.rm_child(zbc, jn)
  PT.add_child(zgc, jn)

  #Setup some data (Only one rank so next lines are OK)
  zsr_full = PT.new_ZoneSubRegion('SubRegion', bc_name='Ymin', loc='FaceCenter', parent=zones[1])
  old_id = PT.get_node_from_path(zones[1], 'ZoneBC/Ymin/PointList')[1][0].copy()
  PT.new_DataArray('OldId', old_id, parent=zsr_full)
  

  if merge_only_two:
    n_merged = 2
    merge.merge_zones(tree, ['Base/zone1', 'Base/zone2'], comm, output_path='MergedBase/MergedZone')
    assert len(PT.get_all_CGNSBase_t(tree)) == len(PT.get_all_CGNSBase_t(tree)) == 2
    merged_zone = PT.get_node_from_path(tree, 'MergedBase/MergedZone')
    assert len(PT.get_nodes_from_label(merged_zone, 'GridConnectivity_t')) == 2 + 1 #1 non abbuting
    assert len(PT.get_nodes_from_label(merged_zone, 'Periodic_t')) == 1
  else:
    n_merged = 3
    merge.merge_connected_zones(tree, comm)
    assert len(PT.get_all_Zone_t(tree)) == 1
    merged_zone = PT.get_all_Zone_t(tree)[0]

    assert len(PT.get_nodes_from_label(merged_zone, 'GridConnectivity_t')) == 2 + 1 #1 non abbuting
    for gc in PT.iter_nodes_from_label(merged_zone, 'GridConnectivity_t'):
      assert PT.get_value(gc) == 'Base/mergedZone0'
      assert (PT.get_node_from_label(gc, 'Periodic_t') is not None) == (PT.get_name(gc) != 'Zmin')

  assert len(PT.get_nodes_from_label(merged_zone, 'BC_t')) == 4
  assert PT.Zone.n_cell(merged_zone) == n_merged*((n_vtx-1)**3)
  assert PT.Zone.n_vtx(merged_zone) == n_merged*(n_vtx**3) - (n_merged-1)*(n_vtx**2)
  assert PT.Zone.n_face(merged_zone) == n_merged*(3*n_vtx*(n_vtx-1)**2) - (n_merged-1)*(n_vtx-1)**2
  assert PT.get_node_from_path(merged_zone, 'SubRegion/GridLocation') is not None
  assert PT.get_node_from_path(merged_zone, 'SubRegion/BCRegionName') is None
  assert (PT.get_node_from_path(merged_zone, 'SubRegion/OldId')[1] == old_id).all()
  assert not (PT.get_node_from_path(merged_zone, 'SubRegion/PointList')[1] == old_id).all()


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("subset_merge", ["none", "name", "family"])
def test_merge_subsets(subset_merge, comm):
  # Setup : create 2 2*2*2 cubes, not even connected
  n_vtx = 3
  dcubes = [dcube_generate(n_vtx, 1., [0,0,0], comm), 
            dcube_generate(n_vtx, 1., [1,0,0], comm)]
  zones = [PT.get_all_Zone_t(dcube)[0] for dcube in dcubes]
  tree = PT.new_CGNSTree()
  base = PT.new_CGNSBase(parent=tree)
  for izone, zone in enumerate(zones):
    zone[0] = f'zone{izone+1}'
    PT.add_child(base, zone)

  for bc in PT.get_nodes_from_label(tree, 'BC_t'):
    PT.new_FamilyName(PT.get_name(bc).upper(), parent=bc)

  # A full BCDS existing on the two zones
  for i,zone in enumerate(zones):
    bc = PT.get_node_from_name(zone, 'Xmin')
    pl = PT.get_child_from_name(bc, 'PointList')[1]
    bcds = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
    bcda = PT.new_child(bcds, 'DirichletData', 'BCData_t')
    PT.new_DataArray('iZone', i*np.ones(pl.size), parent=bcda)
    PT.new_DataArray('iRank', comm.rank*np.ones(pl.size), parent=bcda)
  # A full BCDS existing only on one zone ---> NOT SUPPORTED with name/family
  bc = PT.get_node_from_name(zones[1], 'Xmax')
  pl = PT.get_child_from_name(bc, 'PointList')[1]
  bcds = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
  bcda = PT.new_child(bcds, 'DirichletData', 'BCData_t')
  PT.new_DataArray('iZone', i*np.ones(pl.size), parent=bcda)
  PT.new_DataArray('iRank', comm.rank*np.ones(pl.size), parent=bcda)
  # A partial BCDS existing on the two zones
  for i,zone in enumerate(zones):
    bc = PT.get_node_from_name(zone, 'Ymin')
    pl = PT.get_child_from_name(bc, 'PointList')[1][0][::2]
    bcds = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
    PT.new_IndexArray(value=pl.reshape((1,-1), order='F'), parent=bcds)
    PT.new_GridLocation(PT.Subset.GridLocation(bc), parent=bcds)
    bcda = PT.new_child(bcds, 'DirichletData', 'BCData_t')
    PT.new_DataArray('iZone', i*np.ones(pl.size), parent=bcda)
    PT.new_DataArray('iRank', comm.rank*np.ones(pl.size), parent=bcda)
    MT.newDistribution({'Index' : par_utils.dn_to_distribution(pl.size, comm)}, parent=bcds)
  # A partial BCDS existing only on one zone
  bc = PT.get_node_from_name(zones[1], 'Ymax')
  pl = PT.get_child_from_name(bc, 'PointList')[1][0][::2]
  bcds = PT.new_child(bc, 'BCDataSet', 'BCDataSet_t')
  PT.new_IndexArray(value=pl.reshape((1,-1), order='F'), parent=bcds)
  PT.new_GridLocation(PT.Subset.GridLocation(bc), parent=bcds)
  bcda = PT.new_child(bcds, 'DirichletData', 'BCData_t')
  PT.new_DataArray('iZone', i*np.ones(pl.size), parent=bcda)
  PT.new_DataArray('iRank', comm.rank*np.ones(pl.size), parent=bcda)
  MT.newDistribution({'Index' : par_utils.dn_to_distribution(pl.size, comm)}, parent=bcds)

  if subset_merge != "none":
    for bc in PT.get_nodes_from_name(tree, 'Xmax'):
      PT.rm_children_from_label(bc, 'BCDataSet_t')

  merge.merge_zones(tree, '*', comm, subset_merge=subset_merge)

  # For easier comparaison, gather data on rank 0
  ftree = maia.factory.dist_to_full_tree(tree, comm, 0)
  if comm.rank == 0:
    if subset_merge != "none":
      zmin = PT.new_BC('Zmin', "Null", point_list=[[1,2,3,4, 37,38,39,40]], loc='FaceCenter', family='ZMIN')
      zmax = PT.new_BC('Zmax', "Null", point_list=[[9,10,11,12, 45,46,47,48]], loc='FaceCenter', family='ZMAX')
      xmin = PT.new_BC('Xmin', "Null", point_list=[[13,14,15,16, 49,50,51,52]], loc='FaceCenter', family='XMIN')
      ds = PT.new_BCDataSet(type=None, parent=xmin)
      PT.new_BCData('DirichletData', fields={'iZone': np.array([0.,0,0,0,1,1,1,1]), 'iRank' : np.zeros(8)}, parent=ds)
      xmax = PT.new_BC('Xmax', "Null", point_list=[[21,22,23,24, 57,58,59,60]], loc='FaceCenter', family='XMAX')
      ymin = PT.new_BC('Ymin', "Null", point_list=[[25,26,27,28, 61,62,63,64]], loc='FaceCenter', family='YMIN')
      ds = PT.new_BCDataSet(type=None, loc='FaceCenter', point_list=[[25,27, 61,63]], parent=ymin)
      PT.new_BCData('DirichletData', fields={'iZone': np.array([0.,0,1,1]), 'iRank' : np.ones(4)}, parent=ds)
      ymax = PT.new_BC('Ymax', "Null", point_list=[[33,34,35,36, 69,70,71,72]], loc='FaceCenter', family='YMAX')
      ds = PT.new_BCDataSet(loc='FaceCenter', type=None, point_list=[[69,71]], parent=ymax)
      PT.new_BCData('DirichletData', fields={'iZone': np.array([1.,1]), 'iRank' : np.ones(2)}, parent=ds)
      zbc = PT.new_node('ZoneBC', 'ZoneBC_t', children=[xmin,xmax,ymin,ymax,zmin,zmax])

      if subset_merge == 'family':
        for bc in PT.get_children(zbc):
          PT.set_name(bc, PT.get_name(bc).upper())

    else:
      zmin0 = PT.new_BC('Zmin.0', "Null", point_list=[[1,2,3,4]], loc='FaceCenter', family='ZMIN')
      zmin1 = PT.new_BC('Zmin.1', "Null", point_list=[[37,38,39,40]], loc='FaceCenter', family='ZMIN')

      zmax0 = PT.new_BC('Zmax.0', "Null", point_list=[[9,10,11,12]], loc='FaceCenter', family='ZMAX')
      zmax1 = PT.new_BC('Zmax.1', "Null", point_list=[[45,46,47,48]], loc='FaceCenter', family='ZMAX')

      xmin0 = PT.new_BC('Xmin.0', "Null", point_list=[[13,14,15,16]], loc='FaceCenter', family='XMIN')
      ds = PT.new_BCDataSet(type=None, parent=xmin0)
      PT.new_BCData('DirichletData', {'iZone' : np.zeros(4), 'iRank' : np.zeros(4)}, ds)
      xmin1 = PT.new_BC('Xmin.1', "Null", point_list=[[49,50,51,52]], loc='FaceCenter', family='XMIN')
      ds = PT.new_BCDataSet(type=None, parent=xmin1)
      PT.new_BCData('DirichletData', {'iZone' : np.ones(4), 'iRank' : np.zeros(4)}, ds)

      xmax0 = PT.new_BC('Xmax.0', "Null", point_list=[[21,22,23,24]], loc='FaceCenter', family='XMAX')
      xmax1 = PT.new_BC('Xmax.1', "Null", point_list=[[57,58,59,60]], loc='FaceCenter', family='XMAX')
      ds = PT.new_BCDataSet(type=None, parent=xmax1)
      PT.new_BCData('DirichletData', {'iZone' : np.ones(4), 'iRank' : np.ones(4)}, ds)

      ymin0 = PT.new_BC('Ymin.0', "Null", point_list=[[25,26,27,28]], loc='FaceCenter', family='YMIN')
      ds = PT.new_BCDataSet(type=None, loc='FaceCenter', point_list=[[25,27]], parent=ymin0)
      PT.new_BCData('DirichletData', {'iZone' : np.zeros(2), 'iRank' : np.ones(2)}, ds)
      ymin1 = PT.new_BC('Ymin.1', "Null", point_list=[[61,62,63,64]], loc='FaceCenter', family='YMIN')
      ds = PT.new_BCDataSet(type=None, loc='FaceCenter', point_list=[[61,63]], parent=ymin1)
      PT.new_BCData('DirichletData', {'iZone' : np.ones(2), 'iRank' : np.ones(2)}, ds)

      ymax0 = PT.new_BC('Ymax.0', "Null", point_list=[[33,34,35,36]], loc='FaceCenter', family='YMAX')
      ymax1 = PT.new_BC('Ymax.1', "Null", point_list=[[69,70,71,72]], loc='FaceCenter', family='YMAX')
      ds = PT.new_BCDataSet(type=None, loc='FaceCenter', point_list=[[69,71]], parent=ymax1)
      PT.new_BCData('DirichletData', {'iZone' : np.ones(2), 'iRank' : np.ones(2)}, ds)

 
      zbc = PT.new_node('ZoneBC', 'ZoneBC_t', children=[xmin0,xmax0,ymin0,ymax0,zmin0,zmax0,\
                                                        xmin1,xmax1,ymin1,ymax1,zmin1,zmax1])

    assert PT.is_same_tree(zbc, PT.get_node_from_name(ftree, 'ZoneBC'), type_tol=True)


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

@pytest_parallel.mark.parallel([2])
def test_input_overflow(comm):
  tree = PT.yaml.to_cgns_tree("""
  Zone1 Zone_t I4 [[1, 400000000, 0]]:
    :CGNS#Distribution UserDefinedData_t: # Fake distribution to avoid check
    ZoneType ZoneType_t "Unstructured":
    NGON Elements_t [22, 0]:
      ElementRange IndexRange_t I4 [1, 800000000]: # Fake value to overflow
    NFace Elements_t [23, 0]:
      ElementRange IndexRange_t I4 [800000001, 1200000000]: # Fake value to overflow
  Zone2 Zone_t [[1, 500000000, 0]]:
    :CGNS#Distribution UserDefinedData_t: # Fake distribution to avoid check
    ZoneType ZoneType_t "Unstructured":
    NGON Elements_t [22, 0]:
      ElementRange IndexRange_t I4 [1, 900000000]: # Fake value to overflow
    NFace Elements_t [23, 0]:
      ElementRange IndexRange_t I4 [900000001, 1400000000]: # Fake value to overflow
  """)
  if pdm_dtype == np.int32:
    with pytest.raises(OverflowError):
      merge.merge_zones(tree, ['Base/Zone1', 'Base/Zone2'], comm)
  else:
    log_collector = log_capture()
    mlog.add_printer_to_logger('maia-warnings', log_collector)
    with pytest.raises(Exception): # Test will fail but we should get the warning
      merge.merge_zones(tree, ['Base/Zone1', 'Base/Zone2'], comm)
    assert "I4 integers, but result of _merge_zones would overflow it" in log_collector.logs
