import pytest
import pytest_parallel
import os
import numpy as np

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils          import par_utils
from maia.factory        import full_to_dist as F2D

import maia.utils.test_utils as TU

from maia.algo.dist import concat_nodes as GN

@pytest.mark.parametrize("default_bcds", [True, False])
@pytest_parallel.mark.parallel([1,2])
def test_concatenate_subset_nodes(default_bcds, comm):
  yt = """
  BCa BC_t "BCFarfield":
    GridLocation GridLocation_t "FaceCenter":
    PointList IndexArray_t [[1, 2, 3, 4]]:
    BCDataSet BCDataSet_t:
      BCData BCData_t:
        Data DataArray_t [10., 20., 30., 40.]:
  BCb BC_t "BCFarfield":
    GridLocation GridLocation_t "FaceCenter":
    PointList IndexArray_t [[10, 20, 30, 40, 50, 60, 70, 80]]:
    BCDataSet BCDataSet_t:
      BCData BCData_t:
        Data DataArray_t [1., 2., 3., 4., 5., 6., 7., 8.]:
  """
  subset_nodes_f = PT.yaml.to_nodes(yt)
  subset_nodes = [F2D.distribute_pl_node(node, comm) for node in subset_nodes_f]

  expected_distri = par_utils.uniform_distribution(4+8, comm)
  if comm.Get_size() == 1:
    expected_pl = [[1,2,3,4, 10,20,30,40,50,60,70,80]]
    expected_data = [10,20,30,40, 1.,2.,3.,4.,5.,6.,7.,8]
  elif comm.Get_size() == 2:
    if comm.Get_rank() == 0:
      expected_pl = [[1,2, 10,20,30,40]]
      expected_data = [10,20, 1.,2.,3.,4]
    elif comm.Get_rank() == 1:
      expected_pl = [[3,4, 50,60,70,80]]
      expected_data = [30,40, 5.,6.,7.,8]

  if default_bcds:
    node = GN.concatenate_subset_nodes(subset_nodes, comm, output_name='BothBC', \
        additional_data_queries = ['BCDataSet/BCData/Data'])
  else:
    node = GN.concatenate_bc_nodes(subset_nodes, comm, output_name='BothBC')
  assert PT.get_name(node) == 'BothBC'
  assert PT.get_value(node) == 'BCFarfield'
  assert PT.Subset.GridLocation(node) == 'FaceCenter'
  assert (MT.getDistribution(node, 'Index')[1] == expected_distri).all()
  assert (PT.get_child_from_name(node, 'PointList')[1][0] == expected_pl).all()

  assert PT.get_label(PT.get_node_from_path(node, 'BCDataSet/BCData')) == 'BCData_t'
  assert (PT.get_node_from_name(node, 'Data')[1] == expected_data).all()

@pytest_parallel.mark.parallel([1])
@pytest.mark.parametrize("mode", ['', 'intrazone', 'periodic', 'intraperio'])
def test_concatenate_jns(comm, mode):
  yt = """
  ZoneA Zone_t [[11, 10, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      match1 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 2, 3, 4]]:
        PointListDonor IndexArray_t [[10, 20, 30, 40]]:
      match2 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[5]]:
        PointListDonor IndexArray_t [[50]]:
  ZoneB Zone_t [[101, 100, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      match3 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[10, 20, 30, 40]]:
        PointListDonor IndexArray_t [[1, 2, 3, 4]]:
      match4 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[50]]:
        PointListDonor IndexArray_t [[5]]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  dist_tree = F2D.full_to_dist_tree(tree, comm)
  zones = PT.get_all_Zone_t(dist_tree)

  if mode in ['intrazone', 'intraperio']:
    zgc1 = PT.get_node_from_path(dist_tree, 'Base/ZoneA/ZGC')
    for gc in PT.iter_children_from_label(zgc1, 'GridConnectivity_t'):
      PT.set_value(gc, "ZoneA")
    zgc2 = PT.get_node_from_path(dist_tree, 'Base/ZoneB/ZGC')
    for gc in PT.iter_children_from_label(zgc2, 'GridConnectivity_t'):
      PT.add_child(zgc1, gc)
    PT.rm_nodes_from_name(dist_tree, PT.get_name(zones[1]))

  if mode in ['periodic', 'intraperio']:
    for gc in PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')[:2]:
      PT.new_GridConnectivityProperty({'rotation_angle': [45.,0.,0.]}, parent=gc)
    for gc in PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')[2:]:
      PT.new_GridConnectivityProperty({'rotation_angle': [-45.,0.,0.]}, parent=gc)

  GN.concatenate_jns(dist_tree, comm)

  gcs = PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')
  opp_names = [PT.get_value(PT.get_child_from_name(gc, "GridConnectivityDonorName")) for gc in gcs]
  assert len(gcs) == 2
  assert opp_names == [gc[0] for gc in gcs[::-1]]

  if mode=='intrazone':
    assert all(['.I' in gc[0] for gc in gcs])
  if mode=='periodic':
    assert all(['.P' in gc[0] for gc in gcs])
    assert len(PT.get_nodes_from_label(dist_tree, 'GridConnectivityProperty_t')) == 2

@pytest_parallel.mark.parallel([1])
def test_concatenate_jns_len32(comm):
  yt = """
  ZoneA Zone_t [[11, 10, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      match1 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 2, 3, 4]]:
        PointListDonor IndexArray_t [[10, 20, 30, 40]]:
      match2 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[5]]:
        PointListDonor IndexArray_t [[50]]:
      mergedGC444_5555 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[6]]:
        PointListDonor IndexArray_t [[7]]:
      mergedGC33AAAAA3333BB GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[7]]:
        PointListDonor IndexArray_t [[6]]:
      mergedGC2222GC GridConnectivity_t "ZoneC":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[8]]:
        PointListDonor IndexArray_t [[800]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-45., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
  ZoneB Zone_t [[101, 100, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      match3 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[10, 20, 30, 40]]:
        PointListDonor IndexArray_t [[1, 2, 3, 4]]:
      match4 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[50]]:
        PointListDonor IndexArray_t [[5]]:
  ZoneC Zone_t [[1001, 1000, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      match5 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[800]]:
        PointListDonor IndexArray_t [[8]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [45., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  dist_tree = F2D.full_to_dist_tree(tree, comm)
  zones = PT.get_all_Zone_t(dist_tree)

  GN.concatenate_jns(dist_tree, comm)

  gcs = PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')
  opp_names = [PT.get_value(PT.get_child_from_name(gc, "GridConnectivityDonorName")) for gc in gcs]
  
  assert len(gcs) == 6
  assert sorted(opp_names) == sorted([gc[0] for gc in gcs[::-1]])
  
  mergedgcs = PT.get_nodes_from_predicates(dist_tree, [lambda n: PT.get_label(n) == 'GridConnectivity_t' 
                                                                 and PT.get_name(n).startswith('mergedGC')])
  assert len(mergedgcs) == 6
  mergedgc_names     = sorted([PT.get_name(mgc).split(".")[0] for mgc in mergedgcs])
  ref_mergedgc_names = ['mergedGC0']*2 + [f'mergedGC{i}' for i in range(2223, 2227)]
  assert mergedgc_names == ref_mergedgc_names

@pytest.mark.parametrize("specified", [True, False])
@pytest_parallel.mark.parallel(3)
def test_concatenate_patch(specified, comm):
  mesh_path = os.path.join(TU.mesh_dir,'flat_plate_3d.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_path, comm)

  def tag_fam_in_bcs(dist_tree, bc_names, family_name):
    for bc_name in bc_names:
      bc_n = PT.get_node_from_name_and_label(dist_tree, bc_name, 'BC_t')
      PT.new_FamilyName(family_name, parent=bc_n)

  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [3]        ], 'WALL')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [0,1,6,8,9]], 'SYM')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [4,7]      ], 'FARFIELD')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [5]        ], 'INLET')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [2]        ], 'OUTLET')
  tag_fam_in_bcs(dist_tree, [f'ridge.{i}'   for i in range(0,20)], 'RIDGE')

  # > Create ZSR with BCRegionName
  dist_zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  bc_n = PT.get_node_from_name_and_label(dist_zone, 'surface.3', 'BC_t')
  bc_pl = PT.get_value(PT.get_child_from_name(bc_n, 'PointList'))
  PT.new_ZoneSubRegion('zsr_surface.3', bc_name='surface.3',
                       fields={'fld': bc_pl[0]}, parent=dist_zone)

  if specified:
    families = ['WALL','FARFIELD','RIDGE']
    GN.concatenate_subsets_from_families(dist_tree, comm, families)
  else:
    families = ['SYM','OUTLET','WALL','FARFIELD','INLET','RIDGE']
    GN.concatenate_subsets_from_families(dist_tree, comm)

  is_merged_bc = lambda n: PT.get_label(n)=='BC_t' and PT.get_name(n) in families
  bc_nodes = PT.get_nodes_from_predicate(dist_tree, is_merged_bc)
  assert ([PT.get_name(n) for n in bc_nodes]==families)
  for bc_n in bc_nodes:
    assert PT.get_node_from_path (bc_n, ':maia#concatenate/DirichletData/OriginalBCId') is not None
    assert PT.get_child_from_name(bc_n, 'BCNames') is not None
    assert PT.get_child_from_name(bc_n, 'BCOrdinal') is not None
  
    bcd_n = PT.get_node_from_path(bc_n, ':maia#concatenate/DirichletData')
    assert PT.get_child_from_name(bcd_n, 'OriginalBCId') is not None
    
    bcd_n = PT.get_node_from_path(bc_n, 'BCDataSet/NeumannData')
    assert PT.get_child_from_name(bcd_n, 'ParamU') is not None
    assert PT.get_child_from_name(bcd_n, 'OriginalBCId') is not None

  if specified:
    is_sym = lambda n: PT.get_label(n) in ['BC_t'] and\
                       PT.predicate.belongs_to_family(n, 'SYM', True)

    assert len(PT.get_nodes_from_predicate(dist_tree, is_sym))==5

  zsr_n = PT.get_node_from_name(dist_zone, 'zsr_surface.3')
  assert PT.get_child_from_name(zsr_n, 'BCRegionName') is None
  assert PT.Subset.GridLocation(zsr_n) == 'FaceCenter'
  zsr_pl_n = PT.get_child_from_name(zsr_n, 'PointList')
  zsr_fld_n = PT.get_child_from_name(zsr_n, 'fld')
  assert zsr_pl_n is not None
  assert np.array_equal(zsr_pl_n[1],bc_pl)
  assert np.array_equal(zsr_fld_n[1],bc_pl[0])


@pytest.mark.parametrize("specified", [True, False])
@pytest_parallel.mark.parallel(3)
def test_deconcatenate_patch(specified, comm):
  mesh_path = os.path.join(TU.mesh_dir,'flat_plate_3d.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_path, comm)

  def tag_fam_in_bcs(dist_tree, bc_names, family_name):
    for bc_name in bc_names:
      bc_n = PT.get_node_from_name_and_label(dist_tree, bc_name, 'BC_t')
      PT.new_FamilyName(family_name, parent=bc_n)

  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [3]        ], 'WALL')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [0,1,6,8,9]], 'SYM')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [4,7]      ], 'FARFIELD')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [5]        ], 'INLET')
  tag_fam_in_bcs(dist_tree, [f'surface.{i}' for i in [2]        ], 'OUTLET')
  tag_fam_in_bcs(dist_tree, [f'ridge.{i}'   for i in range(0,20)], 'RIDGE')

  dist_tree_cp = PT.deep_copy(dist_tree)

  if specified:
    families = ['WALL','FARFIELD','RIDGE']
    GN.concatenate_subsets_from_families(dist_tree, comm, families)
  else:
    families = ['WALL','SYM','FARFIELD','INLET','OUTLET','RIDGE']
    GN.concatenate_subsets_from_families(dist_tree, comm)

  GN.deconcatenate_subsets_from_families(dist_tree, comm, families)

  assert PT.is_same_tree(dist_tree, dist_tree_cp)
