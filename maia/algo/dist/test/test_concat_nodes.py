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
def test_concatenate_jns_1to1(comm, mode):
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
  
  len_names = np.array([len(name) for name in opp_names])
  assert np.all(len_names < 33)

  if mode=='intrazone':
    assert all(['.I' in gc[0] for gc in gcs])
  if mode=='periodic':
    assert all(['.P' in gc[0] for gc in gcs])
    assert len(PT.get_nodes_from_label(dist_tree, 'GridConnectivityProperty_t')) == 2

@pytest_parallel.mark.parallel([1])
def test_concatenate_jns_all_types(comm):
  yt = """
  ZoneA Zone_t [[101, 100, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC1 ZoneGridConnectivity_t:
      intra1 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 2, 3, 4]]:
        PointListDonor IndexArray_t [[5, 6, 7, 8]]:
      match2 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[9]]:
        PointListDonor IndexArray_t [[109]]:
      intraperio3 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[10, 11, 12]]:
        PointListDonor IndexArray_t [[13, 14, 15]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intra4 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[17]]:
        PointListDonor IndexArray_t [[16]]:
      intra5 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[16]]:
        PointListDonor IndexArray_t [[17]]:
      intra6 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[5, 6, 7, 8]]:
        PointListDonor IndexArray_t [[1, 2, 3, 4]]:
      perio7 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[18]]:
        PointListDonor IndexArray_t [[118]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio8 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[19]]:
        PointListDonor IndexArray_t [[119]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio9 GridConnectivity_t "ZoneC":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[20]]:
        PointListDonor IndexArray_t [[220]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio10 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[13, 14, 15]]:
        PointListDonor IndexArray_t [[10, 11, 12]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
    ZGC2 ZoneGridConnectivity_t:
      intraperio11 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t [[21, 22, 23, 24]]:
        PointListDonor IndexArray_t [[25, 26, 27, 28]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio12 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t [[25, 26, 27, 28]]:
        PointListDonor IndexArray_t [[21, 22, 23, 24]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
  ZoneB Zone_t [[201, 200, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      intra1 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[103, 104]]:
        PointListDonor IndexArray_t [[101, 102]]:
      match2 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[109]]:
        PointListDonor IndexArray_t [[9]]:
      perio3 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[118]]:
        PointListDonor IndexArray_t [[18]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio4 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[119]]:
        PointListDonor IndexArray_t [[19]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intra5 GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[101, 102]]:
        PointListDonor IndexArray_t [[103, 104]]:
  ZoneC Zone_t [[301, 300, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      perio1 GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[220]]:
        PointListDonor IndexArray_t [[20]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      nomatch2 GridConnectivity_t "ZoneC":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[221, 222]]:
      nomatch3 GridConnectivity_t "ZoneC":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[223]]:
      nomatch4 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[224, 225]]:
      nomatch5 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[226]]:
  ZoneD Zone_t [[101, 100, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC1 ZoneGridConnectivity_t:
      intra1 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 2, 3, 4]]:
        PointListDonor IndexArray_t [[5, 6, 7, 8]]:
      nomatch2 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[21, 22]]:
      nomatch3 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[23]]:
      intra4 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[5, 6, 7, 8]]:
        PointListDonor IndexArray_t [[1, 2, 3, 4]]:
      overset5 GridConnectivity_t "ZoneD":
        GridLocation GridLocation_t "CellCenter":
        GridConnectivityType GridConnectivityType_t "Overset":
        PointList IndexArray_t [[11, 12]]:
      OversetHole0 GridConnectivity_t "ZoneD":
        GridLocation GridLocation_t "CellCenter":
        GridConnectivityType GridConnectivityType_t "Overset":
        PointList IndexArray_t [[13, 14]]:
      userdefined6 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "UserDefined":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[31, 32]]:
      null7 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Null":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t [[41, 42]]:
      nomatch8 GridConnectivity_t "ZoneC":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[24, 25, 26, 27]]:
      nomatch9 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[28]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [0., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., -1.]:
      nomatch10 GridConnectivity_t "ZoneD":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[29]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [0., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 1.]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  dist_tree = F2D.full_to_dist_tree(tree, comm)
  zones = PT.get_all_Zone_t(dist_tree)

  GN.concatenate_jns(dist_tree, comm)

  gcs = PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')
  gcs_1to1 = [gc for gc in gcs if PT.GridConnectivity.is1to1(gc)]
  opp_names_1to1 = [PT.get_value(PT.get_child_from_name(gc, "GridConnectivityDonorName")) for gc in gcs_1to1]
  
  assert sorted(opp_names_1to1) == sorted([gc[0] for gc in gcs_1to1[::-1]])
  
  len_names = np.array([len(gc[0]) for gc in gcs])


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
