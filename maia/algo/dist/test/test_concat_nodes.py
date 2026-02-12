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
        Sca DataArray_t [24.]:
  BCb BC_t "BCFarfield":
    GridLocation GridLocation_t "FaceCenter":
    PointList IndexArray_t [[10, 20, 30, 40, 50, 60, 70, 80]]:
    BCDataSet BCDataSet_t:
      BCData BCData_t:
        Data DataArray_t [1., 2., 3., 4., 5., 6., 7., 8.]:
        Sca DataArray_t [42.]:
  """
  subset_nodes_f = PT.yaml.to_nodes(yt)
  subset_nodes = [F2D.distribute_pl_node(node, comm) for node in subset_nodes_f]

  expected_distri = par_utils.uniform_distribution(4+8, comm)
  if comm.Get_size() == 1:
    expected_pl = [[1,2,3,4, 10,20,30,40,50,60,70,80]]
    expected_data = [10,20,30,40, 1.,2.,3.,4.,5.,6.,7.,8]
    expected_data_sca = [24.,24,24,24, 42,42,42,42,42,42,42,42]
  elif comm.Get_size() == 2:
    if comm.Get_rank() == 0:
      expected_pl = [[1,2, 10,20,30,40]]
      expected_data = [10,20, 1.,2.,3.,4]
      expected_data_sca = [24.,24, 42,42,42,42]
    elif comm.Get_rank() == 1:
      expected_pl = [[3,4, 50,60,70,80]]
      expected_data = [30,40, 5.,6.,7.,8]
      expected_data_sca = [24.,24, 42,42,42,42]

  if default_bcds:
    node = GN.concatenate_subset_nodes(subset_nodes, comm, output_name='BothBC', \
        additional_data_queries = ['BCDataSet/BCData/Data'])
  else:
    node = GN.concatenate_bc_nodes(subset_nodes, comm, output_name='BothBC')
  assert PT.get_name(node) == 'BothBC'
  assert PT.get_value(node) == 'BCFarfield'
  assert PT.Subset.GridLocation(node) == 'FaceCenter'
  assert (MT.Subset.distribution(node) == expected_distri).all()
  assert (PT.get_child_from_name(node, 'PointList')[1][0] == expected_pl).all()

  assert PT.get_label(PT.get_node_from_path(node, 'BCDataSet/BCData')) == 'BCData_t'
  assert (PT.get_node_from_name(node, 'Data')[1] == expected_data).all()
  if not default_bcds:
    assert (PT.get_node_from_name(node, 'Sca')[1] == expected_data_sca).all()
    assert PT.get_node_from_name(node, 'Sca')[1].dtype == np.float32

@pytest_parallel.mark.parallel([1])
def test_concatenate_jns_all_types(comm):
  yt = """
  ZoneWithVeryVeryVeryLongName Zone_t [[101, 100, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      intra1a GridConnectivity_t "ZoneWithVeryVeryVeryLongName":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 2, 3, 4]]:
        PointListDonor IndexArray_t [[5, 6, 7, 8]]:
      intra1b GridConnectivity_t "ZoneWithVeryVeryVeryLongName":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[5, 6, 7, 8]]:
        PointListDonor IndexArray_t [[1, 2, 3, 4]]:
      overset5 GridConnectivity_t "ZoneWithVeryVeryVeryLongName":
        GridLocation GridLocation_t "CellCenter":
        GridConnectivityType GridConnectivityType_t "Overset":
        PointList IndexArray_t [[11, 12]]:
      Overset0 GridConnectivity_t "ZoneWithVeryVeryVeryLongName":
        GridLocation GridLocation_t "CellCenter":
        GridConnectivityType GridConnectivityType_t "Overset":
        PointList IndexArray_t [[13, 14]]:
      userdefined6 GridConnectivity_t "ZoneWithVeryVeryVeryLongName":
        GridConnectivityType GridConnectivityType_t "UserDefined":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[31, 32]]:
      null7 GridConnectivity_t "ZoneWithVeryVeryVeryLongName":
        GridConnectivityType GridConnectivityType_t "Null":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t [[41, 42]]:
      intranomatch1a GridConnectivity_t "ZoneWithVeryVeryVeryLongName":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[24, 25, 26]]:
      intranomatch1b GridConnectivity_t "ZoneWithVeryVeryVeryLongName":
        GridConnectivityType GridConnectivityType_t "Abutting":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[27]]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  dist_tree = F2D.full_to_dist_tree(tree, comm)

  GN.concatenate_jns(dist_tree, comm)

  gcs = PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')
  
  assert all(len(PT.get_name(gc)) < 33 for gc in gcs)
  
  oversets = [gc for gc in gcs if PT.GridConnectivity.Type(gc)=="Overset"]
  assert len(oversets) == 2
  assert all(PT.get_name(overset).startswith('Overset') for overset in oversets)
  
  userdefineds = [gc for gc in gcs if PT.GridConnectivity.Type(gc)=="UserDefined"]
  assert len(userdefineds) == 1
  assert all(PT.get_name(userdefined).startswith('UserDefined') for userdefined in userdefineds)
  
  nulls  = [gc for gc in gcs if PT.GridConnectivity.Type(gc)=="Null"]
  assert len(nulls) == 1
  assert all(PT.get_name(null).startswith('Null') for null in nulls)
  
  matchs = [gc for gc in gcs if PT.GridConnectivity.Type(gc)=="Abutting1to1"]
  assert len(matchs) == 2
  assert all('GCMatch' in PT.get_name(match) for match in matchs)
  assert all(['.I' in match[0] for match in matchs])
  
  nomatchs = [gc for gc in gcs if PT.GridConnectivity.Type(gc)=="Abutting"]
  assert len(nomatchs) == 2
  assert all('GCNoMatch' in PT.get_name(nomatch) for nomatch in nomatchs)
  
  loc_suffix = {'Vertex' : 'Vtx', 'FaceCenter' : 'Face', 'CellCenter' : 'Cell'}
  assert all(PT.get_name(gc).split("@")[-1].startswith(loc_suffix[PT.Subset.GridLocation(gc)]) for gc in gcs)

@pytest_parallel.mark.parallel([1])
@pytest.mark.parametrize("type", ["Abutting1to1", "Abutting"])
@pytest.mark.parametrize("perio", [True, False])
def test_concatenate_jns_all_abutting(comm, type, perio):
  yt = """
  ZoneA Zone_t [[101, 100, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      intraperio1a GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 2, 3]]:
        PointListDonor IndexArray_t [[4, 5, 6]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio1b GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[4, 5, 6]]:
        PointListDonor IndexArray_t [[1, 2, 3]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio2a GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[10, 11, 12]]:
        PointListDonor IndexArray_t [[13, 14, 15]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio2b GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[13, 14, 15]]:
        PointListDonor IndexArray_t [[10, 11, 12]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio3a GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t [[21, 22, 23, 24]]:
        PointListDonor IndexArray_t [[25, 26, 27, 28]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio3b GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t [[25, 26, 27, 28]]:
        PointListDonor IndexArray_t [[21, 22, 23, 24]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio1a GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[18]]:
        PointListDonor IndexArray_t [[118]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio2a GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[19]]:
        PointListDonor IndexArray_t [[119]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio2b GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[20]]:
        PointListDonor IndexArray_t [[120]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio3a GridConnectivity_t "ZoneC":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[21]]:
        PointListDonor IndexArray_t [[221]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
  ZoneB Zone_t [[201, 200, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      perio1b GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[118]]:
        PointListDonor IndexArray_t [[18]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio2a GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[119]]:
        PointListDonor IndexArray_t [[19]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      perio2b GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[120]]:
        PointListDonor IndexArray_t [[20]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
  ZoneC Zone_t [[301, 300, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      perio3b GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[221]]:
        PointListDonor IndexArray_t [[21]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [-20., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  dist_tree = F2D.full_to_dist_tree(tree, comm)
  for gc in PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t'):
    if type =="Abutting":
      PT.update_child(gc, "GridConnectivityType", value=type)
      PT.rm_node_from_path(gc, "PointListDonor")
    if not perio:
      PT.rm_node_from_path(gc, "GridConnectivityProperty")

  GN.concatenate_jns(dist_tree, comm)
  
  gcs    = PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')
  perios = PT.get_nodes_from_label(dist_tree, 'GridConnectivityProperty_t')
  
  assert all(len(PT.get_name(gc)) <= 32 for gc in gcs)
  
  nb_perios = 0
  if type == "Abutting1to1":
    nb_gcs    = 8
    gc_partial_name = 'GCMatch'
    opp_names = [PT.get_str_value(PT.find_child_from_name(gc, "GridConnectivityDonorName")) for gc in gcs]
    assert sorted(opp_names) == sorted([PT.get_name(gc) for gc in gcs[::-1]])
    if perio:
      nb_gcs    = 10
      nb_perios = 10
      assert all(['.P' in PT.get_name(gc) for gc in gcs])
  else:
    gc_partial_name = 'GCNoMatch'
    nb_gcs = 10 
    if perio:
      nb_perios = 10
      assert all(['.P' in PT.get_name(gc) for gc in gcs])
  
  assert all(gc_partial_name in PT.get_name(gc) for gc in gcs)
  
  assert len(gcs)    == nb_gcs
  assert len(perios) == nb_perios
  
  if perio:
    if type == "Abutting1to1":
      suffix_to_rot_angle = {".P0": [ 10., 0., 0.],
                             ".P1": [-10., 0., 0.],
                             ".P2": [ 20., 0., 0.],
                             ".P3": [-20., 0., 0.]}
    else:
      suffix_to_rot_angle = {".P0": [-10., 0., 0.],
                             ".P1": [ 10., 0., 0.],
                             ".P2": [ 20., 0., 0.],
                             ".P3": [-20., 0., 0.]}
    for gc in gcs:
      suffix = f'.P{PT.get_name(gc).split(".P")[-1]}'
      assert np.array_equal(PT.GridConnectivity.periodic_values(gc)[1], suffix_to_rot_angle[suffix])



@pytest_parallel.mark.parallel([1])
def test_concatenate_jns_with_same_perio(comm):
  yt = """
  ZoneA Zone_t [[101, 100, 0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      intraperio1a GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 2, 3]]:
        PointListDonor IndexArray_t [[4, 5, 6]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio1b GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[4, 5, 6]]:
        PointListDonor IndexArray_t [[1, 2, 3]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio2a GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[7, 8]]:
        PointListDonor IndexArray_t [[9, 10]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
      intraperio2b GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[9, 10]]:
        PointListDonor IndexArray_t [[7, 8]]:
        GridConnectivityProperty GridConnectivityProperty_t:
          Periodic Periodic_t:
            RotationAngle DataArray_t R4 [10., 0., 0.]:
            RotationCenter DataArray_t R4 [0., 0., 0.]:
            Translation DataArray_t R4 [0., 0., 0.]:
  """
  tree = PT.yaml.to_cgns_tree(yt)
  dist_tree = F2D.full_to_dist_tree(tree, comm)

  GN.concatenate_jns(dist_tree, comm)
  
  gcs    = PT.get_nodes_from_label(dist_tree, 'GridConnectivity_t')
  perios = PT.get_nodes_from_label(dist_tree, 'GridConnectivityProperty_t')
  
  assert all(len(PT.get_name(gc)) <= 32 for gc in gcs)
  
  assert all(['.P' in PT.get_name(gc) for gc in gcs])
  
  assert all('GCMatch' in PT.get_name(gc) for gc in gcs)
  
  assert len(gcs)    == 2
  assert len(perios) == 2
  
  suffix_to_rot_angle = {".P0": [ 10., 0., 0.],
                         ".P1": [ 10., 0., 0.]}
  for gc in gcs:
    assert np.array_equal(PT.GridConnectivity.periodic_values(gc)[1], [ 10., 0., 0.])

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

@pytest_parallel.mark.parallel(1)
def test_deconcatenate_patch_zsr(comm):
  yt = """
  Base CGNSBase_t [3, 3]:
    ZoneA Zone_t [[101, 100, 0]]:
      ZoneType ZoneType_t "Unstructured":
      ZoneBC ZoneBC_t:
        BCA1 BC_t "FamilySpecified":
          GridLocation GridLocation_t "FaceCenter":
          FamilyName FamilyName_t "FAM":
          PointList IndexArray_t [[1, 2, 3, 4]]:
        BCA2 BC_t "FamilySpecified":
          GridLocation GridLocation_t "FaceCenter":
          FamilyName FamilyName_t "FAM":
          PointList IndexArray_t [[5, 6, 7, 8]]:
      ZSR_BCA1 ZoneSubRegion_t:
        GridLocation GridLocation_t "FaceCenter":
        BCRegionName Descriptor_t "BCA1":
        FakeData DataArray_t [1., 2., 3., 4.]:
      ZSR_BCA2 ZoneSubRegion_t:
        GridLocation GridLocation_t "FaceCenter":
        BCRegionName Descriptor_t "BCA2":
        FakeData DataArray_t [5., 6., 7., 8.]:
    ZoneB Zone_t [[101, 100, 0]]:
      ZoneType ZoneType_t "Unstructured":
      ZoneBC ZoneBC_t:
        BCB1 BC_t "FamilySpecified":
          GridLocation GridLocation_t "FaceCenter":
          FamilyName FamilyName_t "FAM":
          PointList IndexArray_t [[11, 12, 13, 14]]:
        BCB2 BC_t "FamilySpecified":
          GridLocation GridLocation_t "FaceCenter":
          FamilyName FamilyName_t "FAM":
          PointList IndexArray_t [[15, 16, 17, 18]]:
      ZSR_BCB1 ZoneSubRegion_t:
        GridLocation GridLocation_t "FaceCenter":
        BCRegionName Descriptor_t "BCB1":
        FakeData DataArray_t [11., 12., 13., 14.]:
      ZSR_BCB2 ZoneSubRegion_t:
        GridLocation GridLocation_t "FaceCenter":
        BCRegionName Descriptor_t "BCB2":
        FakeData DataArray_t [15., 16., 17., 18.]:
        FakeData2 DataArray_t [15.5, 16.5, 17.5, 18.5]:
    FAM Family_t:
      FamilyBC FamilyBC_t "Null":
  """
  
  tree = PT.yaml.to_cgns_tree(yt)
  dist_tree = F2D.full_to_dist_tree(tree, comm)
  dist_tree_cp = PT.deep_copy(dist_tree)

  GN.concatenate_subsets_from_families(dist_tree, comm)
  dist_tree_concat = PT.deep_copy(dist_tree)
  assert len(PT.get_nodes_from_label(dist_tree, 'ZoneSubRegion_t')) == 3

  GN.deconcatenate_subsets_from_families(dist_tree, comm)
  assert PT.is_same_tree(dist_tree, dist_tree_cp)

