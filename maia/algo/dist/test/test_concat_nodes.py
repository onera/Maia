import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia                import npy_pdm_gnum_dtype as pdm_dtype
from maia.pytree.yaml    import parse_yaml_cgns
from maia.utils          import par_utils
from maia.factory        import full_to_dist as F2D

from maia.algo.dist import concat_nodes as GN

@pytest_parallel.mark.parallel([1,2])
def test_concatenate_subset_nodes(comm):
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
  subset_nodes_f = parse_yaml_cgns.to_nodes(yt)
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

  node = GN.concatenate_subset_nodes(subset_nodes, comm, output_name='BothBC', \
      additional_data_queries = ['BCDataSet/BCData/Data'])
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
  tree = parse_yaml_cgns.to_cgns_tree(yt)
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
