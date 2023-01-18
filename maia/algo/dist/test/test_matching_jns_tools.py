from pytest_mpi_check._decorator import mark_mpi_test

import mpi4py.MPI as MPI
import numpy as np
import maia.pytree as PT

from maia.pytree.yaml import parse_yaml_cgns
from maia.algo.dist import matching_jns_tools as MJT
from maia.factory import full_to_dist

class Test_compare_pointrange():
  def test_ok(self):
    jn1 = PT.new_GridConnectivity1to1(point_range      =[[17,17],[3,9],[1,5]], point_range_donor=[[7,1],[9,9],[5,1]])
    jn2 = PT.new_GridConnectivity1to1(point_range_donor=[[17,17],[3,9],[1,5]], point_range      =[[7,1],[9,9],[5,1]])
    assert(MJT._compare_pointrange(jn1, jn2) == True)
  def test_ko(self):
    jn1 = PT.new_GridConnectivity1to1(point_range      =[[17,17],[3,9],[1,5]], point_range_donor=[[7,1],[9,9],[5,1]])
    jn2 = PT.new_GridConnectivity1to1(point_range_donor=[[17,17],[3,9],[1,5]], point_range      =[[1,7],[9,9],[1,5]])
    assert(MJT._compare_pointrange(jn1, jn2) == False)
  def test_empty(self):
    jn1 = PT.new_GridConnectivity1to1(point_range      =np.empty((3,2), np.int32), point_range_donor=np.empty((3,2), np.int32))
    jn2 = PT.new_GridConnectivity1to1(point_range_donor=np.empty((3,2), np.int32), point_range      =np.empty((3,2), np.int32))
    assert(MJT._compare_pointrange(jn1, jn2) == True)

class Test_compare_pointlist():
  def test_ok(self):
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[12,14,16,18]], point_list_donor=[[9,7,5,3]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[12,14,16,18]], point_list      =[[9,7,5,3]])
    assert(MJT._compare_pointlist(jn1, jn2) == True)
  def test_ko(self):
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =[[12,14,16,18]], point_list_donor=[[9,7,5,3]])
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=[[12,14,16,18]], point_list      =[[3,9,5,7]])
    assert(MJT._compare_pointlist(jn1, jn2) == False)
  def test_empty(self):
    jn1 = PT.new_GridConnectivity(type='Abutting1to1', point_list      =np.empty((1,0), np.int32), point_list_donor=np.empty((1,0), np.int32))
    jn2 = PT.new_GridConnectivity(type='Abutting1to1', point_list_donor=np.empty((1,0), np.int32), point_list      =np.empty((1,0), np.int32))
    assert(MJT._compare_pointlist(jn1, jn2) == True)

@mark_mpi_test([1,3])
def test_add_joins_donor_name(sub_comm):
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t [[27,8,0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity1to1_t "ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [[1,4,7,10]]:    # HERE
        PointListDonor IndexArray_t [[13,16,7,10]]:  # HERE
  ZoneB Zone_t [[27,8,0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity_t "ZoneA":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [[13,16,7,10]]:  # HERE
        PointListDonor IndexArray_t [[1,4,7,10]]:    # HERE
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
      matchBC1 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [[32,34]]:
        PointListDonor IndexArray_t [[1,3]]:
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
      matchBC2 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[33,35]]:
        PointListDonor IndexArray_t [[2,4]]:
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
Base1 CGNSBase_t [3,3]:
  ZoneC Zone_t [[18,4,0]]:
    ZoneType ZoneType_t "Unstructured":
    ZGC ZoneGridConnectivity_t:
      matchCB2 GridConnectivity1to1_t "Base0/ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [[2,4]]:         # HERE
        PointListDonor IndexArray_t [[33,35]]:       # HERE
      matchCB1 GridConnectivity1to1_t "Base0/ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1,3]]:
        PointListDonor IndexArray_t [[32,34]]:
"""
  full_tree = parse_yaml_cgns.to_cgns_tree(yt)
  dist_tree = full_to_dist.distribute_tree(full_tree, sub_comm)

  MJT.add_joins_donor_name(dist_tree, sub_comm)

  expected_donor_names = ['matchBA', 'matchAB', 'matchCB1', 'matchCB2', 'matchBC2', 'matchBC1']
  query = lambda n : PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  for i, jn in enumerate(PT.iter_nodes_from_predicate(dist_tree, query)):
    assert PT.get_value(PT.get_child_from_name(jn, 'GridConnectivityDonorName')) == expected_donor_names[i]

@mark_mpi_test(1)
def test_force(sub_comm):
  yt = """
Base0 CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [1,4,7,10]:
        PointListDonor IndexArray_t [13,16,7,10]:
        GridConnectivityDonorName Descriptor_t "WrongOldValue":
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity_t "ZoneA":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        PointList IndexArray_t [13,16,7,10]:
        PointListDonor IndexArray_t [1,4,7,10]:
        GridConnectivityDonorName Descriptor_t "WrongOldValue":
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)
  jn_donor_path = 'Base0/ZoneA/ZGC/matchAB/GridConnectivityDonorName'
  assert PT.get_value(PT.get_node_from_path(dist_tree, jn_donor_path)) == 'WrongOldValue'
  MJT.add_joins_donor_name(dist_tree, sub_comm)
  assert PT.get_value(PT.get_node_from_path(dist_tree, jn_donor_path)) == 'WrongOldValue'
  MJT.add_joins_donor_name(dist_tree, sub_comm, force=True)
  assert PT.get_value(PT.get_node_from_path(dist_tree, jn_donor_path)) == 'matchBA'

class Test_gcdonorname_utils:
  dt = """
Base CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      perio1 GridConnectivity_t 'ZoneA':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridConnectivityDonorName Descriptor_t 'perio2':
        PointList IndexArray_t [[1,3]]:
      perio2 GridConnectivity_t 'Base/ZoneA':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridConnectivityDonorName Descriptor_t 'perio1':
        PointList IndexArray_t [[2,4]]:
      match1 GridConnectivity_t 'ZoneB':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridConnectivityDonorName Descriptor_t 'match2':
        PointList IndexArray_t [[10,100]]:
        GridLocation GridLocation_t "FaceCenter":
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      match2 GridConnectivity_t 'ZoneA':
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridConnectivityDonorName Descriptor_t 'match1':
        PointList IndexArray_t [[-100,-10]]:
        GridLocation GridLocation_t "FaceCenter":
  """
  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)

  def test_get_jn_donor_path(self):
    assert MJT.get_jn_donor_path(self.dist_tree, 'Base/ZoneA/ZGC/perio2') == 'Base/ZoneA/ZGC/perio1'
    assert MJT.get_jn_donor_path(self.dist_tree, 'Base/ZoneA/ZGC/match1') == 'Base/ZoneB/ZGC/match2'

  def test_update_jn_name(self):
    dist_tree = PT.deep_copy(self.dist_tree)
    ini_gc = PT.get_node_from_path(dist_tree, 'Base/ZoneA/ZGC/perio2')
    MJT.update_jn_name(dist_tree, 'Base/ZoneA/ZGC/perio2', 'PERIO2')
    assert ini_gc[0] == 'PERIO2'
    assert PT.get_value(PT.get_node_from_path(dist_tree, 'Base/ZoneA/ZGC/perio1/GridConnectivityDonorName')) == 'PERIO2'

  def test_get_matching_jns(self):
    pathes = MJT.get_matching_jns(self.dist_tree)
    assert pathes[0] == ('Base/ZoneA/ZGC/perio1', 'Base/ZoneA/ZGC/perio2')
    assert pathes[1] == ('Base/ZoneA/ZGC/match1', 'Base/ZoneB/ZGC/match2')
    pathes = MJT.get_matching_jns(self.dist_tree, 'Vertex')
    assert len(pathes) == 1
    assert pathes[0] == ('Base/ZoneA/ZGC/perio1', 'Base/ZoneA/ZGC/perio2')

  def test_match_jn_from_ordinals(self):
    dist_tree = PT.deep_copy(self.dist_tree)
    MJT.copy_donor_subset(dist_tree)
    expected_pl_opp = [[2,4], [1,3], [-100,-10], [10,100]]
    for i, jn in enumerate(PT.iter_nodes_from_label(dist_tree, 'GridConnectivity_t')):
      assert (PT.get_child_from_name(jn, 'PointListDonor')[1] == expected_pl_opp[i]).all()

  def test_store_interfaces_ids(self):
    dist_tree = PT.deep_copy(self.dist_tree)
    MJT.store_interfaces_ids(dist_tree)
    expected_id = [1,1,2,2]
    expected_pos = [0,1,0,1]
    for i, jn in enumerate(PT.iter_nodes_from_label(dist_tree, 'GridConnectivity_t')):
      assert (PT.get_child_from_name(jn, 'DistInterfaceId')[1] == expected_id[i]).all()
      assert (PT.get_child_from_name(jn, 'DistInterfaceOrd')[1] == expected_pos[i]).all()


def test_clear_interfaces_ids():
  yt = """
Base0 CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        DistInterfaceId DataArray_t [1]:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity_t "ZoneA":
        PointList IndexArray_t [[13,16,7,10]]:
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)
  MJT.clear_interface_ids(dist_tree)
  assert PT.get_node_from_name(dist_tree, 'DistInterfaceId')  is None
  assert PT.get_node_from_name(dist_tree, 'DistInterfaceOrd') is None
