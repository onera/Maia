import pytest
import pytest_parallel

import numpy as np
import maia.pytree        as PT

from maia.pytree.yaml   import parse_yaml_cgns
from maia.factory.partitioning import post_split as PS

def test_pl_idx_ijk():
  yt = """
Zone Zone_t [[10,9,0], [10,9,0], [10,9,0]]:
  ZoneType ZoneType_t "Structured":
  ZoneBC ZoneBC_t:
    BCa BC_t:
      PointList IndexArray_t [[1,1,1,1], [1,2,3,5], [1,1,1,1]]:
      GridLocation GridLocation_t "IFaceCenter":
    BCb BC_t: #This one is ignored
      PointRange IndexRange_t [[1,9], [10,10], [1,9]]:
      GridLocation GridLocation_t "JFaceCenter":
  """
  zone = parse_yaml_cgns.to_node(yt)
  bca = PT.get_node_from_name(zone, 'BCa')
  bcb = PT.get_node_from_name(zone, 'BCb')

  PS.pl_as_idx(zone, 'ZoneBC_t/BC_t')
  assert np.array_equal(PT.get_node_from_name(bca, 'PointList')[1], [[1,11,21,41]])
  assert PT.get_node_from_name(bcb, 'PointList') is None

  PS.pl_as_ijk(zone, 'ZoneBC_t/BC_t')
  assert np.array_equal(PT.get_node_from_name(bca, 'PointList')[1], [[1,1,1,1], [1,2,3,5], [1,1,1,1]])
  assert PT.get_node_from_name(bcb, 'PointList') is None

  # Wrong location
  PT.set_value(PT.get_node_from_name(bca, 'GridLocation'), 'Vertexx')
  with pytest.raises(ValueError):
    PS.pl_as_idx(zone, 'ZoneBC_t/BC_t')
  # Wrong zone type
  PT.set_value(PT.get_node_from_name(zone, 'ZoneType'), 'Unstructured')
  with pytest.raises(AssertionError):
    PS.pl_as_idx(zone, 'ZoneBC_t/BC_t')

@pytest_parallel.mark.parallel(2)
def test_hybrid_jns_as_ijk(comm):
  if comm.Get_rank() == 0:
    pt = """
    ZoneU.P0.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      ZGC ZoneGridConnectivity_t:
        GCU GridConnectivity_t "ZoneS.P1.N0":
          GridConnectivityType GridConnectivityProperty_t "Abutting1to1":
          PointList IndexArray_t [[101,102,103,104]]:
          PointListDonor IndexArray_t [[1,11,21,41]]:
          GridLocation GridLocation_t "FaceCenter":
          GridConnectivityDonorName Descriptor_t "GCS":
    """
  elif comm.Get_rank() == 1:
    pt = """
    ZoneS.P1.N0 Zone_t [[10,9,0], [10,9,0], [10,9,0]]:
      ZoneType ZoneType_t "Structured":
      ZGC ZoneGridConnectivity_t:
        GCS GridConnectivity_t "ZoneU.P0.N0":
          GridConnectivityType GridConnectivityProperty_t "Abutting1to1":
          PointList IndexArray_t [[1,11,21,41]]:
          PointListDonor IndexArray_t [[101,102,103,104]]:
          GridLocation GridLocation_t "IFaceCenter":
          GridConnectivityDonorName Descriptor_t "GCS":
    """
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)
  PS.hybrid_jns_as_ijk(part_tree, comm)

  expected_pl  = np.array([[101,102,103,104]])
  expected_pld = np.array([[1,1,1,1], [1,2,3,5], [1,1,1,1]])
  if comm.Get_rank() == 1:
    expected_pl, expected_pld = expected_pld, expected_pl

  assert np.array_equal(PT.get_node_from_name(part_tree, 'PointList')[1], expected_pl)
  assert np.array_equal(PT.get_node_from_name(part_tree, 'PointListDonor')[1], expected_pld)

def test_copy_additional_nodes():
  dt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
      PointList IndexArray_t:
      GridLocation GridLocation_t 'FaceCenter':
      FamilyName FamilyName_t "FAM":
      .Solver#BC UserDefinedData_t:
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t:
      PointList IndexArray_t:
      GridConnectivityDonorName Descriptor_t "toto":
      GridConnectivityProperty GridConnectivityProperty_t:
        Periodic Periodic_t:
          Translation DataArray_t [1,1,1]:
"""
  pt = """
Zone.P2.N3 Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
      PointList IndexArray_t:
      GridLocation GridLocation_t 'FaceCenter':
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t:
      PointList IndexArray_t:
"""

  dist_zone = parse_yaml_cgns.to_node(dt)
  part_zone = parse_yaml_cgns.to_node(pt)
  PS.copy_additional_nodes(dist_zone, part_zone)
  assert PT.get_label(PT.get_node_from_name(dist_zone, '.Solver#BC')) == PT.get_label(PT.get_node_from_name(part_zone, '.Solver#BC'))
  assert PT.get_value(PT.get_node_from_name(dist_zone, 'GridConnectivityDonorName')) == PT.get_value(PT.get_node_from_name(part_zone, 'GridConnectivityDonorName'))
  assert (PT.get_value(PT.get_node_from_name(dist_zone, 'Translation')) == \
          PT.get_value(PT.get_node_from_name(part_zone, 'Translation'))).all()

def test_update_zone_pointers():
  part_tree = parse_yaml_cgns.to_cgns_tree("""
  ZoneA.P0.N0 Zone_t:
  ZoneA.P1.N0 Zone_t:
  BaseIterativeData BaseIterativeData_t [2]:
    TimeValues DataArray_t [0., 1.]:
    ZonePointers DataArray_t [["ZoneA"], ["ZoneA", "ZoneB"]]:
    NumberOfZones DataArray_t [1,2]:
  """)
  PS.update_zone_pointers(part_tree)
  assert (PT.get_value(PT.get_node_from_name(part_tree, 'NumberOfZones')) == [2,2]).all()
  assert PT.get_value(PT.get_node_from_name(part_tree, 'ZonePointers'))[0] == ["ZoneA.P0.N0", "ZoneA.P1.N0"]
  assert PT.get_value(PT.get_node_from_name(part_tree, 'ZonePointers'))[1] == ["ZoneA.P0.N0", "ZoneA.P1.N0"]

  part_tree = parse_yaml_cgns.to_cgns_tree("""
  ZoneC.P0.N0 Zone_t:
  BaseIterativeData BaseIterativeData_t [2]:
    TimeValues DataArray_t [0., 1.]:
    ZonePointers DataArray_t [["ZoneA"], ["ZoneA", "ZoneB"]]:
    NumberOfZones DataArray_t [1,2]:
  """)
  PS.update_zone_pointers(part_tree)
  assert PT.get_value(PT.get_node_from_name(part_tree, 'ZonePointers')) == [[], []]

  part_tree = parse_yaml_cgns.to_cgns_tree("""
  ZoneA.P0.N0 Zone_t:
  BaseIterativeData BaseIterativeData_t [2]:
    TimeValues DataArray_t [0., 1.]:
  """)
  PS.update_zone_pointers(part_tree)
  assert PT.get_node_from_name(part_tree, 'ZonePointers') is None

def test_generate_related_zsr():
  dt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
      PointList IndexArray_t:
      GridLocation GridLocation_t "FaceCenter":
      FamilyName FamilyName_t "FAM":
      .Solver#BC UserDefinedData_t:
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t:
      PointList IndexArray_t:
      GridConnectivityDonorName Descriptor_t "toto":
      GridConnectivityProperty GridConnectivityProperty_t:
        Periodic Periodic_t:
          Translation DataArray_t [1,1,1]:
  ZSR_BC ZoneSubRegion_t:
    SomeRandomDescriptor Descriptor_t "Let's go party":
    BCRegionName Descriptor_t "BC":
  ZSR_GC ZoneSubRegion_t:
    GridConnectivityRegionName Descriptor_t "GC":
"""
  pt = """
Zone.P2.N3 Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
      PointList IndexArray_t:
      GridLocation GridLocation_t "FaceCenter":
  ZGC ZoneGridConnectivity_t:
    GC.0 GridConnectivity_t:
      PointList IndexArray_t:
    GC.1 GridConnectivity_t:
      PointList IndexArray_t:
    JN.P2.N3.LT.P1.N0 GridConnectivity_t: # Simulate an intra JN
"""

  dist_zone = parse_yaml_cgns.to_node(dt)
  part_zone = parse_yaml_cgns.to_node(pt)
  PS.generate_related_zsr(dist_zone, part_zone)
  assert PT.is_same_node(PT.get_node_from_name(dist_zone, 'ZSR_BC'), PT.get_node_from_name(part_zone, 'ZSR_BC'))
  assert PT.get_value(PT.get_node_from_predicates(part_zone, 'ZSR_GC.0/Descriptor_t'))=='GC.0'
  assert PT.get_value(PT.get_node_from_predicates(part_zone, 'ZSR_GC.1/Descriptor_t'))=='GC.1'

def test_split_original_joins():
  pt = """
ZoneB.P1.N0 Zone_t:
  ZGC ZoneGridConnectivity_t:
    matchBA GridConnectivity_t "ZoneA":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      PointList IndexArray_t [[8,12,9,20,1]]:
      PointListDonor IndexArray_t [[11,21,8,25,13]]:
      Donor IndexArray_t [[0,0],[0,0],[1,0],[0,0],[1,0]]:
      GridConnectivityDonorName Descriptor_t "matchAB":
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t [5,3,1,2,6]:
"""
  p_tree = parse_yaml_cgns.to_cgns_tree(pt)
  p_zone = PT.get_all_Zone_t(p_tree)[0]
  PS.split_original_joins(p_tree)

  assert PT.get_node_from_name(p_zone, 'matchBA') is None
  assert len(PT.get_nodes_from_name(p_zone, 'matchBA*')) == 2
  new_jn0 = PT.get_node_from_name(p_zone, 'matchBA.0')
  new_jn1 = PT.get_node_from_name(p_zone, 'matchBA.1')
  assert PT.get_value(new_jn0) == 'ZoneA.P0.N0'
  assert PT.get_value(new_jn1) == 'ZoneA.P1.N0'

  assert (PT.get_child_from_name(new_jn0, 'PointList')[1] == [8,12,20]).all()
  assert (PT.get_child_from_name(new_jn0, 'PointListDonor')[1] == [11,21,25]).all()
  assert (PT.get_node_from_name (new_jn0, 'Index')[1] == [5,3,2]).all()
  assert (PT.get_child_from_name(new_jn1, 'PointList')[1] == [9,1]).all()
  assert (PT.get_child_from_name(new_jn1, 'PointListDonor')[1] == [8,13]).all()
  assert (PT.get_node_from_name (new_jn1, 'Index')[1] == [1,6]).all()

@pytest_parallel.mark.parallel(2)
def test_update_gc_donor_name(comm):
  if comm.Get_rank() == 0:
    pt = """
    Base CGNSBase_t:
      ZoneA.P0.N0 Zone_t:
        ZGC ZoneGridConnectivity_t:
          matchAB.0 GridConnectivity_t "ZoneB.P0.N0":
            GridConnectivityType GridConnectivityType_t "Abutting1to1":
            GridConnectivityDonorName Descriptor_t "matchBA":
          matchAB.1 GridConnectivity_t "ZoneB.P1.N0":
            GridConnectivityType GridConnectivityType_t "Abutting1to1":
            GridConnectivityDonorName Descriptor_t "matchBA":
      ZoneB.P0.N0 Zone_t:
        ZGC ZoneGridConnectivity_t:
          matchBA.0 GridConnectivity_t "ZoneA.P0.N0":
            GridConnectivityType GridConnectivityType_t "Abutting1to1":
            GridConnectivityDonorName Descriptor_t "matchAB":
    """
    expected = ['matchBA.0', 'matchBA.0', 'matchAB.0']
  elif comm.Get_rank() == 1:
    pt = """
    Base CGNSBase_t:
      ZoneB.P1.N0 Zone_t:
        ZGC ZoneGridConnectivity_t:
          matchBA.0 GridConnectivity_t "ZoneA.P0.N0":
            GridConnectivityType GridConnectivityType_t "Abutting1to1":
            GridConnectivityDonorName Descriptor_t "matchAB":
    """
    expected = ['matchAB.1']
  p_tree = parse_yaml_cgns.to_cgns_tree(pt)
  PS.update_gc_donor_name(p_tree, comm)

  assert [PT.get_value(n) for n in PT.get_nodes_from_name(p_tree, 'GridConnectivityDonorName')] == expected
