from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np
import maia.pytree        as PT

from maia.pytree.yaml   import parse_yaml_cgns
from maia.factory.partitioning import post_split as PS

def test_copy_additional_nodes():
  dt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
      PointList IndexArray_t:
      GridLocation GridLocation_t:
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
      GridLocation GridLocation_t:
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

def test_split_original_joins():
  pt = """
ZoneB.P1.N0 Zone_t:
  ZGC ZoneGridConnectivity_t:
    matchBA GridConnectivity_t "ZoneA":
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

@mark_mpi_test(2)
def test_update_gc_donor_name(sub_comm):
  if sub_comm.Get_rank() == 0:
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
  elif sub_comm.Get_rank() == 1:
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
  PS.update_gc_donor_name(p_tree, sub_comm)

  assert [PT.get_value(n) for n in PT.get_nodes_from_name(p_tree, 'GridConnectivityDonorName')] == expected
