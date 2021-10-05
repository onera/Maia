import numpy as np
import Converter.Internal as I

from maia.utils        import parse_yaml_cgns
from maia.partitioning import post_split as PS

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
      Ordinal UserDefinedData_t 2:
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
  assert I.getType(I.getNodeFromName(dist_zone, '.Solver#BC')) == I.getType(I.getNodeFromName(part_zone, '.Solver#BC'))
  assert I.getValue(I.getNodeFromName(dist_zone, 'Ordinal')) == I.getValue(I.getNodeFromName(part_zone, 'Ordinal'))
  assert (I.getValue(I.getNodeFromName(dist_zone, 'Translation')) == \
          I.getValue(I.getNodeFromName(part_zone, 'Translation'))).all()

def test_split_original_joins():
  pt = """
ZoneB.P1.N0 Zone_t:
  ZGC ZoneGridConnectivity_t:
    matchBA GridConnectivity_t "ZoneA":
      PointList IndexArray_t [[8,12,9,20,1]]:
      PointListDonor IndexArray_t [[11,21,8,25,13]]:
      Donor IndexArray_t [[0,0],[0,0],[1,0],[0,0],[1,0]]:
      Ordinal UserDefinedData_t [2]:
      OrdinalOpp UserDefinedData_t [1]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t [5,3,1,2,6]:
"""
  p_zone = parse_yaml_cgns.to_node(pt)
  PS.split_original_joins(p_zone)

  assert I.getNodeFromName(p_zone, 'matchBA') is None
  assert len(I.getNodesFromName(p_zone, 'matchBA*')) == 2
  new_jn0 = I.getNodeFromName(p_zone, 'matchBA.0')
  new_jn1 = I.getNodeFromName(p_zone, 'matchBA.1')
  assert I.getValue(new_jn0) == 'ZoneA.P0.N0'
  assert I.getValue(new_jn1) == 'ZoneA.P1.N0'

  assert (I.getNodeFromName1(new_jn0, 'PointList')[1] == [8,12,20]).all()
  assert (I.getNodeFromName1(new_jn0, 'PointListDonor')[1] == [11,21,25]).all()
  assert (I.getNodeFromName (new_jn0, 'Index')[1] == [5,3,2]).all()
  assert (I.getNodeFromName1(new_jn1, 'PointList')[1] == [9,1]).all()
  assert (I.getNodeFromName1(new_jn1, 'PointListDonor')[1] == [8,13]).all()
  assert (I.getNodeFromName (new_jn1, 'Index')[1] == [1,6]).all()
