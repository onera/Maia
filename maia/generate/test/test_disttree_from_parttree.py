from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I

from maia.utils         import parse_yaml_cgns
from maia.generate import disttree_from_parttree as DFP

@mark_mpi_test(3)
class Test_discover_nodes_of_kind:
  pt = [\
  """
Zone.P0.N0 Zone_t:
  ZBC ZoneBC_t:
    BCA BC_t "wall":
      Family FamilyName_t "myfamily":
  """,
  """
Zone.P1.N0 Zone_t:
  ZGC ZoneGridConnectivity_t:
    match.0 GridConnectivity_t:
    match.1 GridConnectivity_t:
  """,
  """
Zone.P2.N0 Zone_t:
  ZBC ZoneBC_t:
    BCB BC_t "farfield":
      GridLocation GridLocation_t "FaceCenter":
Zone.P2.N1 Zone_t:
  ZBC ZoneBC_t:
    BCA BC_t "wall":
      Family FamilyName_t "myfamily":
  """]
  def test_simple(self, sub_comm):
    dist_zone = I.newZone('Zone')
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])

    DFP.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm)
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == "wall")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == "farfield")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == "BC_t")

  def test_short(self, sub_comm):
    dist_node = I.createNode('SomeName', 'UserDefinedData_t')
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])
    part_nodes = [I.getNodeFromPath(zone, 'ZBC') for zone in I.getZones(part_tree)\
        if I.getNodeFromPath(zone, 'ZBC') is not None]
    DFP.discover_nodes_of_kind(dist_node, part_nodes, 'BC_t', sub_comm)
    assert I.getNodeFromPath(dist_node, 'BCA') is not None
    assert I.getNodeFromPath(dist_node, 'BCB') is not None

  def test_with_childs(self, sub_comm):
    dist_zone = I.newZone('Zone')
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])

    DFP.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm,
        child_list=['FamilyName_t', 'GridLocation'])
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA/Family')) == "myfamily")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA/Family')) == "FamilyName_t")
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCB/GridLocation')) == "FaceCenter")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB/GridLocation')) == "GridLocation_t")

  def test_with_rule(self, sub_comm):
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])

    #Exclude from node name
    dist_zone = I.newZone('Zone')
    DFP.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm,
        child_list=['FamilyName_t', 'GridLocation'],
        skip_rule=lambda node: 'A' in I.getName(node))
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCA') is None
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCB') is not None

    #Exclude from node content
    dist_zone = I.newZone('Zone')
    DFP.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm,
        skip_rule=lambda node: I.getNodeFromType1(node, 'FamilyName_t') is None)
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCA') is not None
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCB') is None

  def test_multiple(self, sub_comm):
    gc_path = 'ZoneGridConnectivity_t/GridConnectivity_t'
    part_tree = parse_yaml_cgns.to_complete_pytree(self.pt[sub_comm.Get_rank()])

    #Exclude from node name
    dist_zone = I.newZone('Zone')
    DFP.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), gc_path, sub_comm, allow_multiple=False)
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.0') is not None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.1') is not None

    dist_zone = I.newZone('Zone')
    DFP.discover_nodes_of_kind(dist_zone, I.getZones(part_tree), gc_path, sub_comm, allow_multiple=True)
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.0') is None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.1') is None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match') is not None

def test_match_jn_from_ordinals():
  dt = """
Base CGNSBase_t:
  ZoneA Zone_t:
    ZGC ZoneGridConnectivity_t:
      perio1 GridConnectivity_t:
        Ordinal UserDefinedData_t [1]:
        OrdinalOpp UserDefinedData_t [2]:
        PointList IndexArray_t [[1,3]]:
      perio2 GridConnectivity_t:
        Ordinal UserDefinedData_t [2]:
        OrdinalOpp UserDefinedData_t [1]:
        PointList IndexArray_t [[2,4]]:
      match1 GridConnectivity_t:
        Ordinal UserDefinedData_t [3]:
        OrdinalOpp UserDefinedData_t [4]:
        PointList IndexArray_t [[10,100]]:
  ZoneB Zone_t:
    ZGC ZoneGridConnectivity_t:
      match2 GridConnectivity_t:
        Ordinal UserDefinedData_t [4]:
        OrdinalOpp UserDefinedData_t [3]:
        PointList IndexArray_t [[-100,-10]]:
  """
  dist_tree = parse_yaml_cgns.to_complete_pytree(dt)
  DFP.match_jn_from_ordinals(dist_tree)
  expected_names  = ['ZoneA', 'ZoneA', 'ZoneB', 'ZoneA']
  expected_pl_opp = [[2,4], [1,3], [-100,-10], [10,100]]
  for i, jn in enumerate(I.getNodesFromType(dist_tree, 'GridConnectivity_t')):
    assert I.getValue(jn) == expected_names[i]
    assert (I.getNodeFromName1(jn, 'PointListDonor')[1] == expected_pl_opp[i]).all()

