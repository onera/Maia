import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I

from maia.sids          import conventions as conv
from maia.utils         import parse_yaml_cgns
from maia.sids.cgns_keywords import Label as CGL
from maia.tree_exchange.part_to_dist import discover as disc

@mark_mpi_test(3)
class Test_discover_nodes_from_matching:
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
    part_zones = parse_yaml_cgns.to_nodes(self.pt[sub_comm.Get_rank()])
    # I.printTree(part_tree)

    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, part_zones, 'ZoneBC_t/BC_t', sub_comm)
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == "BC_t")

    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, part_zones, ['ZoneBC_t', 'BC_t'], sub_comm)
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == "BC_t")

    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, part_zones, [CGL.ZoneBC_t, CGL.BC_t], sub_comm)
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == "BC_t")

    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, part_zones, [CGL.ZoneBC_t, 'BC_t'], sub_comm)
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == "BC_t")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == "BC_t")

    dist_zone = I.newZone('Zone')
    queries = [CGL.ZoneBC_t, lambda n : I.getType(n) == "BC_t" and I.getName(n) != "BCA"]
    disc.discover_nodes_from_matching(dist_zone, part_zones, queries, sub_comm)
    assert (I.getNodeFromPath(dist_zone, 'ZBC/BCA') == None)
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == "BC_t")
    # I.printTree(dist_zone)

  def test_short(self, sub_comm):
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[sub_comm.Get_rank()])
    part_nodes = [I.getNodeFromPath(zone, 'ZBC') for zone in I.getZones(part_tree)\
      if I.getNodeFromPath(zone, 'ZBC') is not None]

    dist_node = I.createNode('SomeName', 'UserDefinedData_t')
    disc.discover_nodes_from_matching(dist_node, part_nodes, 'BC_t', sub_comm)
    assert I.getNodeFromPath(dist_node, 'BCA') is not None
    assert I.getNodeFromPath(dist_node, 'BCB') is not None

    dist_node = I.createNode('SomeName', 'UserDefinedData_t')
    queries = [lambda n : I.getType(n) == "BC_t" and I.getName(n) != "BCA"]
    disc.discover_nodes_from_matching(dist_node, part_nodes, queries, sub_comm)
    assert I.getNodeFromPath(dist_node, 'BCA') is None
    assert (I.getType(I.getNodeFromPath(dist_node, 'BCB')) == "BC_t")
    # I.printTree(dist_node)

  def test_getvalue(self, sub_comm):
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[sub_comm.Get_rank()])
    for zbc in I.getNodesFromName(part_tree, 'ZBC'):
      I.setValue(zbc, 'test')

    # get_value as a string
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm)
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == 'test'
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == None
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value='none')
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == None
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == None
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value='all')
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == 'test'
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == 'wall'
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value='ancestors')
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == 'test'
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == None
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value='leaf')
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == None
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == 'wall'

    # dist_zone = I.newZone('Zone')
    # with pytest.raises(ValueError):
    #   disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value="toto")

    # get_value as a list
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value=[False, False])
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == None
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == None
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value=[True, True])
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == 'test'
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == 'wall'
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value=[True, False])
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == 'test'
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == None
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value=[False, True])
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == None
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA')) == 'wall'

    # dist_zone = I.newZone('Zone')
    # with pytest.raises(TypeError):
    #   disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm, get_value=2)

    # get_value and search with predicate as lambda
    dist_zone = I.newZone('Zone')
    queries = [CGL.ZoneBC_t, lambda n : I.getType(n) == "BC_t" and I.getName(n) != "BCA"]
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), queries, sub_comm, get_value='all')
    assert I.getNodeFromPath(dist_zone, 'BCA') is None
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC')) == 'test'
    assert I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCB')) == 'farfield'
    # I.printTree(dist_zone)


  def test_with_childs(self, sub_comm):
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[sub_comm.Get_rank()])

    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), 'ZoneBC_t/BC_t', sub_comm,
                                child_list=['FamilyName_t', 'GridLocation'])
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCA/Family')) == "myfamily")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCA/Family')) == "FamilyName_t")
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCB/GridLocation')) == "FaceCenter")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB/GridLocation')) == "GridLocation_t")

    dist_zone = I.newZone('Zone')
    queries = [CGL.ZoneBC_t, lambda n : I.getType(n) == "BC_t" and I.getName(n) != "BCA"]
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), queries, sub_comm,
                                      child_list=['FamilyName_t', 'GridLocation'])
    assert (I.getValue(I.getNodeFromPath(dist_zone, 'ZBC/BCB/GridLocation')) == "FaceCenter")
    assert (I.getType(I.getNodeFromPath(dist_zone, 'ZBC/BCB/GridLocation')) == "GridLocation_t")

  def test_with_rule(self, sub_comm):
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[sub_comm.Get_rank()])

    # Exclude from node name
    dist_zone = I.newZone('Zone')
    queries = [CGL.ZoneBC_t, lambda n : I.getType(n) == "BC_t" and not 'A' in I.getName(n)]
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), queries, sub_comm,
                                      child_list=['FamilyName_t', 'GridLocation'])
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCA') is None
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCB') is not None

    # Exclude from node content
    dist_zone = I.newZone('Zone')
    queries = [CGL.ZoneBC_t, lambda n : I.getType(n) == "BC_t" and I.getNodeFromType1(n, 'FamilyName_t') is not None]
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), queries, sub_comm,
                                      child_list=['FamilyName_t', 'GridLocation'])
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCA') is not None
    assert I.getNodeFromPath(dist_zone, 'ZBC/BCB') is None

  def test_multiple(self, sub_comm):
    gc_path = 'ZoneGridConnectivity_t/GridConnectivity_t'
    part_tree = parse_yaml_cgns.to_cgns_tree(self.pt[sub_comm.Get_rank()])

    # Test discover_nodes_from_matching(...)
    # --------------------------------------
    dist_zone = I.newZone('Zone')
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), gc_path, sub_comm)
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.0') is not None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.1') is not None

    dist_zone = I.newZone('Zone')
    queries = [CGL.ZoneGridConnectivity_t, CGL.GridConnectivity_t]
    disc.discover_nodes_from_matching(dist_zone, I.getZones(part_tree), queries, sub_comm,\
        merge_rule=lambda path : conv.get_split_prefix(path))
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.0') is None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match.1') is None
    assert I.getNodeFromPath(dist_zone, 'ZGC/match') is not None

  def test_zones(self, sub_comm):
    part_tree = I.newCGNSTree()
    if sub_comm.Get_rank() == 0:
      part_base = I.newCGNSBase('BaseA', parent=part_tree)
      I.newZone('Zone.P0.N0', parent=part_base)
    elif sub_comm.Get_rank() == 1:
      part_base = I.newCGNSBase('BaseB', parent=part_tree)
      I.newZone('Zone.withdot.P1.N0', parent=part_base)
    elif sub_comm.Get_rank() == 2:
      part_base = I.newCGNSBase('BaseA', parent=part_tree)
      I.newZone('Zone.P2.N0', parent=part_base)
      I.newZone('Zone.P2.N1', parent=part_base)

    # Test discover_nodes_from_matching(...)
    # --------------------------------------
    dist_tree = I.newCGNSTree()
    disc.discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t/Zone_t', sub_comm,\
        merge_rule=lambda zpath : conv.get_part_prefix(zpath))

    assert len(I.getZones(dist_tree)) == 2
    assert I.getNodeFromPath(dist_tree, 'BaseA/Zone') is not None
    assert I.getNodeFromPath(dist_tree, 'BaseB/Zone.withdot') is not None

if __name__ == "__main__":
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  test = Test_discover_nodes_of_kind()
  # test.test_simple(comm)
  # test.test_short(comm)
