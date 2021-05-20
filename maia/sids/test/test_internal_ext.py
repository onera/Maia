import pytest
import re
import fnmatch
import numpy as np
from Converter import Internal     as I
from maia.sids import Internal_ext as IE
from maia.sids.cgns_keywords import Label as CGL

from   maia.utils        import parse_yaml_cgns

def test_getValue():
  node = I.newDataArray('Test', [1,2])
  assert (IE.getValue(node) == [1,2]).all()
  assert isinstance(IE.getValue(node), np.ndarray)
  node = I.newDataArray('Test', np.array([1]))
  # Result from Cassiopee
  assert I.getValue(node) == 1
  assert isinstance(I.getValue(node), int)
  # Result expected
  assert IE.getValue(node) == [1]
  assert isinstance(IE.getValue(node), np.ndarray)

def test_getNodesFromQuery():
  yt = """
Base CGNSBase_t:
  ZoneI Zone_t:
    Ngon Elements_t [22,0]:
    NFace Elements_t [23,0]:
    ZBCA ZoneBC_t:
      bc1 BC_t:
        Index_i IndexArray_t:
      bc2 BC_t:
        Index_ii IndexArray_t:
    ZBCB ZoneBC_t:
      bc3 BC_t:
        Index_iii IndexArray_t:
      bc4 BC_t:
      bc5 BC_t:
        Index_iv IndexArray_t:
        Index_v IndexArray_t:
        Index_vi IndexArray_t:
"""
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  # I.printTree(tree)

  # Test from Name
  bases = I.getBases(tree)
  nodes_from_name1 = I.getNodesFromName1(bases, "ZoneI")
  assert IE.getNodesFromQuery1(bases, lambda n: I.getName(n) == "ZoneI") == nodes_from_name1
  # Wildcard is not allowed in getNodesFromName1() from Cassiopee
  assert I.getNodesFromName1(bases, "Zone*") == []
  assert IE.getNodesFromQuery1(bases, lambda n: fnmatch.fnmatch(I.getName(n), "Zone*")) == nodes_from_name1
  assert IE.getNodesFromName1(bases, "Zone*") == nodes_from_name1

  # Test from Type
  nodes_from_type1 = I.getNodesFromType1(bases, CGL.Zone_t.name)
  assert IE.getNodesFromQuery1(bases, lambda n: I.getType(n) == CGL.Zone_t.name) == nodes_from_type1
  assert IE.getNodesFromType1(bases, CGL.Zone_t.name) == nodes_from_type1

  # Test from Value
  zones = I.getZones(tree)
  ngon_value = np.array([22,0], dtype=np.int32)
  elements_from_type_value1 = [n for n in I.getNodesFromType1(zones, CGL.Elements_t.name) if np.array_equal(IE.getValue(n), ngon_value)]
  assert IE.getNodesFromQuery1(zones, lambda n: I.getType(n) == CGL.Elements_t.name and np.array_equal(IE.getValue(n), ngon_value)) == elements_from_type_value1
  assert IE.getNodesFromValue1(zones, ngon_value) == elements_from_type_value1

  zonebcs_from_type_name1 = [n for n in I.getNodesFromType1(zones, CGL.ZoneBC_t.name) if I.getName(n) != "ZBCA"]
  assert IE.getNodesFromQuery1(zones, lambda n: I.getType(n) == CGL.ZoneBC_t.name and I.getName(n) != "ZBCA") == zonebcs_from_type_name1


def test_getNodesByMatching():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index1_i IndexArray_t:
    bc2 BC_t:
      Index2_i IndexArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index3_i IndexArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index4_i IndexArray_t:
      Index4_ii IndexArray_t:
      Index4_iii IndexArray_t:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)
  zoneI = I.getNodeFromName(root, 'ZoneI')
  zbcB  = I.getNodeFromName(root, 'ZBCB' )

  assert list(IE.getNodesByMatching(zoneI, '')) == []
  assert list(IE.getNodesByMatching(zoneI, 'BC_t')) == []
  assert list(IE.getNodesByMatching(zoneI, 'Index4_ii')) == []

  onelvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(node) for node in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = IE.getNodesByMatching(zbcB, 'bc5')
  assert [I.getName(node) for node in onelvl] == ['bc5']

  twolvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [I.getName(node) for node in twolvl] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  twolvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t/bc5')
  assert [I.getName(node) for node in twolvl] == ['bc5']
  twolvl = IE.getNodesByMatching(zbcB, 'BC_t/IndexArray_t')
  assert [I.getName(node) for node in twolvl] == ['Index3_i','Index4_i','Index4_ii','Index4_iii']

  results3 = ['Index1_i','Index2_i','Index3_i','Index4_i','Index4_ii','Index4_iii']
  threelvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  assert [I.getName(node) for node in threelvl] == results3
  assert len(list(IE.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t/Index*_i'))) == 4

  threelvl1 = IE.getNodesByMatching(zoneI, "ZoneBC_t/BC_t/lambda n: I.getType(n) == 'IndexArray_t'")
  # print(f"threelvl1 = {[I.getName(node) for node in threelvl1]}")
  assert [I.getName(node) for node in threelvl1] == results3

  threelvl2 = IE.getNodesByMatching(zoneI, ["ZoneBC_t", CGL.BC_t.name, lambda n: I.getType(n) == 'IndexArray_t'])
  # print(f"threelvl2 = {[I.getName(node) for node in threelvl2]}")
  assert [I.getName(node) for node in threelvl2] == results3

  threelvl3 = IE.getNodesByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: I.getType(n) == 'IndexArray_t'])
  # print(f"threelvl3 = {[I.getName(node) for node in threelvl3]}")
  assert [I.getName(node) for node in threelvl3] == results3


def test_getNodesByMatchingApplies():
  yt = """
Base CGNSBase_t:
  ZoneI.P0.N0 Zone_t:
    Ngon Elements_t [22,0]:
  ZoneI.P1.N0 Zone_t:
    Ngon Elements_t [22,0]:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)

  zones = IE.getNodesByMatching(root, ["CGNSBase_t", "Zone_t"])
  # print(f"zones = {[I.getName(node) for node in zones]}")
  assert [I.getName(node) for node in zones] == ["ZoneI.P0.N0", "ZoneI.P1.N0"]

  def reduce_name(n):
    name = I.getName(n)
    new_name = p.search(name).group(1)
    I.setName(n, new_name)
    return n

  p = re.compile('([a-zA-Z0-9_]*)(\\.P[0-9]*)(\\.N[0-9]*)')
  zones = IE.getNodesByMatching(root, ["CGNSBase_t", "Zone_t"], {1:reduce_name})
  # print(f"zones = {[I.getName(node) for node in zones]}")
  assert [I.getName(node) for node in zones] == ["ZoneI", "ZoneI"]


def test_getNodesWithParentsByMatching():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index_i IndexArray_t:
      PL1 DataArray_t:
    bc2 BC_t:
      Index_ii IndexArray_t:
      PL2 DataArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index_iii IndexArray_t:
      PL3 DataArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index_iv IndexArray_t:
      Index_v IndexArray_t:
      Index_vi IndexArray_t:
      PL4 DataArray_t:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)
  zoneI = I.getNodeFromName(root, 'ZoneI')
  zbcB  = I.getNodeFromName(root, 'ZBCB' )

  assert list(IE.getNodesWithParentsByMatching(zoneI, '')) == []
  assert list(IE.getNodesWithParentsByMatching(zoneI, 'BC_t')) == []

  onelvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = IE.getNodesWithParentsByMatching(zbcB, 'BC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCA', 'bc1'), ('ZBCA', 'bc2'), ('ZBCB', 'bc3'), ('ZBCB', 'bc4'), ('ZBCB', 'bc5')]
  twolvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/bc3')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCB', 'bc3')]
  twolvl = IE.getNodesWithParentsByMatching(zbcB, 'BC_t/IndexArray_t')
  for bc in I.getNodesFromType1(zbcB, 'BC_t'):
    for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
      assert next(twolvl) == (bc, idx)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
        assert next(threelvl) == (zbc, bc, idx)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/PL*')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, ['ZoneBC_t', 'BC_t', 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, 'BC_t', 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, 'PL*'])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in IE.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  p = re.compile('PL[12]')
  threelvl = IE.getNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: p.match(I.getName(n))])
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for pl in [n for n in IE.getNodesFromName1(bc, 'PL*') if p.match(I.getName(n))]:
        assert next(threelvl) == (zbc, bc, pl)

def test_getNodesByParentsMatchingApplies():
  yt = """
Base CGNSBase_t:
  ZoneI.P0.N0 Zone_t:
    Ngon Elements_t [22,0]:
  ZoneI.P1.N0 Zone_t:
    Ngon Elements_t [22,0]:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)

  zones = IE.getNodesWithParentsByMatching(root, ["CGNSBase_t", "Zone_t"])
  # print(f"zones = {[(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in zones]}")
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in zones] == [("Base", "ZoneI.P0.N0"), ("Base", "ZoneI.P1.N0")]

  def reduce_name(n):
    name = I.getName(n)
    new_name = p.search(name).group(1)
    I.setName(n, new_name)
    return n

  p = re.compile('([a-zA-Z0-9_]*)(\\.P[0-9]*)(\\.N[0-9]*)')
  zones = IE.getNodesWithParentsByMatching(root, ["CGNSBase_t", "Zone_t"], {1:reduce_name})
  # print(f"zones = {[(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in zones]}")
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in zones] == [("Base", "ZoneI"), ("Base", "ZoneI")]


def test_getNodesFromTypePath():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index_i IndexArray_t:
    bc2 BC_t:
      Index_ii IndexArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index_iii IndexArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index_iv IndexArray_t:
      Index_v IndexArray_t:
      Index_vi IndexArray_t:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)
  zoneI = I.getNodeFromName(root, 'ZoneI')
  zbcB  = I.getNodeFromName(root, 'ZBCB')

  assert list(IE.getNodesByMatching(zoneI, '')) == []
  assert list(IE.getNodesByMatching(zoneI, 'BC_t')) == []

  onelvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(node) for node in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = IE.getNodesByMatching(zbcB, 'BC_t')
  assert [I.getName(node) for node in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [I.getName(node) for node in twolvl] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  twolvl = IE.getNodesByMatching(zbcB, 'BC_t/IndexArray_t')
  assert [I.getName(node) for node in twolvl] == ['Index_iii','Index_iv','Index_v','Index_vi']

  threelvl = IE.getNodesByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  assert [I.getName(node) for node in threelvl] == ['Index_i', 'Index_ii','Index_iii','Index_iv','Index_v','Index_vi']

def test_getNodesWithParentsFromTypePath():
  yt = """
ZoneI Zone_t:
  Ngon Elements_t [22,0]:
  ZBCA ZoneBC_t:
    bc1 BC_t:
      Index_i IndexArray_t:
    bc2 BC_t:
      Index_ii IndexArray_t:
  ZBCB ZoneBC_t:
    bc3 BC_t:
      Index_iii IndexArray_t:
    bc4 BC_t:
    bc5 BC_t:
      Index_iv IndexArray_t:
      Index_v IndexArray_t:
      Index_vi IndexArray_t:
"""
  root = parse_yaml_cgns.to_complete_pytree(yt)
  zoneI = I.getNodeFromName(root, 'ZoneI')
  zbcB  = I.getNodeFromName(root, 'ZBCB' )

  assert list(IE.getNodesWithParentsByMatching(zoneI, '')) == []
  assert list(IE.getNodesWithParentsByMatching(zoneI, 'BC_t')) == []

  onelvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = IE.getNodesWithParentsByMatching(zbcB, 'BC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCA', 'bc1'), ('ZBCA', 'bc2'), ('ZBCB', 'bc3'), ('ZBCB', 'bc4'), ('ZBCB', 'bc5')]
  twolvl = IE.getNodesWithParentsByMatching(zbcB, 'BC_t/IndexArray_t')
  for bc in I.getNodesFromType1(zbcB, 'BC_t'):
    for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
      assert next(twolvl) == (bc, idx)

  threelvl = IE.getNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  for zbc in I.getNodesFromType1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zbc, 'BC_t'):
      for idx in I.getNodesFromType1(bc, 'IndexArray_t'):
        assert next(threelvl) == (zbc, bc, idx)


def test_getSubregionExtent():
  yt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
    BC2 BC_t:
  ZGC1 ZoneGridConnectivity_t:
    GCA GridConnectivity_t:
    GCB GridConnectivity_t:
  ZGC2 ZoneGridConnectivity_t:
    GC1to1A GridConnectivity1to1_t:
    GC1to1B GridConnectivity1to1_t:
  UnLinkedZSR ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[]]:
  BCLinkedZSR ZoneSubRegion_t:
    BCRegionName Descriptor_t "BC2":
  GCLinkedZSR ZoneSubRegion_t:
    GridConnectivityRegionName Descriptor_t "GC1to1B":
  OrphelanZSR ZoneSubRegion_t:
    BCRegionName Descriptor_t "BC9":
  WrongZSR WrongType_t:
    BCRegionName Descriptor_t "BC":
  """
  tree = parse_yaml_cgns.to_complete_pytree(yt)
  zone = I.getZones(tree)[0]

  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'UnLinkedZSR'), zone) == 'UnLinkedZSR'
  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'BCLinkedZSR'), zone) == 'ZBC/BC2'
  assert IE.getSubregionExtent(I.getNodeFromName(zone, 'GCLinkedZSR'), zone) == 'ZGC2/GC1to1B'

  with pytest.raises(ValueError):
    IE.getSubregionExtent(I.getNodeFromName(zone, 'OrphelanZSR'), zone)
  with pytest.raises(AssertionError):
    IE.getSubregionExtent(I.getNodeFromName(zone, 'WrongZSR'), zone)


def test_newDistribution():
  distri = IE.newDistribution()
  assert I.getName(distri) == ':CGNS#Distribution'
  assert I.getType(distri) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  distri = IE.newDistribution(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#Distribution') is not None

  zone = I.newZone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = I.newZone('zone')
  distri_arrayA = {'Cell' : [0,15,30]}
  distri_arrayB = {'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrayA, parent=zone)
  distri = IE.newDistribution(distri_arrayB, parent=zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(I.getNodesFromName(zone, ':CGNS#Distribution')) == 1

def test_newGlobalNumbering():
  gnum = IE.newGlobalNumbering()
  assert I.getName(gnum) == ':CGNS#GlobalNumbering'
  assert I.getType(gnum) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  gnum = IE.newGlobalNumbering(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#GlobalNumbering') is not None

  zone = I.newZone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = IE.newGlobalNumbering(gnum_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None

def test_getDistribution():
  zone = I.newZone()
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = IE.newDistribution(distri_arrays, zone)
  assert IE.getDistribution(zone) is distri
  assert (IE.getDistribution(zone, 'Cell') == [0,15,30]).all()
  assert (IE.getDistribution(zone, 'Vertex') == [100,1000,1000]).all()

def test_getGlobalNumbering():
  zone = I.newZone()
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = IE.newGlobalNumbering(gnum_arrays, zone)
  assert IE.getGlobalNumbering(zone) is gnum_node
  assert (IE.getGlobalNumbering(zone, 'Cell') == [4,21,1,2,8,12]).all()
  assert  IE.getGlobalNumbering(zone, 'Vertex') == None
