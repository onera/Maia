import pytest
import re
import fnmatch
import numpy as np
from maia.sids import Internal_ext as IE
from maia.sids import internal as I
from maia.sids.cgns_keywords import Label as CGL

from   maia.utils        import parse_yaml_cgns


def test_getValue():
  # Camel case
  # ==========
  node = I.newDataArray('Test', [1,2])
  assert (I.getVal(node) == [1,2]).all()
  assert isinstance(I.getVal(node), np.ndarray)
  node = I.newDataArray('Test', np.array([1]))
  # Result from Cassiopee
  assert I.getValue(node) == 1
  assert isinstance(I.getValue(node), int)
  # Result expected
  assert I.getVal(node) == [1]
  assert isinstance(I.getVal(node), np.ndarray)

  # Snake case
  # ==========
  node = I.new_data_array('Test', [1,2])
  assert (I.get_val(node) == [1,2]).all()
  assert isinstance(I.get_val(node), np.ndarray)
  node = I.new_data_array('Test', np.array([1]))
  # Result from Cassiopee
  assert I.get_value(node) == 1
  assert isinstance(I.get_value(node), int)
  # Result expected
  assert I.get_val(node) == [1]
  assert isinstance(I.get_val(node), np.ndarray)


def test_getNodesDispatch1():
  fs = I.newFlowSolution()
  data_a   = I.newDataArray('DataA', [1,2,3], parent=fs)
  data_b   = I.newDataArray('DataB', [4,6,8], parent=fs)
  grid_loc = I.newGridLocation('Vertex', fs)
  assert I.getNodesDispatch1(fs, 'DataB') == [data_b]
  assert I.getNodesDispatch1(fs, 'DataArray_t') == [data_a, data_b]
  assert I.getNodesDispatch1(fs, CGL.GridLocation_t) == [grid_loc]
  assert I.getNodesDispatch1(fs, np.array([4,6,8])) == [data_b]
  assert I.getNodesDispatch1(fs, lambda n: isinstance(I.getValue(n), str) and I.getValue(n) == 'Vertex') == [grid_loc]
  with pytest.raises(TypeError):
    I.getNodesDispatch1(fs, False)

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
  zoneI = parse_yaml_cgns.to_node(yt)
  zbcB  = I.getNodeFromName(zoneI, 'ZBCB' )

  _isbc =  lambda n : I.getType(n) == 'BC_t'
  _iszbc = lambda n : I.getType(n) == 'ZoneBC_t'
  # Equivalence with NodesWalkers
  # -----------------------------
  walker = I.NodesWalkers(zoneI, [], search='dfs', depth=1)
  assert list(walker()) == []
  walker.predicates = [_isbc]
  assert list(walker()) == []
  walker.predicates = [lambda n : I.getName(n) == 'Index4_ii']
  assert list(walker()) == []

  walker.predicates = [_iszbc]
  assert [I.getName(node) for node in walker()] == ['ZBCA', 'ZBCB']
  walker.root       = zbcB
  walker.predicates = [lambda n : I.getName(n) == 'bc5']
  assert [I.getName(node) for node in walker()] == ['bc5']

  walker.root       = zoneI
  walker.predicates = [_iszbc, _isbc]
  assert [I.getName(node) for node in walker()] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  walker.predicates = [_iszbc, lambda n : I.getName(n) == 'bc5']
  assert [I.getName(node) for node in walker()] == ['bc5']

  results3 = ['Index1_i','Index2_i','Index3_i','Index4_i','Index4_ii','Index4_iii']

  walker.predicates = [_iszbc, _isbc, lambda n : I.getType(n) == 'IndexArray_t']
  assert [I.getName(node) for node in walker()] == results3
  walker.predicates = [_iszbc, _isbc, lambda n : fnmatch.fnmatch(n[0], 'Index*_i')]
  assert([I.getName(node) for node in walker()] == ['Index1_i','Index2_i','Index3_i','Index4_i'])

  walker.predicates = [_iszbc, _isbc, lambda n: I.getLabel(n) == 'IndexArray_t']
  # print(f"threelvl2 = {[I.getName(node) for node in walker()]}")
  assert [I.getName(node) for node in walker()] == results3

  with pytest.raises(TypeError):
    walker.predicates = 12

  # iterNodesByMatching
  # -------------------
  assert list(I.iterNodesByMatching(zoneI, '')) == []
  assert list(I.iterNodesByMatching(zoneI, 'BC_t')) == []
  assert list(I.iterNodesByMatching(zoneI, 'Index4_ii')) == []

  onelvl = I.iterNodesByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(node) for node in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = I.iterNodesByMatching(zbcB, 'bc5')
  assert [I.getName(node) for node in onelvl] == ['bc5']

  twolvl = I.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [I.getName(node) for node in twolvl] == ['bc1', 'bc2', 'bc3', 'bc4', 'bc5']
  twolvl = I.iterNodesByMatching(zoneI, 'ZoneBC_t/bc5')
  assert [I.getName(node) for node in twolvl] == ['bc5']
  twolvl = I.iterNodesByMatching(zbcB, 'BC_t/IndexArray_t')
  assert [I.getName(node) for node in twolvl] == ['Index3_i','Index4_i','Index4_ii','Index4_iii']

  results3 = ['Index1_i','Index2_i','Index3_i','Index4_i','Index4_ii','Index4_iii']
  threelvl = I.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  assert [I.getName(node) for node in threelvl] == results3
  assert len(list(I.iterNodesByMatching(zoneI, 'ZoneBC_t/BC_t/Index*_i'))) == 4

  # threelvl1 = I.iterNodesByMatching(zoneI, "ZoneBC_t/BC_t/lambda n: I.getType(n) == 'IndexArray_t'")
  # # print(f"threelvl1 = {[I.getName(node) for node in threelvl1]}")
  # assert [I.getName(node) for node in threelvl1] == results3

  threelvl2 = I.iterNodesByMatching(zoneI, ["ZoneBC_t", CGL.BC_t.name, lambda n: I.getLabel(n) == 'IndexArray_t'])
  # print(f"threelvl2 = {[I.getName(node) for node in threelvl2]}")
  assert [I.getName(node) for node in threelvl2] == results3

  threelvl3 = I.iterNodesByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: I.getLabel(n) == 'IndexArray_t'])
  # print(f"threelvl3 = {[I.getName(node) for node in threelvl3]}")
  assert [I.getName(node) for node in threelvl3] == results3

  with pytest.raises(TypeError):
    list(I.iterNodesByMatching(zoneI, 12))

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
  zoneI = parse_yaml_cgns.to_node(yt)
  zbcB  = I.getNodeFromName(zoneI, 'ZBCB' )

  _isbc =  lambda n : I.getType(n) == 'BC_t'
  _iszbc = lambda n : I.getType(n) == 'ZoneBC_t'
  # Equivalence with NodesWalkers
  # -----------------------------
  walker = I.NodesWalkers(zoneI, [], search='dfs', depth=1, ancestors=True)
  assert list(walker()) == []
  walker.predicates = [_isbc]
  assert list(walker()) == []
  walker.predicates = [lambda n : I.getName(n) == 'Index4_ii']
  assert list(walker()) == []

  walker.predicates = [_iszbc]
  assert [[I.getName(n) for n in nodes] for nodes in walker()] == [['ZBCA'], ['ZBCB']]
  walker.root       = zbcB
  walker.predicates = [_isbc]
  assert [[I.getName(n) for n in nodes] for nodes in walker()] == [['bc3'], ['bc4'], ['bc5']]

  fpath = lambda nodes: '/'.join([I.get_name(n) for n in nodes])
  walker.root       = zoneI
  walker.predicates = [_iszbc, _isbc]
  assert [fpath(nodes) for nodes in walker()] == ['ZBCA/bc1', 'ZBCA/bc2', 'ZBCB/bc3', 'ZBCB/bc4', 'ZBCB/bc5']
  walker.predicates = [_iszbc, lambda n : I.getName(n) == 'bc3']
  assert [fpath(nodes) for nodes in walker()] == ['ZBCB/bc3']

  walker.root       = zbcB
  walker.predicates = [_isbc, lambda n : I.getType(n) == 'IndexArray_t']
  iterator = walker()
  for bc in I.getNodesFromLabel1(zbcB, 'BC_t'):
    for idx in I.getNodesFromLabel1(bc, 'IndexArray_t'):
      assert next(iterator) == (bc, idx)

  walker.root       = zoneI
  walker.predicates = [_iszbc, _isbc, lambda n : I.getType(n) == 'IndexArray_t']
  iterator = walker()
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for idx in I.getNodesFromLabel1(bc, 'IndexArray_t'):
        assert next(iterator) == (zbc, bc, idx)

  walker.predicates = [_iszbc, _isbc, lambda n : 'PL' in I.getName(n)]
  iterator = walker()
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(iterator) == (zbc, bc, pl)

  p = re.compile('PL[12]')
  walker.predicates = [_iszbc, _isbc, lambda n: p.match(I.getName(n))]
  iterator = walker()
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in [n for n in I.getNodesFromName1(bc, 'PL*') if p.match(I.getName(n))]:
        assert next(iterator) == (zbc, bc, pl)

  with pytest.raises(TypeError):
    walker.predicates = 12

  # iterNodesWithParentsByMatching
  # ------------------------------
  assert list(I.iterNodesWithParentsByMatching(zoneI, '')) == []
  assert list(I.iterNodesWithParentsByMatching(zoneI, 'BC_t')) == []

  onelvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['ZBCA', 'ZBCB']
  onelvl = I.iterNodesWithParentsByMatching(zbcB, 'BC_t')
  assert [I.getName(nodes[0]) for nodes in onelvl] == ['bc3', 'bc4', 'bc5']

  twolvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCA', 'bc1'), ('ZBCA', 'bc2'), ('ZBCB', 'bc3'), ('ZBCB', 'bc4'), ('ZBCB', 'bc5')]
  twolvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/bc3')
  assert [(I.getName(nodes[0]), I.getName(nodes[1])) for nodes in twolvl] == \
      [('ZBCB', 'bc3')]
  twolvl = I.iterNodesWithParentsByMatching(zbcB, 'BC_t/IndexArray_t')
  for bc in I.getNodesFromLabel1(zbcB, 'BC_t'):
    for idx in I.getNodesFromLabel1(bc, 'IndexArray_t'):
      assert next(twolvl) == (bc, idx)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/IndexArray_t')
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for idx in I.getNodesFromLabel1(bc, 'IndexArray_t'):
        assert next(threelvl) == (zbc, bc, idx)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, 'ZoneBC_t/BC_t/PL*')
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, ['ZoneBC_t', 'BC_t', 'PL*'])
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, 'BC_t', 'PL*'])
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  threelvl = I.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, 'PL*'])
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in I.getNodesFromName1(bc, 'PL*'):
        assert next(threelvl) == (zbc, bc, pl)

  p = re.compile('PL[12]')
  threelvl = I.iterNodesWithParentsByMatching(zoneI, [CGL.ZoneBC_t, CGL.BC_t.name, lambda n: p.match(I.getName(n))])
  for zbc in I.getNodesFromLabel1(zoneI, 'ZoneBC_t'):
    for bc in I.getNodesFromLabel1(zbc, 'BC_t'):
      for pl in [n for n in I.getNodesFromName1(bc, 'PL*') if p.match(I.getName(n))]:
        assert next(threelvl) == (zbc, bc, pl)

  with pytest.raises(TypeError):
    list(I.iterNodesWithParentsByMatching(zoneI, 12))

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
  zone = parse_yaml_cgns.to_node(yt)

  assert I.getSubregionExtent(I.getNodeFromName(zone, 'UnLinkedZSR'), zone) == 'UnLinkedZSR'
  assert I.getSubregionExtent(I.getNodeFromName(zone, 'BCLinkedZSR'), zone) == 'ZBC/BC2'
  assert I.getSubregionExtent(I.getNodeFromName(zone, 'GCLinkedZSR'), zone) == 'ZGC2/GC1to1B'

  with pytest.raises(ValueError):
    I.getSubregionExtent(I.getNodeFromName(zone, 'OrphelanZSR'), zone)
  with pytest.raises(I.CGNSLabelNotEqualError):
    I.getSubregionExtent(I.getNodeFromName(zone, 'WrongZSR'), zone)

def test_newDistribution():
  distri = I.newDistribution()
  assert I.getName(distri) == ':CGNS#Distribution'
  assert I.getLabel(distri) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  distri = I.newDistribution(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#Distribution') is not None

  zone = I.newZone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = I.newDistribution(distri_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = I.newZone('zone')
  distri_arrayA = {'Cell' : [0,15,30]}
  distri_arrayB = {'Vertex' : [100,1000,1000]}
  distri = I.newDistribution(distri_arrayA, parent=zone)
  distri = I.newDistribution(distri_arrayB, parent=zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(I.getNodesFromName(zone, ':CGNS#Distribution')) == 1

def test_newGlobalNumbering():
  gnum = I.newGlobalNumbering()
  assert I.getName(gnum) == ':CGNS#GlobalNumbering'
  assert I.getLabel(gnum) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  gnum = I.newGlobalNumbering(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#GlobalNumbering') is not None

  zone = I.newZone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = I.newGlobalNumbering(gnum_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None

def test_getDistribution():
  zone = I.newZone()
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = I.newDistribution(distri_arrays, zone)
  assert I.getDistribution(zone) is distri
  assert (I.getVal(I.getDistribution(zone, 'Cell')) == [0,15,30]).all()
  assert (I.getVal(I.getDistribution(zone, 'Vertex')) == [100,1000,1000]).all()

def test_getGlobalNumbering():
  zone = I.newZone()
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = I.newGlobalNumbering(gnum_arrays, zone)
  assert I.getGlobalNumbering(zone) is gnum_node
  assert (I.getVal(I.getGlobalNumbering(zone, 'Cell')) == [4,21,1,2,8,12]).all()
  assert  I.getVal(I.getGlobalNumbering(zone, 'Vertex')) == None

