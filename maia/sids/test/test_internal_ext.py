import numpy as np
from Converter import Internal     as I
from maia.sids import Internal_ext as IE

def test_getVal():
  node = I.newDataArray('Test', [1,2])
  assert (IE.getVal(node) == [1,2]).all()
  assert isinstance(IE.getVal(node), np.ndarray)
  node = I.newDataArray('Test', np.array([1]))
  assert IE.getVal(node) == [1]
  assert isinstance(IE.getVal(node), np.ndarray)

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
