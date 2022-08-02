import pytest
import Converter.Internal as I
import maia.pytree        as PT

from maia.pytree.maia import maia_nodes as mNode

def test_newDistribution():
  distri = mNode.newDistribution()
  assert I.getName(distri) == ':CGNS#Distribution'
  assert I.getType(distri) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  distri = mNode.newDistribution(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#Distribution') is not None

  zone = I.newZone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = mNode.newDistribution(distri_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = I.newZone('zone')
  distri_arrayA = {'Cell' : [0,15,30]}
  distri_arrayB = {'Vertex' : [100,1000,1000]}
  distri = mNode.newDistribution(distri_arrayA, parent=zone)
  distri = mNode.newDistribution(distri_arrayB, parent=zone)
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(PT.get_nodes_from_name(zone, ':CGNS#Distribution')) == 1

def test_newGlobalNumbering():
  gnum = mNode.newGlobalNumbering()
  assert I.getName(gnum) == ':CGNS#GlobalNumbering'
  assert I.getType(gnum) == 'UserDefinedData_t'

  zone = I.newZone('zone')
  gnum = mNode.newGlobalNumbering(parent=zone)
  assert I.getNodeFromName(zone, ':CGNS#GlobalNumbering') is not None

  zone = I.newZone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = mNode.newGlobalNumbering(gnum_arrays, zone)
  assert (I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert I.getNodeFromPath(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None

def test_getDistribution():
  zone = I.newZone()
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = mNode.newDistribution(distri_arrays, zone)
  assert mNode.getDistribution(zone) is distri
  assert (I.getVal(mNode.getDistribution(zone, 'Cell')) == [0,15,30]).all()
  assert (I.getVal(mNode.getDistribution(zone, 'Vertex')) == [100,1000,1000]).all()

def test_getGlobalNumbering():
  zone = I.newZone()
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = mNode.newGlobalNumbering(gnum_arrays, zone)
  assert mNode.getGlobalNumbering(zone) is gnum_node
  assert (I.getVal(mNode.getGlobalNumbering(zone, 'Cell')) == [4,21,1,2,8,12]).all()
  assert  I.getVal(mNode.getGlobalNumbering(zone, 'Vertex')) == None

