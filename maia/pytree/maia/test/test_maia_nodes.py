import pytest
import maia.pytree        as PT

from maia.pytree.maia import maia_nodes as mNode

def test_newDistribution():
  distri = mNode.newDistribution()
  assert PT.get_name(distri)  == ':CGNS#Distribution'
  assert PT.get_label(distri) == 'UserDefinedData_t'

  zone = PT.new_Zone('zone')
  distri = mNode.newDistribution(parent=zone)
  assert PT.get_child_from_name(zone, ':CGNS#Distribution') is not None

  zone = PT.new_Zone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = mNode.newDistribution(distri_arrays, zone)
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = PT.new_Zone('zone')
  distri = mNode.newDistribution({'Cell' : [0,10,20,30]}, parent=zone)
  distri = mNode.newDistribution({'Cell' : [0,15,30]}, parent=zone) #Try update
  distri = mNode.newDistribution({'Vertex' : [100,1000,1000]}, parent=zone)
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(PT.get_nodes_from_name(zone, ':CGNS#Distribution')) == 1

def test_newGlobalNumbering():
  gnum = mNode.newGlobalNumbering()
  assert PT.get_name(gnum)  == ':CGNS#GlobalNumbering'
  assert PT.get_label(gnum) == 'UserDefinedData_t'

  zone = PT.new_Zone('zone')
  gnum = mNode.newGlobalNumbering(parent=zone)
  assert PT.get_child_from_name(zone, ':CGNS#GlobalNumbering') is not None

  zone = PT.new_Zone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = mNode.newGlobalNumbering(gnum_arrays, zone)
  assert (PT.get_node_from_path(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert PT.get_node_from_path(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None

def test_getDistribution():
  zone = PT.new_Zone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = mNode.newDistribution(distri_arrays, zone)
  assert mNode.getDistribution(zone) is distri
  assert (PT.get_value(mNode.getDistribution(zone, 'Cell')) == [0,15,30]).all()
  assert (PT.get_value(mNode.getDistribution(zone, 'Vertex')) == [100,1000,1000]).all()

def test_getGlobalNumbering():
  zone = PT.new_Zone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = mNode.newGlobalNumbering(gnum_arrays, zone)
  assert mNode.getGlobalNumbering(zone) is gnum_node
  assert (PT.get_value(mNode.getGlobalNumbering(zone, 'Cell')) == [4,21,1,2,8,12]).all()
  assert  PT.get_value(mNode.getGlobalNumbering(zone, 'Vertex')) == None

