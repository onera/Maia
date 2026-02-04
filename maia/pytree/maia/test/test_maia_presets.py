import pytest
import maia.pytree        as PT
import maia.pytree.maia   as MT


def test_new_Distribution():
  distri = MT.new_Distribution()
  assert PT.get_name(distri)  == ':CGNS#Distribution'
  assert PT.get_label(distri) == 'UserDefinedData_t'

  zone = PT.new_Zone('zone')
  distri = MT.new_Distribution(parent=zone)
  assert PT.get_child_from_name(zone, ':CGNS#Distribution') is not None

  zone = PT.new_Zone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = MT.new_Distribution(distri_arrays, zone)
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = PT.new_Zone('zone')
  distri = MT.new_Distribution({'Cell' : [0,10,20,30]}, parent=zone)
  distri = MT.new_Distribution({'Cell' : [0,15,30]}, parent=zone) #Try update
  distri = MT.new_Distribution({'Vertex' : [100,1000,1000]}, parent=zone)
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(PT.get_nodes_from_name(zone, ':CGNS#Distribution')) == 1

def test_new_GlobalNumbering():
  gnum = MT.new_GlobalNumbering()
  assert PT.get_name(gnum)  == ':CGNS#GlobalNumbering'
  assert PT.get_label(gnum) == 'UserDefinedData_t'

  zone = PT.new_Zone('zone')
  gnum = MT.new_GlobalNumbering(parent=zone)
  assert PT.get_child_from_name(zone, ':CGNS#GlobalNumbering') is not None

  zone = PT.new_Zone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = MT.new_GlobalNumbering(gnum_arrays, zone)
  assert (PT.get_node_from_path(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert PT.get_node_from_path(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None