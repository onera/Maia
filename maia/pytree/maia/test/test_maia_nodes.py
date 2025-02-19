import pytest_parallel
import pytest
import numpy as np
import maia.pytree        as PT

from maia.utils import par_utils

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

def test_get_edge_node():
  zone = PT.new_Zone('zone')
  with pytest.raises(AssertionError):
    mNode.Zone.EdgeNode(zone)

  elt = PT.new_Elements('BAR', 'BAR_2', parent=zone)
  assert PT.is_same_node(elt, mNode.Zone.EdgeNode(zone))

  elt = PT.new_Elements('SECONDBAR', 'BAR_2', parent=zone)
  with pytest.raises(AssertionError):
    mNode.Zone.EdgeNode(zone)

@pytest_parallel.mark.parallel(2)
def test_element_connectivity(comm):
  eso = np.array([0,6,12,18], np.int32)
  ec = np.array([1, 2, 7, 8, 13, 14,  -14, 3, 4, 9, 10, 15,  -15, 5, 6, 11, 12, 16], np.int32)
  nface = PT.new_NFaceElements(eso=eso, ec=ec)
  cell_face = mNode.Element.connectivity(nface)

  assert cell_face._values is ec
  assert cell_face._displs is eso

  ec = np.array([1, 2, 6, 5,   5, 6, 10, 9,   10, 14, 13, 9], np.int32)
  quad = PT.new_Elements(type='QUAD_4', econn=ec)
  cell_vtx = mNode.Element.connectivity(quad) # Auto compute counts for std elts
  assert cell_vtx._values is ec
  assert np.array_equal(cell_vtx.displs, [0,4,8,12]) and cell_vtx.displs.dtype == np.int32
  
  if comm.rank == 0:
    eso = np.array([0,6,12], np.int32)
    ec = np.array([1, 2, 7, 8, 13, 14,  -14, 3, 4, 9, 10, 15], np.int32)
  elif comm.rank == 1:
    eso = np.array([12,18], np.int32)
    ec = np.array([-15, 5, 6, 11, 12, 16], np.int32)
  distri_e  = par_utils.full_to_partial_distribution(np.array([0,2,3]), comm)
  distri_ec = par_utils.full_to_partial_distribution(np.array([0,12,18]), comm)
  nface = PT.new_NFaceElements(eso=eso, ec=ec)
  mNode.new_distribution({'Element' : distri_e, 'ElementConnectivity' : distri_ec}, parent=nface)
  
  cell_face = mNode.Element.connectivity(nface) # In //, recompute displs
  assert cell_face._values is ec
  if comm.rank == 0:
    assert np.array_equal(cell_face.displs, [0,6,12])
  elif comm.rank == 1:
    assert np.array_equal(cell_face.displs, [0,6])
 
  from maia.pytree.meta import CGNSLabelNotEqualError
  with pytest.raises(CGNSLabelNotEqualError):
    mNode.Element.connectivity(PT.new_Zone())