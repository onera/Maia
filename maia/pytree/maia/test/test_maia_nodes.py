import pytest_parallel
import pytest
import numpy as np
import maia.pytree        as PT

from maia.utils import par_utils

from maia.pytree.maia import maia_nodes as mNode

def test_new_Distribution():
  distri = mNode.new_Distribution()
  assert PT.get_name(distri)  == ':CGNS#Distribution'
  assert PT.get_label(distri) == 'UserDefinedData_t'

  zone = PT.new_Zone('zone')
  distri = mNode.new_Distribution(parent=zone)
  assert PT.get_child_from_name(zone, ':CGNS#Distribution') is not None

  zone = PT.new_Zone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = mNode.new_Distribution(distri_arrays, zone)
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()

  zone = PT.new_Zone('zone')
  distri = mNode.new_Distribution({'Cell' : [0,10,20,30]}, parent=zone)
  distri = mNode.new_Distribution({'Cell' : [0,15,30]}, parent=zone) #Try update
  distri = mNode.new_Distribution({'Vertex' : [100,1000,1000]}, parent=zone)
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Cell')[1] == [0,15,30]).all()
  assert (PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex')[1] == [100, 1000, 1000]).all()
  assert len(PT.get_nodes_from_name(zone, ':CGNS#Distribution')) == 1

def test_new_GlobalNumbering():
  gnum = mNode.new_GlobalNumbering()
  assert PT.get_name(gnum)  == ':CGNS#GlobalNumbering'
  assert PT.get_label(gnum) == 'UserDefinedData_t'

  zone = PT.new_Zone('zone')
  gnum = mNode.new_GlobalNumbering(parent=zone)
  assert PT.get_child_from_name(zone, ':CGNS#GlobalNumbering') is not None

  zone = PT.new_Zone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum = mNode.new_GlobalNumbering(gnum_arrays, zone)
  assert (PT.get_node_from_path(zone, ':CGNS#GlobalNumbering/Cell')[1] == [4,21,1,2,8,12]).all()
  assert PT.get_node_from_path(zone, ':CGNS#GlobalNumbering/Vertex')[1] == None

def test_get_Distribution():
  zone = PT.new_Zone('zone')
  distri_arrays = {'Cell' : [0,15,30], 'Vertex' : [100,1000,1000]}
  distri = mNode.new_Distribution(distri_arrays, zone)
  assert mNode.get_Distribution(zone) is distri
  assert (mNode.distribution_value(zone, 'Cell') == [0,15,30]).all()
  assert (mNode.distribution_value(zone, 'Vertex') == [100,1000,1000]).all()

def test_get_GlobalNumbering():
  zone = PT.new_Zone('zone')
  gnum_arrays = {'Cell' : [4,21,1,2,8,12], 'Vertex' : None}
  gnum_node = mNode.new_GlobalNumbering(gnum_arrays, zone)
  assert mNode.get_GlobalNumbering(zone) is gnum_node
  assert (PT.get_value(mNode.get_GlobalNumbering(zone, 'Cell')) == [4,21,1,2,8,12]).all()
  assert  PT.get_value(mNode.get_GlobalNumbering(zone, 'Vertex')) == None

def test_get_edge_node():
  zone = PT.new_Zone('zone')
  with pytest.raises(RuntimeError):
    mNode.Zone.EdgeNode(zone)

  elt = PT.new_Elements('BAR', 'BAR_2', parent=zone)
  assert PT.is_same_node(elt, mNode.Zone.EdgeNode(zone))

  elt = PT.new_Elements('SECONDBAR', 'BAR_2', parent=zone)
  with pytest.raises(RuntimeError):
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
  mNode.new_Distribution({'Element' : distri_e, 'ElementConnectivity' : distri_ec}, parent=nface)
  
  cell_face = mNode.Element.connectivity(nface) # In //, recompute displs
  assert cell_face._values is ec
  if comm.rank == 0:
    assert np.array_equal(cell_face.displs, [0,6,12])
  elif comm.rank == 1:
    assert np.array_equal(cell_face.displs, [0,6])
 

@pytest_parallel.mark.parallel(2)
def test_maia_sizes(comm):
  dzone = PT.new_Zone(type='Unstructured', size=[[50,40,0]])
  distri_vtx = np.array([0,30,50]) if comm.rank == 0 else np.array([30,50,50])
  distri_cell = np.array([0,20,40]) if comm.rank == 0 else np.array([20,40,40])
  mNode.new_Distribution({'Vertex' : distri_vtx, 'Cell' : distri_cell}, parent=dzone)
  dbc1 = PT.new_BC(point_range=[[1,5], [4,4], [10,12]])
  delt = PT.new_Elements('TRI', 'TRI_3', erange=[10,20])
  distri_elt= np.array([0,4,10]) if comm.rank == 0 else np.array([4,10,10])
  mNode.new_Distribution({'Element' : distri_elt}, parent=delt)
  if comm.rank == 0:
    dbc2 = PT.new_BC(point_list=[[11,22]])
    mNode.new_Distribution({'Index' : np.array([0,0,15])}, parent=dbc1)
    mNode.new_Distribution({'Index' : np.array([0,2,5])}, parent=dbc2)
  else:
    dbc2 = PT.new_BC(point_list=[[33,44,55]])
    mNode.new_Distribution({'Index' : np.array([0,15,15])}, parent=dbc1)
    mNode.new_Distribution({'Index' : np.array([2,5,5])}, parent=dbc2)

  assert mNode.Zone.dn_vtx(dzone) == (30 if comm.rank == 0 else 20)
  assert mNode.Zone.dn_cell(dzone) == 20
  assert mNode.Subset.dn_elem(dbc1) == (0 if comm.rank == 0 else 15)
  assert mNode.Subset.dn_elem(dbc2) == (2 if comm.rank == 0 else 3)
  assert mNode.Element.dn_elt(delt) == (4 if comm.rank == 0 else 6)


  if comm.rank == 0:
    pzones = [PT.new_Zone(type='Unstructured', size=[[34,22,0]]),
              PT.new_Zone(type='Unstructured', size=[[22,18,0]])]
    gnum_cell_0 = np.ones(22) # Fake gnum
    gnum_cell_1 = np.ones(18) # Fake gnum
    gnum_cell_1[8] = 40
    gnum_vtx_0 = np.ones(34) # Fake gnum
    gnum_vtx_1 = np.ones(22) # Fake gnum
    gnum_vtx_0[32] = 50
    mNode.new_GlobalNumbering({'Cell': gnum_cell_0, 'Vertex': gnum_vtx_0}, parent=pzones[0])
    mNode.new_GlobalNumbering({'Cell': gnum_cell_1, 'Vertex': gnum_vtx_1}, parent=pzones[1])

    pbc1s = [PT.new_BC(point_range=[[1,2], [4,4], [10,12]]), PT.new_BC(point_range=[[1,1], [4,4], [10,12]])]
    pbc2s = [PT.new_BC(point_list=[[1,4,2]])]
    mNode.new_GlobalNumbering({'Index': np.array([1,2,3,4,5,6])}, parent=pbc1s[0])
    mNode.new_GlobalNumbering({'Index': np.array([7,8,9])}, parent=pbc1s[1])
    mNode.new_GlobalNumbering({'Index': np.array([4,2,3])}, parent=pbc2s[0])
    pelts = [PT.new_Elements('TRI', 'TRI_3', erange=[1,6])]
    mNode.new_GlobalNumbering({'Element': np.array([1,3,4,2,9,5])}, parent=pelts[0])

    assert mNode.Zone.pn_cell(pzones[0]) == 22
    assert mNode.Zone.pn_cell(pzones[1]) == 18
    assert mNode.Zone.pn_vtx(pzones[0]) == 34
    assert mNode.Zone.pn_vtx(pzones[1]) == 22

    assert mNode.Subset.pn_elem(pbc1s[0]) == 6
    assert mNode.Subset.pn_elem(pbc1s[1]) == 3
    assert mNode.Subset.pn_elem(pbc2s[0]) == 3

    assert mNode.Element.pn_elt(pelts[0]) == 6
  else:
    pzones = []
    pbc1s =  [PT.new_BC(point_range=[[1,2], [1,1], [1,3]])]
    pbc2s = [PT.new_BC(point_list=[[11,2,5]])]
    mNode.new_GlobalNumbering({'Index': np.array([10,11,12,13,14,15])}, parent=pbc1s[0])
    mNode.new_GlobalNumbering({'Index': np.array([2,1,5])}, parent=pbc2s[0])

    pelts = [PT.new_Elements('TRI', 'TRI_3', erange=[1,6])]
    mNode.new_GlobalNumbering({'Element': np.array([6,8,10,7,4,9])}, parent=pelts[0])

    assert mNode.Subset.pn_elem(pbc1s[0]) == 6
    assert mNode.Subset.pn_elem(pbc2s[0]) == 3

    assert mNode.Element.pn_elt(pelts[0]) == 6

  assert mNode.Zone.n_cell(dzone) == 40
  assert mNode.Zone.n_cell(pzones, comm) == 40
  assert mNode.Zone.n_vtx(dzone) == 50
  assert mNode.Zone.n_vtx(pzones, comm) == 50

  assert mNode.Subset.n_elem(dbc1) == 15
  assert mNode.Subset.n_elem(dbc2) == 5
  assert mNode.Subset.n_elem(pbc1s, comm) == 15
  assert mNode.Subset.n_elem(pbc2s, comm) == 5

  assert mNode.Element.n_elt(delt) == 10
  assert mNode.Element.n_elt(pelts, comm) == 10

  # Simulate case where GN are not created
  [PT.rm_nodes_from_name(n, ':CGNS#GlobalNumbering') for n in pbc1s]
  with pytest.raises(RuntimeError):
    mNode.Subset.n_elem(pbc1s, comm)