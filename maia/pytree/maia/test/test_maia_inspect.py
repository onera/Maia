import pytest_parallel
import pytest
import numpy as np
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import par_utils

def test_get_edge_node():
  zone = PT.new_Zone('zone')
  with pytest.raises(RuntimeError):
    MT.Zone.EdgeNode(zone)

  elt = PT.new_Elements('BAR', 'BAR_2', parent=zone)
  assert PT.is_same_node(elt, MT.Zone.EdgeNode(zone))

  elt = PT.new_Elements('SECONDBAR', 'BAR_2', parent=zone)
  with pytest.raises(RuntimeError):
    MT.Zone.EdgeNode(zone)

@pytest_parallel.mark.parallel(2)
def test_element_connectivity(comm):
  eso = np.array([0,6,12,18], np.int32)
  ec = np.array([1, 2, 7, 8, 13, 14,  -14, 3, 4, 9, 10, 15,  -15, 5, 6, 11, 12, 16], np.int32)
  nface = PT.new_NFaceElements(eso=eso, ec=ec)
  cell_face = MT.Element.connectivity(nface)

  assert cell_face._values is ec
  assert cell_face._displs is eso

  ec = np.array([1, 2, 6, 5,   5, 6, 10, 9,   10, 14, 13, 9], np.int32)
  quad = PT.new_Elements(type='QUAD_4', econn=ec)
  cell_vtx = MT.Element.connectivity(quad) # Auto compute counts for std elts
  assert cell_vtx._values is ec
  assert np.array_equal(cell_vtx.displs, [0,4,8,12]) and cell_vtx.displs.dtype == np.int32
  
  if comm.rank == 0:
    eso = np.array([0,6,12], np.int32)
    ec = np.array([1, 2, 7, 8, 13, 14,  -14, 3, 4, 9, 10, 15], np.int32)
  elif comm.rank == 1:
    eso = np.array([12,18], np.int32)
    ec = np.array([-15, 5, 6, 11, 12, 16], np.int32)
  distri_e  = par_utils.full_to_partial_distribution(np.array([0,2,3]), comm)
  nface = PT.new_NFaceElements(eso=eso, ec=ec)
  MT.new_Distribution({'Element' : distri_e}, parent=nface)
  
  cell_face = MT.Element.connectivity(nface) # In //, recompute displs
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
  MT.new_Distribution({'Vertex' : distri_vtx, 'Cell' : distri_cell}, parent=dzone)
  dbc1 = PT.new_BC(point_range=[[1,5], [4,4], [10,12]])
  delt = PT.new_Elements('TRI', 'TRI_3', erange=[10,20])
  distri_elt= np.array([0,4,10]) if comm.rank == 0 else np.array([4,10,10])
  MT.new_Distribution({'Element' : distri_elt}, parent=delt)
  if comm.rank == 0:
    dbc2 = PT.new_BC(point_list=[[11,22]])
    MT.new_Distribution({'Index' : np.array([0,0,15])}, parent=dbc1)
    MT.new_Distribution({'Index' : np.array([0,2,5])}, parent=dbc2)
  else:
    dbc2 = PT.new_BC(point_list=[[33,44,55]])
    MT.new_Distribution({'Index' : np.array([0,15,15])}, parent=dbc1)
    MT.new_Distribution({'Index' : np.array([2,5,5])}, parent=dbc2)

  assert MT.Zone.dn_vtx(dzone) == (30 if comm.rank == 0 else 20)
  assert MT.Zone.dn_cell(dzone) == 20
  assert MT.Subset.dn_elem(dbc1) == (0 if comm.rank == 0 else 15)
  assert MT.Subset.dn_elem(dbc2) == (2 if comm.rank == 0 else 3)
  assert MT.Element.dn_elt(delt) == (4 if comm.rank == 0 else 6)


  if comm.rank == 0:
    pzones = [PT.new_Zone(type='Unstructured', size=[[34,22,0]]),
              PT.new_Zone(type='Unstructured', size=[[22,18,0]])]
    gnum_cell_0 = np.ones(22) # Fake gnum
    gnum_cell_1 = np.ones(18) # Fake gnum
    gnum_cell_1[8] = 40
    gnum_vtx_0 = np.ones(34) # Fake gnum
    gnum_vtx_1 = np.ones(22) # Fake gnum
    gnum_vtx_0[32] = 50
    MT.new_GlobalNumbering({'Cell': gnum_cell_0, 'Vertex': gnum_vtx_0}, parent=pzones[0])
    MT.new_GlobalNumbering({'Cell': gnum_cell_1, 'Vertex': gnum_vtx_1}, parent=pzones[1])

    pbc1s = [PT.new_BC(point_range=[[1,2], [4,4], [10,12]]), PT.new_BC(point_range=[[1,1], [4,4], [10,12]])]
    pbc2s = [PT.new_BC(point_list=[[1,4,2]])]
    MT.new_GlobalNumbering({'Index': np.array([1,2,3,4,5,6])}, parent=pbc1s[0])
    MT.new_GlobalNumbering({'Index': np.array([7,8,9])}, parent=pbc1s[1])
    MT.new_GlobalNumbering({'Index': np.array([4,2,3])}, parent=pbc2s[0])
    pelts = [PT.new_Elements('TRI', 'TRI_3', erange=[1,6])]
    MT.new_GlobalNumbering({'Element': np.array([1,3,4,2,9,5])}, parent=pelts[0])

    assert MT.Zone.pn_cell(pzones[0]) == 22
    assert MT.Zone.pn_cell(pzones[1]) == 18
    assert MT.Zone.pn_vtx(pzones[0]) == 34
    assert MT.Zone.pn_vtx(pzones[1]) == 22

    assert MT.Subset.pn_elem(pbc1s[0]) == 6
    assert MT.Subset.pn_elem(pbc1s[1]) == 3
    assert MT.Subset.pn_elem(pbc2s[0]) == 3

    assert MT.Element.pn_elt(pelts[0]) == 6
  else:
    pzones = []
    pbc1s =  [PT.new_BC(point_range=[[1,2], [1,1], [1,3]])]
    pbc2s = [PT.new_BC(point_list=[[11,2,5]])]
    MT.new_GlobalNumbering({'Index': np.array([10,11,12,13,14,15])}, parent=pbc1s[0])
    MT.new_GlobalNumbering({'Index': np.array([2,1,5])}, parent=pbc2s[0])

    pelts = [PT.new_Elements('TRI', 'TRI_3', erange=[1,6])]
    MT.new_GlobalNumbering({'Element': np.array([6,8,10,7,4,9])}, parent=pelts[0])

    assert MT.Subset.pn_elem(pbc1s[0]) == 6
    assert MT.Subset.pn_elem(pbc2s[0]) == 3

    assert MT.Element.pn_elt(pelts[0]) == 6

  assert MT.Zone.n_cell(dzone) == 40
  assert MT.Zone.n_cell(pzones, comm) == 40
  assert MT.Zone.n_vtx(dzone) == 50
  assert MT.Zone.n_vtx(pzones, comm) == 50

  assert MT.Subset.n_elem(dbc1) == 15
  assert MT.Subset.n_elem(dbc2) == 5
  assert MT.Subset.n_elem(pbc1s, comm) == 15
  assert MT.Subset.n_elem(pbc2s, comm) == 5

  assert MT.Element.n_elt(delt) == 10
  assert MT.Element.n_elt(pelts, comm) == 10

  # Simulate case where GN are not created
  [PT.rm_nodes_from_name(n, ':CGNS#GlobalNumbering') for n in pbc1s]
  with pytest.raises(RuntimeError):
    MT.Subset.n_elem(pbc1s, comm)

def test_container_distribution():
  zone = PT.yaml.to_node("""
  Zone Zone_t [[1,1,0]]:
    ZoneType ZoneType_t "Unstructured":
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [1]:
      Cell DataArray_t I4 [1]:
    FlowSolVtx FlowSolution_t:
      GridLocation GridLocation_t "Vertex":
    PartialFlowSolFace FlowSolution_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[1]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t I4 [2]:
    FlowSolCell FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
    PartialFlowSolVtx FlowSolution_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[1]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t I4 [4]:
    WrongFlowSolCell FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[1]]:
  """)
  node_names     = ['FlowSolVtx','PartialFlowSolFace','FlowSolCell','PartialFlowSolVtx']
  expected_vals  = [[1]    ,      [2],                 [1],          [4]]
  for node_name, expected_val in zip(node_names, expected_vals):
    node = PT.get_node_from_name(zone, node_name)
    assert (MT.Container.distribution(node, zone) == expected_val).all()
  with pytest.raises(Exception):
    node = PT.get_node_from_name(zone, 'WrongFlowSolCell')
    MT.Container.distribution(node, zone)


def test_subset_distributed_pl():
  # Unstruct
  bc = PT.new_BC(point_list=[[4,5,6,7]])
  MT.new_Distribution({'Index' : [8,12,12]}, bc)
  assert (MT.Subset.distributed_pointlist(bc) == [[4,5,6,7]]).all()
  bc = PT.new_BC(point_range=[[20,40]])
  MT.new_Distribution({'Index' : [8,12,20]}, bc)
  assert (MT.Subset.distributed_pointlist(bc) == [[28,29,30,31]]).all()

  # Struct
  bc = PT.new_BC(point_list=np.array([[1,1,1,1],[5,7,9,11]], order='F'))
  MT.new_Distribution({'Index' : [8,12,20]}, bc)
  assert (MT.Subset.distributed_pointlist(bc) == [[1,1,1,1], [5,7,9,11]]).all()

  bc = PT.new_BC(point_range=np.array([[1,1],[6,25]], order='F'))
  MT.new_Distribution({'Index' : [8,12,20]}, bc)
  assert (MT.Subset.distributed_pointlist(bc) == [[1,1,1,1], [14,15,16,17]]).all()

  bc = PT.new_BC(point_range=np.array([[3,1],[6,25], [5,6]], order='F'))
  MT.new_Distribution({'Index' : [8,14,120]}, bc)
  assert (MT.Subset.distributed_pointlist(bc) == [[1,3,2,1,3,2], [8,9,9,9,10,10], [5,5,5,5,5,5]]).all()
