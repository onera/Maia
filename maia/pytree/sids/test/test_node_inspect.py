import pytest
import numpy              as np

from maia.pytree      import node as N
from maia.pytree      import walk as W

from maia.pytree.sids import node_inspect as SIDS

def test_ZoneType():
  #With numpy arrays
  zone_u = N.new_Zone('ZoneU', type='Unstructured')
  zone_s = N.new_Zone('ZoneS', type='Structured')
  assert SIDS.Zone.Type(zone_u) == 'Unstructured'
  assert SIDS.Zone.Type(zone_s) == 'Structured'
  #With strings
  W.get_child_from_label(zone_u, 'ZoneType_t')[1] = 'Unstructured'
  W.get_child_from_label(zone_s, 'ZoneType_t')[1] = 'Structured'
  assert SIDS.Zone.Type(zone_u) == 'Unstructured'
  assert SIDS.Zone.Type(zone_s) == 'Structured'

def test_IndexDimension():
  assert SIDS.Zone.IndexDimension(N.new_Zone('ZoneU', type='Unstructured', size=[[11,10,0]])) == 1
  assert SIDS.Zone.IndexDimension(N.new_Zone('ZoneS', type='Structured', size=[[10,9,0],[5,4,0],[2,1,0]])) == 3
  assert SIDS.Zone.IndexDimension(N.new_Zone('ZoneS', type='Structured', size=[[10,9,0],[5,4,0]])) == 2
  assert SIDS.Zone.IndexDimension(N.new_Zone('ZoneS', type='Structured', size=[[10,9,0]])) == 1

def test_NGonNode():
  zone = N.new_Zone('Zone', size=[[100, 36, 0]], type='Unstructured')
  with pytest.raises(RuntimeError):
    SIDS.Zone.NGonNode(zone)
  N.new_Elements('NGon',  type='NGON_n',  parent=zone)
  N.new_Elements('NFace', type='NFACE_n', parent=zone)
  ngon = SIDS.Zone.NGonNode(zone)
  assert N.get_name(ngon) == 'NGon' and N.get_value(ngon)[0] == 22
  N.new_Elements('NGon2', type='NGON_n', parent=zone)
  with pytest.raises(RuntimeError):
    SIDS.Zone.NGonNode(zone)

def test_ElementSize():
  elt1 = N.new_Elements(type='QUAD_4', erange=[1,100])
  elt2 = N.new_Elements(type='QUAD_4', erange=[15,15])
  assert SIDS.Element.Size(elt1) == 100
  assert SIDS.Element.Size(elt2) == 1

def test_ElementType():
  elt1 = N.new_Elements(type='QUAD_4', erange=[1,100])
  elt_type = SIDS.Element.Type(elt1)
  assert  isinstance(elt_type, int) and elt_type == 7

def test_ElementCGNSName():
  assert SIDS.Element.CGNSName(N.new_node("Toto", "Elements_t", [22, 0])) == "NGON_n"
  assert SIDS.Element.CGNSName(N.new_node("Toto", "Elements_t", [42, 0])) == "TRI_15"

def test_ElementDimension():
  assert SIDS.Element.Dimension(N.new_node("Toto", "Elements_t", [22, 0])) == 2
  assert SIDS.Element.Dimension(N.new_node("Toto", "Elements_t", [42, 0])) == 2
  assert SIDS.Element.Dimension(N.new_node("Toto", "Elements_t", [34, 0])) == 3

def test_ElementNVtx():
  assert SIDS.Element.NVtx(N.new_node("Toto", "Elements_t", [22, 0])) == None
  assert SIDS.Element.NVtx(N.new_node("Toto", "Elements_t", [42, 0])) == 15

def test_GridLocation():
  bc_no_loc = N.new_BC()
  bc_loc    = N.new_BC()
  N.new_GridLocation('JFaceCenter', bc_loc)
  assert SIDS.Subset.GridLocation(bc_no_loc) == 'Vertex'
  assert SIDS.Subset.GridLocation(bc_loc   ) == 'JFaceCenter'

def test_GridConnectivity_Type():
  gc = N.new_node("gc", "GridConnectivity1to1_t")
  assert SIDS.GridConnectivity.Type(gc) == "Abutting1to1"
  gc = N.new_node("gc", "GridConnectivity_t", 
      children=[N.new_node('GridConnectivityType', 'GridConnectivityType_t', "Abutting")])
  assert SIDS.GridConnectivity.Type(gc) == "Abutting"
  bc = N.new_BC("bc")
  with pytest.raises(Exception):
    SIDS.GridConnectivity.Type(bc)

def test_GridConnectivity_isperiodic():
  gc = N.new_GridConnectivity()
  assert SIDS.GridConnectivity.isperiodic(gc) == False
  N.new_GridConnectivityProperty(periodic={'translation' : [0., 0., 1.]}, parent=gc)
  assert SIDS.GridConnectivity.isperiodic(gc) == True

def test_GridConnectivity_periodic_values():
  gc = N.new_GridConnectivity()
  assert SIDS.GridConnectivity.periodic_values(gc) == (None, None, None)
  N.new_GridConnectivityProperty(parent=gc)
  assert SIDS.GridConnectivity.periodic_values(gc) == (None, None, None)
  W.rm_children_from_label(gc, 'GridConnectivityProperty_t')
  N.new_GridConnectivityProperty(periodic={'translation' : [0., 0., 1.]},parent=gc)
  assert (SIDS.GridConnectivity.periodic_values(gc)[0] == [0., 0., 0.]).all()
  assert (SIDS.GridConnectivity.periodic_values(gc)[1] == [0., 0., 0.]).all()
  assert (SIDS.GridConnectivity.periodic_values(gc)[2] == [0., 0., 1.]).all()

def test_zone_u_size():
  #Simulate a 10*5*2 vtx zone
  zone_u = N.new_Zone('Zone', size=[[100, 36, 0]], type='Unstructured')

  assert SIDS.Zone.VertexSize(zone_u) == 10*5*2
  assert SIDS.Zone.CellSize(zone_u) == 9*4*1
  assert SIDS.Zone.VertexBoundarySize(zone_u) == 0

  assert SIDS.Zone.n_vtx(zone_u) == 10*5*2
  assert SIDS.Zone.n_cell(zone_u) == 9*4*1
  assert SIDS.Zone.n_vtx_bnd(zone_u) == 0

def test_zone_s_size():
  #Simulate a 10*5*2 vtx zone
  zone_s = N.new_Zone('Zone', size=[[10,9,0], [5,4,0], [2,1,0]], type='Structured')

  assert np.all(SIDS.Zone.VertexSize(zone_s) == [10,5,2])
  assert np.all(SIDS.Zone.CellSize(zone_s) == [9,4,1])
  assert np.all(SIDS.Zone.VertexBoundarySize(zone_s) == [0,0,0])

  assert SIDS.Zone.n_vtx(zone_s) == 10*5*2
  assert SIDS.Zone.n_cell(zone_s) == 9*4*1
  assert SIDS.Zone.n_vtx_bnd(zone_s) == 0

def test_get_range_of_ngon():
  zone = N.new_Zone()
  N.new_Elements('ElemA', 'NGON_n',  erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', 'NFACE_n', erange=[1, 10], parent=zone)
  N.new_Elements('ElemC', 'HEXA_8',  erange=[54,60], parent=zone)
  assert (SIDS.Zone.get_range_of_ngon(zone) == [11,53]).all()

def test_get_ordered_elements():
  zone = N.new_Zone()
  N.new_Elements('ElemA', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', erange=[1, 10], parent=zone)
  N.new_Elements('ElemC', erange=[54,60], parent=zone)

  sorted_elems = SIDS.Zone.get_ordered_elements(zone)
  assert [N.get_name(elem) for elem in sorted_elems] == ['ElemB', 'ElemA', 'ElemC']

def test_has_ngon_elements():
  zone = N.new_Zone()
  N.new_Elements('ElemA', erange=[11, 53], parent=zone)
  assert not SIDS.Zone.has_ngon_elements(zone)
  N.new_Elements('ElemB', type='NGON_n', erange=[1, 11], parent=zone)
  assert SIDS.Zone.has_ngon_elements(zone)

def test_has_ngon_elements():
  zone = N.new_Zone()
  N.new_Elements('ElemA', type='NGON_n', erange=[1, 11], parent=zone)
  assert not SIDS.Zone.has_nface_elements(zone)
  N.new_Elements('ElemB', type='NFACE_n', erange=[12, 23], parent=zone)
  assert SIDS.Zone.has_nface_elements(zone)
    
def test_get_ordered_elements_per_dim():
  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)

  sorted_elems_per_dim = SIDS.Zone.get_ordered_elements_per_dim(zone)
  assert sorted_elems_per_dim[0] == []
  assert sorted_elems_per_dim[1] == []
  assert [N.get_name(elem) for elem in sorted_elems_per_dim[2]] == ['ElemC']
  assert [N.get_name(elem) for elem in sorted_elems_per_dim[3]] == ['ElemB', 'ElemA']
    
def test_get_elt_range_per_dim():
  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.get_elt_range_per_dim(zone) == [[0,0], [0,0], [54,60], [1,53]]

  zone = N.new_Zone()
  N.new_Elements('ElemA', type='QUAD_4', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.get_elt_range_per_dim(zone) == [[0,0], [0,0], [11,60], [1,10]]

  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='TRI_3',  erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  with pytest.raises(RuntimeError):
    SIDS.Zone.get_elt_range_per_dim(zone)

def test_elt_ordering_by_dim():
  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == -1

  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == 0

  zone = N.new_Zone()
  N.new_Elements('ElemA', type='HEXA_8', erange=[18, 60], parent=zone)
  N.new_Elements('ElemB', type='HEXA_8', erange=[8, 17],  parent=zone)
  N.new_Elements('ElemC', type='TRI_3',  erange=[1,7],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == 1

def test_PointRange():
  pr = N.new_PointRange('StandardPR', [1,3, 3,5, 1,3])
  assert (SIDS.PointRange.SizePerIndex(pr) == [3,3,3]).all()
  assert (SIDS.PointRange.n_elem(pr) == 3*3*3)

  pr = N.new_PointRange('GCLikePR', [7,1, 9,9, 5,1])
  assert (SIDS.PointRange.SizePerIndex(pr) == [7,1,5]).all()
  assert (SIDS.PointRange.n_elem(pr) == 7*1*5)

  pr = N.new_PointRange('ULike', [[1,15]]) # PR must be 2d
  assert (SIDS.PointRange.SizePerIndex(pr) == [15]).all()
  assert (SIDS.PointRange.n_elem(pr) == 15)

def test_PointList():
  pl = N.new_PointList('StandartPL', [[1,6,12]])
  assert SIDS.PointList.n_elem(pl) == 3

  pl = N.new_PointList('SLike', [[1,1,1,1,1], [1,1,1,2,2], [1,3,5,7,9]])
  assert SIDS.PointList.n_elem(pl) == 5

def test_Subset():
  sol = N.new_FlowSolution(loc='Vertex')
  pl = N.new_PointList('PointList', [[1,6,12]], parent=sol)

  assert SIDS.Subset.GridLocation(sol) == 'Vertex'
  assert SIDS.Subset.getPatch(sol) is pl
  assert SIDS.Subset.n_elem(sol) == SIDS.PointList.n_elem(pl)

  with pytest.raises(AssertionError):
    pr = N.new_PointRange('PointRange', [[1,15]], parent=sol)
    patch = SIDS.Subset.getPatch(sol)

