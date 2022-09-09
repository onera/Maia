import pytest
import numpy              as np

from maia.pytree      import walk
from maia.pytree      import create_nodes as CN
from maia.pytree      import nodes_attr as NA
from maia.pytree.sids import node_inspect as SIDS

def test_ZoneType():
  #With numpy arrays
  zone_u = CN.new_Zone('ZoneU', type='Unstructured')
  zone_s = CN.new_Zone('ZoneS', type='Structured')
  assert SIDS.Zone.Type(zone_u) == 'Unstructured'
  assert SIDS.Zone.Type(zone_s) == 'Structured'
  #With strings
  walk.get_child_from_label(zone_u, 'ZoneType_t')[1] = 'Unstructured'
  walk.get_child_from_label(zone_s, 'ZoneType_t')[1] = 'Structured'
  assert SIDS.Zone.Type(zone_u) == 'Unstructured'
  assert SIDS.Zone.Type(zone_s) == 'Structured'

def test_NGonNode():
  zone = CN.new_Zone('Zone', size=[[100, 36, 0]], type='Unstructured')
  with pytest.raises(RuntimeError):
    SIDS.Zone.NGonNode(zone)
  CN.new_Elements('NGon',  type='NGON_n',  parent=zone)
  CN.new_Elements('NFace', type='NFACE_n', parent=zone)
  ngon = SIDS.Zone.NGonNode(zone)
  assert NA.get_name(ngon) == 'NGon' and NA.get_value(ngon)[0] == 22
  CN.new_Elements('NGon2', type='NGON_n', parent=zone)
  with pytest.raises(RuntimeError):
    SIDS.Zone.NGonNode(zone)

def test_ElementSize():
  elt1 = CN.new_Elements(type='QUAD_4', erange=[1,100])
  elt2 = CN.new_Elements(type='QUAD_4', erange=[15,15])
  assert SIDS.Element.Size(elt1) == 100
  assert SIDS.Element.Size(elt2) == 1

def test_ElementCGNSName():
  assert SIDS.Element.CGNSName(CN.new_node("Toto", "Elements_t", [22, 0])) == "NGON_n"
  assert SIDS.Element.CGNSName(CN.new_node("Toto", "Elements_t", [42, 0])) == "TRI_15"

def test_ElementDimension():
  assert SIDS.Element.Dimension(CN.new_node("Toto", "Elements_t", [22, 0])) == 2
  assert SIDS.Element.Dimension(CN.new_node("Toto", "Elements_t", [42, 0])) == 2
  assert SIDS.Element.Dimension(CN.new_node("Toto", "Elements_t", [34, 0])) == 3

def test_ElementNVtx():
  assert SIDS.Element.NVtx(CN.new_node("Toto", "Elements_t", [22, 0])) == None
  assert SIDS.Element.NVtx(CN.new_node("Toto", "Elements_t", [42, 0])) == 15

def test_GridLocation():
  bc_no_loc = CN.new_BC()
  bc_loc    = CN.new_BC()
  CN.new_GridLocation('JFaceCenter', bc_loc)
  assert SIDS.Subset.GridLocation(bc_no_loc) == 'Vertex'
  assert SIDS.Subset.GridLocation(bc_loc   ) == 'JFaceCenter'

def test_GridConnectivity_Type():
  gc = CN.new_node("gc", "GridConnectivity1to1_t")
  assert SIDS.GridConnectivity.Type(gc) == "Abutting1to1"
  gc = CN.new_node("gc", "GridConnectivity_t", 
      children=[CN.new_node('GridConnectivityType', 'GridConnectivityType_t', "Abutting")])
  assert SIDS.GridConnectivity.Type(gc) == "Abutting"
  bc = CN.new_BC("bc")
  with pytest.raises(Exception):
    SIDS.GridConnectivity.Type(bc)

def test_zone_u_size():
  #Simulate a 10*5*2 vtx zone
  zone_u = CN.new_Zone('Zone', size=[[100, 36, 0]], type='Unstructured')

  assert SIDS.Zone.VertexSize(zone_u) == 10*5*2
  assert SIDS.Zone.CellSize(zone_u) == 9*4*1
  assert SIDS.Zone.VertexBoundarySize(zone_u) == 0

  assert SIDS.Zone.n_vtx(zone_u) == 10*5*2
  assert SIDS.Zone.n_cell(zone_u) == 9*4*1
  assert SIDS.Zone.n_vtx_bnd(zone_u) == 0

def test_zone_s_size():
  #Simulate a 10*5*2 vtx zone
  zone_s = CN.new_Zone('Zone', size=[[10,9,0], [5,4,0], [2,1,0]], type='Structured')

  assert np.all(SIDS.Zone.VertexSize(zone_s) == [10,5,2])
  assert np.all(SIDS.Zone.CellSize(zone_s) == [9,4,1])
  assert np.all(SIDS.Zone.VertexBoundarySize(zone_s) == [0,0,0])

  assert SIDS.Zone.n_vtx(zone_s) == 10*5*2
  assert SIDS.Zone.n_cell(zone_s) == 9*4*1
  assert SIDS.Zone.n_vtx_bnd(zone_s) == 0

def test_get_range_of_ngon():
  zone = CN.new_Zone()
  CN.new_Elements('ElemA', 'NGON_n',  erange=[11, 53], parent=zone)
  CN.new_Elements('ElemB', 'NFACE_n', erange=[1, 10], parent=zone)
  CN.new_Elements('ElemC', 'HEXA_8',  erange=[54,60], parent=zone)
  assert (SIDS.Zone.get_range_of_ngon(zone) == [11,53]).all()

def test_get_ordered_elements():
  zone = CN.new_Zone()
  CN.new_Elements('ElemA', erange=[11, 53], parent=zone)
  CN.new_Elements('ElemB', erange=[1, 10], parent=zone)
  CN.new_Elements('ElemC', erange=[54,60], parent=zone)

  sorted_elems = SIDS.Zone.get_ordered_elements(zone)
  assert [NA.get_name(elem) for elem in sorted_elems] == ['ElemB', 'ElemA', 'ElemC']
    
def test_get_ordered_elements_per_dim():
  zone = CN.new_Zone()
  CN.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  CN.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  CN.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)

  sorted_elems_per_dim = SIDS.Zone.get_ordered_elements_per_dim(zone)
  assert sorted_elems_per_dim[0] == []
  assert sorted_elems_per_dim[1] == []
  assert [NA.get_name(elem) for elem in sorted_elems_per_dim[2]] == ['ElemC']
  assert [NA.get_name(elem) for elem in sorted_elems_per_dim[3]] == ['ElemB', 'ElemA']
    
def test_get_elt_range_per_dim():
  zone = CN.new_Zone()
  CN.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  CN.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  CN.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.get_elt_range_per_dim(zone) == [[0,0], [0,0], [54,60], [1,53]]

  zone = CN.new_Zone()
  CN.new_Elements('ElemA', type='QUAD_4', erange=[11, 53], parent=zone)
  CN.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  CN.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.get_elt_range_per_dim(zone) == [[0,0], [0,0], [11,60], [1,10]]

  zone = CN.new_Zone()
  CN.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  CN.new_Elements('ElemB', type='TRI_3',  erange=[1, 10],  parent=zone)
  CN.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  with pytest.raises(RuntimeError):
    SIDS.Zone.get_elt_range_per_dim(zone)

def test_elt_ordering_by_dim():
  zone = CN.new_Zone()
  CN.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  CN.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  CN.new_Elements('ElemC', type='TRI_3',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == -1

  zone = CN.new_Zone()
  CN.new_Elements('ElemA', type='HEXA_8', erange=[11, 53], parent=zone)
  CN.new_Elements('ElemB', type='HEXA_8', erange=[1, 10],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == 0

  zone = CN.new_Zone()
  CN.new_Elements('ElemA', type='HEXA_8', erange=[18, 60], parent=zone)
  CN.new_Elements('ElemB', type='HEXA_8', erange=[8, 17],  parent=zone)
  CN.new_Elements('ElemC', type='TRI_3',  erange=[1,7],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == 1

def test_PointRange():
  pr = CN.new_PointRange('StandardPR', [1,3, 3,5, 1,3])
  assert (SIDS.PointRange.SizePerIndex(pr) == [3,3,3]).all()
  assert (SIDS.PointRange.n_elem(pr) == 3*3*3)

  pr = CN.new_PointRange('GCLikePR', [7,1, 9,9, 5,1])
  assert (SIDS.PointRange.SizePerIndex(pr) == [7,1,5]).all()
  assert (SIDS.PointRange.n_elem(pr) == 7*1*5)

  pr = CN.new_PointRange('ULike', [[1,15]]) # PR must be 2d
  assert (SIDS.PointRange.SizePerIndex(pr) == [15]).all()
  assert (SIDS.PointRange.n_elem(pr) == 15)

def test_PointList():
  pl = CN.new_PointList('StandartPL', [[1,6,12]])
  assert SIDS.PointList.n_elem(pl) == 3

  pl = CN.new_PointList('SLike', [[1,1,1,1,1], [1,1,1,2,2], [1,3,5,7,9]])
  assert SIDS.PointList.n_elem(pl) == 5

def test_Subset():
  sol = CN.new_FlowSolution(loc='Vertex')
  pl = CN.new_PointList('PointList', [[1,6,12]], parent=sol)

  assert SIDS.Subset.GridLocation(sol) == 'Vertex'
  assert SIDS.Subset.getPatch(sol) is pl
  assert SIDS.Subset.n_elem(sol) == SIDS.PointList.n_elem(pl)

  with pytest.raises(AssertionError):
    pr = CN.new_PointRange('PointRange', [[1,15]], parent=sol)
    patch = SIDS.Subset.getPatch(sol)

