import pytest

import Converter.Internal as I
import numpy              as np

from maia.pytree.sids import node_inspect as SIDS

def test_ZoneType():
  #With numpy arrays
  zone_u = I.newZone('ZoneU', ztype='Unstructured')
  zone_s = I.newZone('ZoneS', ztype='Structured')
  assert SIDS.Zone.Type(zone_u) == 'Unstructured'
  assert SIDS.Zone.Type(zone_s) == 'Structured'
  #With strings
  I.getNodeFromType1(zone_u, 'ZoneType_t')[1] = 'Unstructured'
  I.getNodeFromType1(zone_s, 'ZoneType_t')[1] = 'Structured'
  assert SIDS.Zone.Type(zone_u) == 'Unstructured'
  assert SIDS.Zone.Type(zone_s) == 'Structured'

def test_NGonNode():
  zone = I.newZone('Zone', [[100, 36, 0]], 'Unstructured')
  with pytest.raises(RuntimeError):
    SIDS.Zone.NGonNode(zone)
  I.newElements('NGon',  etype='NGON',  parent=zone)
  I.newElements('NFace', etype='NFACE', parent=zone)
  ngon = SIDS.Zone.NGonNode(zone)
  assert I.getName(ngon) == 'NGon' and I.getValue(ngon)[0] == 22
  I.newElements('NGon2', etype='NGON', parent=zone)
  with pytest.raises(RuntimeError):
    SIDS.Zone.NGonNode(zone)

def test_ElementSize():
  elt1 = I.newElements(etype='QUAD', erange=[1,100])
  elt2 = I.newElements(etype='QUAD', erange=[15,15])
  assert SIDS.Element.Size(elt1) == 100
  assert SIDS.Element.Size(elt2) == 1

def test_ElementCGNSName():
  assert SIDS.Element.CGNSName(I.createNode("Toto", "Elements_t", [22, 0])) == "NGON_n"
  assert SIDS.Element.CGNSName(I.createNode("Toto", "Elements_t", [42, 0])) == "TRI_15"

def test_ElementDimension():
  assert SIDS.Element.Dimension(I.createNode("Toto", "Elements_t", [22, 0])) == 2
  assert SIDS.Element.Dimension(I.createNode("Toto", "Elements_t", [42, 0])) == 2
  assert SIDS.Element.Dimension(I.createNode("Toto", "Elements_t", [34, 0])) == 3

def test_ElementNVtx():
  assert SIDS.Element.NVtx(I.createNode("Toto", "Elements_t", [22, 0])) == None
  assert SIDS.Element.NVtx(I.createNode("Toto", "Elements_t", [42, 0])) == 15

def test_GridLocation():
  bc_no_loc = I.newBC()
  bc_loc    = I.newBC()
  I.newGridLocation('JFaceCenter', bc_loc)
  assert SIDS.Subset.GridLocation(bc_no_loc) == 'Vertex'
  assert SIDS.Subset.GridLocation(bc_loc   ) == 'JFaceCenter'

def test_GridConnectivity_Type():
  gc = I.newGridConnectivity1to1("gc")
  assert SIDS.GridConnectivity.Type(gc) == "Abutting1to1"
  gc = I.newGridConnectivity("gc", ctype="Abutting")
  assert SIDS.GridConnectivity.Type(gc) == "Abutting"
  bc = I.newBC("bc")
  with pytest.raises(Exception):
    SIDS.GridConnectivity.Type(bc)

def test_zone_u_size():
  #Simulate a 10*5*2 vtx zone
  zone_u = I.newZone('Zone', [[100, 36, 0]], 'Unstructured')

  assert SIDS.Zone.VertexSize(zone_u) == 10*5*2
  assert SIDS.Zone.CellSize(zone_u) == 9*4*1
  assert SIDS.Zone.VertexBoundarySize(zone_u) == 0

  assert SIDS.Zone.n_vtx(zone_u) == 10*5*2
  assert SIDS.Zone.n_cell(zone_u) == 9*4*1
  assert SIDS.Zone.n_vtx_bnd(zone_u) == 0

def test_zone_s_size():
  #Simulate a 10*5*2 vtx zone
  zone_s = I.newZone('Zone', [[10,9,0], [5,4,0], [2,1,0]], 'Structured')

  assert np.all(SIDS.Zone.VertexSize(zone_s) == [10,5,2])
  assert np.all(SIDS.Zone.CellSize(zone_s) == [9,4,1])
  assert np.all(SIDS.Zone.VertexBoundarySize(zone_s) == [0,0,0])

  assert SIDS.Zone.n_vtx(zone_s) == 10*5*2
  assert SIDS.Zone.n_cell(zone_s) == 9*4*1
  assert SIDS.Zone.n_vtx_bnd(zone_s) == 0

def test_get_range_of_ngon():
  zone = I.newZone()
  I.newElements('ElemA', 'NGON',  erange=[11, 53], parent=zone)
  I.newElements('ElemB', 'NFACE', erange=[1, 10], parent=zone)
  I.newElements('ElemC', 'HEXA',  erange=[54,60], parent=zone)
  assert (SIDS.Zone.get_range_of_ngon(zone) == [11,53]).all()

def test_get_ordered_elements():
  zone = I.newZone()
  I.newElements('ElemA', erange=[11, 53], parent=zone)
  I.newElements('ElemB', erange=[1, 10], parent=zone)
  I.newElements('ElemC', erange=[54,60], parent=zone)

  sorted_elems = SIDS.Zone.get_ordered_elements(zone)
  assert [I.getName(elem) for elem in sorted_elems] == ['ElemB', 'ElemA', 'ElemC']
    
def test_get_ordered_elements_per_dim():
  zone = I.newZone()
  I.newElements('ElemA', etype='HEXA', erange=[11, 53], parent=zone)
  I.newElements('ElemB', etype='HEXA', erange=[1, 10],  parent=zone)
  I.newElements('ElemC', etype='TRI',  erange=[54,60],  parent=zone)

  sorted_elems_per_dim = SIDS.Zone.get_ordered_elements_per_dim(zone)
  assert sorted_elems_per_dim[0] == []
  assert sorted_elems_per_dim[1] == []
  assert [I.getName(elem) for elem in sorted_elems_per_dim[2]] == ['ElemC']
  assert [I.getName(elem) for elem in sorted_elems_per_dim[3]] == ['ElemB', 'ElemA']
    
def test_get_elt_range_per_dim():
  zone = I.newZone()
  I.newElements('ElemA', etype='HEXA', erange=[11, 53], parent=zone)
  I.newElements('ElemB', etype='HEXA', erange=[1, 10],  parent=zone)
  I.newElements('ElemC', etype='TRI',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.get_elt_range_per_dim(zone) == [[0,0], [0,0], [54,60], [1,53]]

  zone = I.newZone()
  I.newElements('ElemA', etype='QUAD', erange=[11, 53], parent=zone)
  I.newElements('ElemB', etype='HEXA', erange=[1, 10],  parent=zone)
  I.newElements('ElemC', etype='TRI',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.get_elt_range_per_dim(zone) == [[0,0], [0,0], [11,60], [1,10]]

  zone = I.newZone()
  I.newElements('ElemA', etype='HEXA', erange=[11, 53], parent=zone)
  I.newElements('ElemB', etype='TRI',  erange=[1, 10],  parent=zone)
  I.newElements('ElemC', etype='TRI',  erange=[54,60],  parent=zone)
  with pytest.raises(RuntimeError):
    SIDS.Zone.get_elt_range_per_dim(zone)

def test_elt_ordering_by_dim():
  zone = I.newZone()
  I.newElements('ElemA', etype='HEXA', erange=[11, 53], parent=zone)
  I.newElements('ElemB', etype='HEXA', erange=[1, 10],  parent=zone)
  I.newElements('ElemC', etype='TRI',  erange=[54,60],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == -1

  zone = I.newZone()
  I.newElements('ElemA', etype='HEXA', erange=[11, 53], parent=zone)
  I.newElements('ElemB', etype='HEXA', erange=[1, 10],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == 0

  zone = I.newZone()
  I.newElements('ElemA', etype='HEXA', erange=[18, 60], parent=zone)
  I.newElements('ElemB', etype='HEXA', erange=[8, 17],  parent=zone)
  I.newElements('ElemC', etype='TRI',  erange=[1,7],  parent=zone)
  assert SIDS.Zone.elt_ordering_by_dim(zone) == 1

def test_PointRange():
  pr = I.newPointRange('StandardPR', [1,3, 3,5, 1,3])
  assert (SIDS.PointRange.SizePerIndex(pr) == [3,3,3]).all()
  assert (SIDS.PointRange.n_elem(pr) == 3*3*3)

  pr = I.newPointRange('GCLikePR', [7,1, 9,9, 5,1])
  assert (SIDS.PointRange.SizePerIndex(pr) == [7,1,5]).all()
  assert (SIDS.PointRange.n_elem(pr) == 7*1*5)

  pr = I.newPointRange('ULike', [[1,15]]) # PR must be 2d
  assert (SIDS.PointRange.SizePerIndex(pr) == [15]).all()
  assert (SIDS.PointRange.n_elem(pr) == 15)

def test_PointList():
  pl = I.newPointList('StandartPL', [[1,6,12]])
  assert SIDS.PointList.n_elem(pl) == 3

  pl = I.newPointList('SLike', [[1,1,1,1,1], [1,1,1,2,2], [1,3,5,7,9]])
  assert SIDS.PointList.n_elem(pl) == 5

def test_Subset():
  sol = I.newFlowSolution(gridLocation='Vertex')
  pl = I.newPointList('PointList', [[1,6,12]], parent=sol)

  assert SIDS.Subset.GridLocation(sol) == 'Vertex'
  assert SIDS.Subset.getPatch(sol) is pl
  assert SIDS.Subset.n_elem(sol) == SIDS.PointList.n_elem(pl)

  with pytest.raises(AssertionError):
    pr = I.newPointRange('PointRange', [[1,15]], parent=sol)
    patch = SIDS.Subset.getPatch(sol)

