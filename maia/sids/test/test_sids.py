import pytest

import Converter.Internal as I
import maia.sids.sids     as SIDS
import numpy              as np


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

def test_ElementSize():
  elt1 = I.newElements(etype='QUAD', erange=[1,100])
  elt2 = I.newElements(etype='QUAD', erange=[15,15])
  assert SIDS.ElementSize(elt1) == 100
  assert SIDS.ElementSize(elt2) == 1

def test_ElementCGNSName():
  assert SIDS.ElementCGNSName(I.createNode("Toto", "Elements_t", [22, 0])) == "NGON_n"
  assert SIDS.ElementCGNSName(I.createNode("Toto", "Elements_t", [42, 0])) == "TRI_15"

def test_ElementDimension():
  assert SIDS.ElementDimension(I.createNode("Toto", "Elements_t", [22, 0])) == 2
  assert SIDS.ElementDimension(I.createNode("Toto", "Elements_t", [42, 0])) == 2
  assert SIDS.ElementDimension(I.createNode("Toto", "Elements_t", [34, 0])) == 3

def test_ElementNVtx():
  assert SIDS.ElementNVtx(I.createNode("Toto", "Elements_t", [22, 0])) == None
  assert SIDS.ElementNVtx(I.createNode("Toto", "Elements_t", [42, 0])) == 15

def test_GridLocation():
  bc_no_loc = I.newBC()
  bc_loc    = I.newBC()
  I.newGridLocation('JFaceCenter', bc_loc)
  assert SIDS.GridLocation(bc_no_loc) == 'Vertex'
  assert SIDS.GridLocation(bc_loc   ) == 'JFaceCenter'

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

