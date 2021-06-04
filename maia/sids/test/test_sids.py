import Converter.Internal as I
import maia.sids.sids     as SIDS
import numpy              as np


def test_ZoneType():
  #With numpy arrays
  zone_u = I.newZone('ZoneU', ztype='Unstructured')
  zone_s = I.newZone('ZoneS', ztype='Structured')
  assert SIDS.ZoneType(zone_u) == 'Unstructured'
  assert SIDS.ZoneType(zone_s) == 'Structured'
  #With strings
  I.getNodeFromType1(zone_u, 'ZoneType_t')[1] = 'Unstructured'
  I.getNodeFromType1(zone_s, 'ZoneType_t')[1] = 'Structured'
  assert SIDS.ZoneType(zone_u) == 'Unstructured'
  assert SIDS.ZoneType(zone_s) == 'Structured'

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

  assert SIDS.VertexSize(zone_u) == 10*5*2
  assert SIDS.CellSize(zone_u) == 9*4*1
  assert SIDS.VertexBoundarySize(zone_u) == 0

  assert SIDS.zone_n_vtx(zone_u) == 10*5*2
  assert SIDS.zone_n_cell(zone_u) == 9*4*1
  assert SIDS.zone_n_vtx_bnd(zone_u) == 0


def test_zone_s_size():
  #Simulate a 10*5*2 vtx zone
  zone_s = I.newZone('Zone', [[10,9,0], [5,4,0], [2,1,0]], 'Structured')

  assert np.all(SIDS.VertexSize(zone_s) == [10,5,2])
  assert np.all(SIDS.CellSize(zone_s) == [9,4,1])
  assert np.all(SIDS.VertexBoundarySize(zone_s) == [0,0,0])

  assert SIDS.zone_n_vtx(zone_s) == 10*5*2
  assert SIDS.zone_n_cell(zone_s) == 9*4*1
  assert SIDS.zone_n_vtx_bnd(zone_s) == 0

def test_point_range():
  pr = I.newPointRange('Standard', [1,3, 3,5, 1,3])
  assert (SIDS.point_range_sizes(pr) == [3,3,3]).all()
  assert SIDS.point_range_n_elt(pr) == 3*3*3
  pr = I.newPointRange('BCLike', [5,5, 2,4, 1,1])
  assert (SIDS.point_range_sizes(pr) == [1,3,1]).all()
  assert SIDS.point_range_n_elt(pr) == 1*3*1
  pr = I.newPointRange('Reversed', [3,1, 5,3, 1,3])
  assert (SIDS.point_range_sizes(pr) == [3,3,3]).all()
  assert SIDS.point_range_n_elt(pr) == 3*3*3
  pr = I.newPointRange('GCLike', [7,1, 9,9, 5,1])
  assert (SIDS.point_range_sizes(pr) == [7,1,5]).all()
  assert SIDS.point_range_n_elt(pr) == 7*1*5

