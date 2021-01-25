import Converter.Internal as I
import Generator.PyTree   as G
import maia.sids.sids     as SIDS
import numpy              as np


def test_ZoneType():
  #With numpy arrays
  zone_u = G.cartNGon((0., 0., 0.), (1., 1., 1.), (10, 5, 2))
  zone_s = G.cart((0., 0., 0.), (1., 1., 1.), (10, 5, 2))
  assert SIDS.ZoneType(zone_u) == 'Unstructured'
  assert SIDS.ZoneType(zone_s) == 'Structured'
  #With strings
  I.getNodeFromType1(zone_u, 'ZoneType_t')[1] = 'Unstructured'
  I.getNodeFromType1(zone_s, 'ZoneType_t')[1] = 'Structured'
  assert SIDS.ZoneType(zone_u) == 'Unstructured'
  assert SIDS.ZoneType(zone_s) == 'Structured'

def test_zone_u_size():
  zone_u = G.cartNGon((0., 0., 0.), (1., 1., 1.), (10, 5, 2))

  assert SIDS.VertexSize(zone_u) == 10*5*2
  assert SIDS.CellSize(zone_u) == 9*4*1
  assert SIDS.VertexBoundarySize(zone_u) == 0

  assert SIDS.zone_n_vtx(zone_u) == 10*5*2
  assert SIDS.zone_n_cell(zone_u) == 9*4*1
  assert SIDS.zone_n_vtx_bnd(zone_u) == 0


def test_zone_s_size():
  zone_s = G.cart((0., 0., 0.), (1., 1., 1.), (10, 5, 2))

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

