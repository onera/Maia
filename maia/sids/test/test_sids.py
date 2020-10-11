import Converter.Internal as I
import Generator.PyTree   as G
import maia.sids.sids     as SIDS
import numpy              as np


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
