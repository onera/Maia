import Converter.Internal  as I
import Generator.PyTree    as G
import maia.utils          as UTL


def test_get_zone_nb_u():

  zone_u = G.cartNGon((0., 0., 0.), (1., 1., 1.), (10, 5, 2))

  nvtx = UTL.get_zone_nb_vtx(zone_u)
  assert nvtx == 10*5*2

  ncell = UTL.get_zone_nb_cell(zone_u)
  assert ncell == 9*4*1

  nvtx_bnd = UTL.get_zone_nb_vtx_bnd(zone_u)
  assert nvtx_bnd == 0

def test_get_zone_nb_s():

  zone_s = G.cart((0., 0., 0.), (1., 1., 1.), (20, 3, 5))

  nvtx = UTL.get_zone_nb_vtx(zone_s)
  assert nvtx == 20*3*5

  ncell = UTL.get_zone_nb_cell(zone_s)
  assert ncell == 19*2*4

  nvtx_bnd = UTL.get_zone_nb_vtx_bnd(zone_s)
  assert nvtx_bnd == 0

