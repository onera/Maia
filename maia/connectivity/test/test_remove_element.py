import pytest

import Converter.Internal as I
import numpy as np

from maia.sids import sids
from maia.connectivity import remove_element as RME

def test_remove_element():
  zone = I.newZone()

  quad_ec = [1,2,6,5,2,3,7,6,3,4,8,7,5,6,10,9,6,7,11,10,7,8,12,11]
  quad = I.newElements('Quad', 'QUAD', quad_ec, [1,6], parent=zone)
  bar  = I.newElements('Bar', 'BAR', [10,9,11,10,12,11,5,1,9,5], [7,11], parent=zone)

  ngon_ec = [2,1,3,2,1,5,4,3,2,6,3,7,6,5,8,4,6,7,5,9,8,7,10,6,7,11,9,10,12,8,10,11,11,12]
  ngon = I.newElements('NGon', 'NGON', ngon_ec, [12,28], parent=zone)
  ngon_pe = np.array([[29,0],[30,0],[29,0],[31,0],[29,30],[30,31],[29,32],[31,0],[33,30],\
                      [32,0],[31,34],[33,32],[33,34],[32,0],[34,0],[33,0],[34,0]])
  expected_pe = np.copy(ngon_pe)
  expected_pe[np.where(ngon_pe > 0)] -= 5
  I.newDataArray('ElementStartOffset', np.arange(0,35,2), parent=ngon)
  I.newDataArray('ParentElements', ngon_pe, parent=ngon)

  nface_ec = [7,5,1,3,6,2,5,9,11,4,6,8,7,10,12,14,13,12,9,16,11,13,15,17]
  nface = I.newElements('NFace', 'NFACE', nface_ec, [29,34], parent=zone)
  I.newDataArray('ElementStartOffset', np.arange(0,24+1,4), parent=nface)

  zbc = I.newZoneBC(zone)
  bc = I.newBC(pointList = [[25,27,28]], parent=zbc)
  I.newGridLocation('EdgeCenter', bc)

  RME.remove_element(zone, bar)
  assert (sids.ElementRange(quad)  == [1,6]).all()
  assert (sids.ElementRange(ngon)  == [12-5,28-5]).all()
  assert (sids.ElementRange(nface) == [29-5,34-5]).all()
  assert (I.getNodeFromName1(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (I.getNodeFromName(zone, 'PointList')[1] == [[20,22,23]]).all()
  assert I.getNodeFromName(zone, 'Bar') is None
