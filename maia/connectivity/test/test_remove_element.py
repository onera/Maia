import pytest
from pytest_mpi_check._decorator import mark_mpi_test

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

@mark_mpi_test(1)
def test_remove_ngons(sub_comm):
  #Generated from G.cartNGon((0,0,0), (1,1,0), (3,4,1))
  # TODO handwritten ngon (4_cubes?)

  ec = [1,4, 2,5, 3,6, 4,7, 5,8, 6,9, 7,10, 8,11, 9,12, 1,2, 2,3, 4,5, 5,6, 7,8, 8,9, 10,11, 11,12]
  pe = np.array([[1,0], [1,2], [2,0], [3,0], [3,4], [4,0], [5,0], [5,6], [6,0],
                 [1,0], [2,0], [1,3], [2,4], [3,5], [4,6], [5,0], [6,0]])
  ngon = I.newElements('NGon', 'NGON', ec, [1,17])
  I.newDataArray('ElementStartOffset', np.arange(0, 35, 2), parent=ngon)
  I.newDataArray('ParentElements', pe, parent=ngon)
  I.newIndexArray('ElementConnectivity#Size', [34], parent=ngon)
  distri = I.createUniqueChild(ngon, ':CGNS#Distribution', 'UserDefinedData_t')
  I.newDataArray('Element', [0, 17, 17], parent=distri)
  I.newDataArray('ElementConnectivity', [0, 34, 34], parent=distri)

  RME.remove_ngons(ngon, [1,15], sub_comm)

  expected_ec = [1,4,    3,6, 4,7, 5,8, 6,9, 7,10, 8,11, 9,12, 1,2, 2,3, 4,5, 5,6, 7,8, 8,9,      11,12]
  expected_pe = np.array([[1,0],        [2,0], [3,0], [3,4], [4,0], [5,0], [5,6], [6,0],
                          [1,0], [2,0], [1,3], [2,4], [3,5], [4,6],        [6,0]])
  assert (I.getNodeFromName(ngon, 'ElementRange')[1] == [1, 15]).all()
  assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1] == expected_ec).all()
  assert (I.getNodeFromName(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (I.getNodeFromName(ngon, 'ElementStartOffset')[1] == np.arange(0,31,2)).all()
  assert  I.getNodeFromName(ngon, 'ElementConnectivity#Size')[1] == 34 - 2*2
  assert (I.getNodeFromPath(ngon, ':CGNS#Distribution/Element')[1] == [0,15,15]).all()
  assert (I.getNodeFromPath(ngon, ':CGNS#Distribution/ElementConnectivity')[1] == [0,30,30]).all()

@mark_mpi_test(2)
def test_remove_ngons_2p(sub_comm):

  #Generated from G.cartNGon((0,0,0), (1,1,0), (3,4,1))

  if sub_comm.Get_rank() == 0:
    ec = [1,4,2,5,3,6,4,7,5,8]
    pe = np.array([[1,0], [1,2], [2,0], [3,0], [3,4]])
    eso = np.arange(0,2*5+1,2)
    distri_e  = [0, 5, 17]
    distri_ec = [0, 10, 34]
    to_remove = [2-1]

    expected_distri_e = [0, 4, 15]
    expected_ec = [1,4,    3,6,4,7,5,8]
    expected_pe = np.array([[1,0],        [2,0], [3,0], [3,4]])
    expected_eso = np.arange(0, 2*4+1, 2)
  elif sub_comm.Get_rank() == 1:
    ec = [6,9,7,10,8,11,9,12,1,2,2,3,4,5,5,6,7,8,8,9,10,11,11,12]
    pe = np.array([[4,0], [5,0], [5,6], [6,0], [1,0], [2,0], [1,3], [2,4], [3,5], [4,6], [5,0], [6,0]])
    eso = np.arange(10, 2*17+1,2)
    distri_e  = [5, 17, 17]
    distri_ec = [10, 34, 34]
    to_remove = [16-5-1]

    expected_distri_e = [4, 15, 15]
    expected_ec = [6,9,7,10,8,11,9,12,1,2,2,3,4,5,5,6,7,8,8,9,      11,12]
    expected_pe = np.array([[4,0], [5,0], [5,6], [6,0], [1,0], [2,0], [1,3], [2,4], [3,5], [4,6],        [6,0]])
    expected_eso = np.arange(8, 2*15+1, 2)

  ngon = I.newElements('NGon', 'NGON', ec, [1,17])
  I.newDataArray('ElementStartOffset', eso, parent=ngon)
  I.newDataArray('ParentElements', pe, parent=ngon)
  I.newIndexArray('ElementConnectivity#Size', [34], parent=ngon)
  distri = I.createUniqueChild(ngon, ':CGNS#Distribution', 'UserDefinedData_t')
  I.newDataArray('Element', distri_e, parent=distri)
  I.newDataArray('ElementConnectivity', distri_ec, parent=distri)

  RME.remove_ngons(ngon, to_remove, sub_comm)

  assert (I.getNodeFromName(ngon, 'ElementRange')[1] == [1, 17-2]).all()
  assert (I.getNodeFromName(ngon, 'ElementConnectivity')[1] == expected_ec).all()
  assert (I.getNodeFromName(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (I.getNodeFromName(ngon, 'ElementStartOffset')[1] == expected_eso).all()
  assert  I.getNodeFromName(ngon, 'ElementConnectivity#Size')[1] == 34 - 2*2
  assert (I.getNodeFromPath(ngon, ':CGNS#Distribution/Element')[1] == expected_distri_e).all()
