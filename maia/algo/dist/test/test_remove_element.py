import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT
import maia.pytree.sids   as sids

from maia.algo.dist import remove_element as RME

def test_remove_element():
  zone = PT.new_Zone()

  quad_ec = [1,2,6,5,2,3,7,6,3,4,8,7,5,6,10,9,6,7,11,10,7,8,12,11]
  quad = PT.new_Elements('Quad', 'QUAD_4', econn=quad_ec, erange=[1,6], parent=zone)
  bar  = PT.new_Elements('Bar', 'BAR_2', econn=[10,9,11,10,12,11,5,1,9,5], erange=[7,11], parent=zone)

  ngon_ec = [2,1,3,2,1,5,4,3,2,6,3,7,6,5,8,4,6,7,5,9,8,7,10,6,7,11,9,10,12,8,10,11,11,12]
  ngon_pe = np.array([[29,0],[30,0],[29,0],[31,0],[29,30],[30,31],[29,32],[31,0],[33,30],\
                      [32,0],[31,34],[33,32],[33,34],[32,0],[34,0],[33,0],[34,0]])
  expected_pe = np.copy(ngon_pe)
  expected_pe[np.where(ngon_pe > 0)] -= 5
  ngon = PT.new_NGonElements('NGon', parent=zone,
      erange=[12,28], eso=np.arange(0, 35, 2), ec=ngon_ec, pe=ngon_pe)

  nface_ec = [7,5,1,3,6,2,5,9,11,4,6,8,7,10,12,14,13,12,9,16,11,13,15,17]
  nface = PT.new_NFaceElements('NFace', erange=[29,34], eso=np.arange(0,24+1,4), ec=nface_ec, parent=zone)

  zbc = PT.new_ZoneBC(zone)
  bc = PT.new_BC(point_list = [[25,27,28]], loc='EdgeCenter', parent=zbc)

  RME.remove_element(zone, bar)
  assert (sids.Element.Range(quad)  == [1,6]).all()
  assert (sids.Element.Range(ngon)  == [12-5,28-5]).all()
  assert (sids.Element.Range(nface) == [29-5,34-5]).all()
  assert (PT.get_child_from_name(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (PT.get_node_from_name(zone, 'PointList')[1] == [[20,22,23]]).all()
  assert PT.get_node_from_name(zone, 'Bar') is None

@pytest_parallel.mark.parallel(1)
def test_remove_ngons(comm):
  #Generated from G.cartNGon((0,0,0), (1,1,0), (3,4,1))
  # TODO handwritten ngon (4_cubes?)

  ec = [1,4, 2,5, 3,6, 4,7, 5,8, 6,9, 7,10, 8,11, 9,12, 1,2, 2,3, 4,5, 5,6, 7,8, 8,9, 10,11, 11,12]
  pe = np.array([[18,0], [18,19], [19,0],  [20,0],  [20,21], [21,0],  [22,0], [22,23], [23,0],
                 [18,0], [19,0],  [18,20], [19,21], [20,22], [21,23], [22,0], [23,0]])
  ngon = PT.new_NGonElements('NGon', erange=[1,17], eso=np.arange(0, 35, 2), ec=ec, pe=pe)
  distri = PT.new_child(ngon, ':CGNS#Distribution', 'UserDefinedData_t')
  PT.new_DataArray('Element', [0, 17, 17], parent=distri)
  PT.new_DataArray('ElementConnectivity', [0, 34, 34], parent=distri)

  RME.remove_ngons(ngon, [1,15], comm)

  expected_ec = [1,4,    3,6, 4,7, 5,8, 6,9, 7,10, 8,11, 9,12, 1,2, 2,3, 4,5, 5,6, 7,8, 8,9,      11,12]
  expected_pe = np.array([[16,0],         [17,0],  [18,0],  [18,19], [19,0], [20,0], [20,21], [21,0],
                          [16,0], [17,0], [16,18], [17,19], [18,20], [19,21],        [21,0]])
  assert (PT.get_node_from_name(ngon, 'ElementRange')[1] == [1, 15]).all()
  assert (PT.get_node_from_name(ngon, 'ElementConnectivity')[1] == expected_ec).all()
  assert (PT.get_node_from_name(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (PT.get_node_from_name(ngon, 'ElementStartOffset')[1] == np.arange(0,31,2)).all()
  assert (PT.get_node_from_path(ngon, ':CGNS#Distribution/Element')[1] == [0,15,15]).all()
  assert (PT.get_node_from_path(ngon, ':CGNS#Distribution/ElementConnectivity')[1] == [0,30,30]).all()

@pytest_parallel.mark.parallel(2)
def test_remove_ngons_2p(comm):

  #Generated from G.cartNGon((0,0,0), (1,1,0), (3,4,1))

  if comm.Get_rank() == 0:
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
  elif comm.Get_rank() == 1:
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

  ngon = PT.new_NGonElements('NGon', erange=[7,24], eso=eso, ec=ec, pe=pe)
  distri = PT.new_child(ngon, ':CGNS#Distribution', 'UserDefinedData_t')
  PT.new_DataArray('Element', distri_e, parent=distri)
  PT.new_DataArray('ElementConnectivity', distri_ec, parent=distri)

  RME.remove_ngons(ngon, to_remove, comm)

  assert (PT.get_node_from_name(ngon, 'ElementRange')[1] == [7, 24-2]).all()
  assert (PT.get_node_from_name(ngon, 'ElementConnectivity')[1] == expected_ec).all()
  assert (PT.get_node_from_name(ngon, 'ParentElements')[1] == expected_pe).all()
  assert (PT.get_node_from_name(ngon, 'ElementStartOffset')[1] == expected_eso).all()
  assert (PT.get_node_from_path(ngon, ':CGNS#Distribution/Element')[1] == expected_distri_e).all()
