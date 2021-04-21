import pytest

import Converter.Internal     as I
import maia.sids.Internal_ext as IE

from maia.utils import zone_elements_utils as EZU

def test_get_npe_with_element_type_cgns():
  assert EZU.get_npe_with_element_type_cgns(3) == 2
  assert EZU.get_npe_with_element_type_cgns(14) == 6
  assert EZU.get_npe_with_element_type_cgns(10) == 4
  with pytest.raises(NotImplementedError):
    EZU.get_npe_with_element_type_cgns(1)
    EZU.get_npe_with_element_type_cgns(30)

def test_get_paradigm_type_with_element_type_cgns():
  assert EZU.get_paradigm_type_with_element_type_cgns(3) == 1
  assert EZU.get_paradigm_type_with_element_type_cgns(14) == 7
  assert EZU.get_paradigm_type_with_element_type_cgns(10) == 5
  with pytest.raises(NotImplementedError):
    EZU.get_paradigm_type_with_element_type_cgns(1)
    EZU.get_paradigm_type_with_element_type_cgns(30)

def test_get_ordered_elements_std():
  zone = I.newZone()
  I.newElements('ElemA', erange=[11, 53], parent=zone)
  I.newElements('ElemB', erange=[1, 10], parent=zone)
  I.newElements('ElemC', erange=[54,60], parent=zone)

  sorted_elems = EZU.get_ordered_elements_std(zone)
  assert [I.getName(elem) for elem in sorted_elems] == ['ElemB', 'ElemA', 'ElemC']

def test_get_range_of_ngon():
  zone = I.newZone()
  I.newElements('ElemA', 'NGON',  erange=[11, 53], parent=zone)
  I.newElements('ElemB', 'NFACE', erange=[1, 10], parent=zone)
  I.newElements('ElemC', 'HEXA',  erange=[54,60], parent=zone)
  assert (EZU.get_range_of_ngon(zone) == [11,53]).all()

def test_get_next_elements_range():
  zone = I.newZone()
  I.newElements('ElemA', 'NGON',  erange=[11, 53], parent=zone)
  I.newElements('ElemB', 'NFACE', erange=[1, 10], parent=zone)
  I.newElements('ElemC', 'HEXA',  erange=[54,60], parent=zone)
  assert EZU.get_next_elements_range(zone) == 60

def test_collect_connectity():
  zone = I.newZone()
  I.newElements('ElemA', 'NGON',  econnectivity=[5,3,9,2], parent=zone)
  I.newElements('ElemB', 'NFACE', econnectivity=[11,8,15,10,6,4], parent=zone)
  I.newElements('ElemC', 'HEXA',  econnectivity=[3,7], parent=zone)
  connectivities = EZU.collect_connectity(I.getNodesFromType1(zone, 'Elements_t'))
  assert len(connectivities) == 3
  assert (connectivities[0] == I.getNodeFromPath(zone, 'ElemA/ElementConnectivity')[1]).all()
  assert (connectivities[2] == I.getNodeFromPath(zone, 'ElemC/ElementConnectivity')[1]).all()
  
def test_collect_pdm_type_and_nelemts():
  zone = I.newZone()
  node = I.newElements('ElemA', 'PENTA', parent=zone)
  IE.newDistribution({'Element' : [0,10,10]}, node)
  node = I.newElements('ElemB', 'TRI'  , parent=zone)
  IE.newDistribution({'Element' : [0,5,34]}, node)
  node = I.newElements('ElemC', 'HEXA' , parent=zone)
  IE.newDistribution({'Element' : [10,20,20]}, node)
  types, n_elts = EZU.collect_pdm_type_and_nelemts(I.getNodesFromType1(zone, 'Elements_t'))
  assert (types  == [7,2,8]).all()
  assert (n_elts == [10,5,10]).all()

