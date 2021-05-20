import pytest
import Converter.Internal as I

from maia.sids import elements_utils as EU

import Pypdm.Pypdm as PDM

def test_element_name():
  assert EU.element_name(5)  == "TRI_3"
  assert EU.element_name(38) == "HEXA_56"
  with pytest.raises(AssertionError):
    EU.element_name(1000)

def test_element_dim():
  assert EU.element_dim(5)  == 2
  assert EU.element_dim(38) == 3
  with pytest.raises(AssertionError):
    EU.element_dim(1000)

def test_element_number_of_nodes():
  assert EU.element_number_of_nodes(5)  == 2
  assert EU.element_number_of_nodes(38) == None
  with pytest.raises(AssertionError):
    EU.element_number_of_nodes(1000)

def test_element_number_of_nodes():
  assert EU.element_number_of_nodes(5)  == 3
  assert EU.element_number_of_nodes(38) == 56
  with pytest.raises(AssertionError):
    EU.element_number_of_nodes(1000)

def test_cgns_elt_name_to_pdm_element_type():
  assert EU.cgns_elt_name_to_pdm_element_type("TRI_3") == PDM._PDM_MESH_NODAL_TRIA3
  assert EU.cgns_elt_name_to_pdm_element_type("NODE") == PDM._PDM_MESH_NODAL_POINT
  with pytest.raises(NameError):
    EU.cgns_elt_name_to_pdm_element_type("TOTO")

def test_pdm_elt_name_to_cgns_element_type():
  assert EU.pdm_elt_name_to_cgns_element_type(PDM._PDM_MESH_NODAL_TRIA3) == "TRI_3"
  assert EU.pdm_elt_name_to_cgns_element_type(PDM._PDM_MESH_NODAL_POINT) == "NODE"
  with pytest.raises(NameError):
    EU.pdm_elt_name_to_cgns_element_type("TOTO")

def test_get_range_of_ngon():
  zone = I.newZone()
  I.newElements('ElemA', 'NGON',  erange=[11, 53], parent=zone)
  I.newElements('ElemB', 'NFACE', erange=[1, 10], parent=zone)
  I.newElements('ElemC', 'HEXA',  erange=[54,60], parent=zone)
  assert (EU.get_range_of_ngon(zone) == [11,53]).all()

def test_get_ordered_elements_std():
  zone = I.newZone()
  I.newElements('ElemA', erange=[11, 53], parent=zone)
  I.newElements('ElemB', erange=[1, 10], parent=zone)
  I.newElements('ElemC', erange=[54,60], parent=zone)

  sorted_elems = EU.get_ordered_elements_std(zone)
  assert [I.getName(elem) for elem in sorted_elems] == ['ElemB', 'ElemA', 'ElemC']
    
