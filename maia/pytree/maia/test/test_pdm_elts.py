import Pypdm.Pypdm as PDM

from maia.pytree.maia import pdm_elts as PE

def test_cgns_elt_name_to_pdm_element_type():
  assert PE.cgns_elt_name_to_pdm_element_type("TRI_3") == PDM._PDM_MESH_NODAL_TRIA3
  assert PE.cgns_elt_name_to_pdm_element_type("NODE") == PDM._PDM_MESH_NODAL_POINT

def test_pdm_elt_name_to_cgns_element_type():
  assert PE.pdm_elt_name_to_cgns_element_type(PDM._PDM_MESH_NODAL_TRIA3) == "TRI_3"
  assert PE.pdm_elt_name_to_cgns_element_type(PDM._PDM_MESH_NODAL_POINT) == "NODE"


