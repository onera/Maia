from maia.pytree.sids.elements_utils import elements_properties
import Pypdm.Pypdm as PDM

elements_dim_to_pdm_kind = [
        PDM._PDM_GEOMETRY_KIND_CORNER,
        PDM._PDM_GEOMETRY_KIND_RIDGE,
        PDM._PDM_GEOMETRY_KIND_SURFACIC,
        PDM._PDM_GEOMETRY_KIND_VOLUMIC
        ]
cgns_to_pdm = {
        "NODE"   :  PDM._PDM_MESH_NODAL_POINT,
        "BAR_2"  :  PDM._PDM_MESH_NODAL_BAR2,
        "TRI_3"  :  PDM._PDM_MESH_NODAL_TRIA3,
        "QUAD_4" :  PDM._PDM_MESH_NODAL_QUAD4,
        "QUAD_8" :     9,
        "TETRA_4":  PDM._PDM_MESH_NODAL_TETRA4,
        "PYRA_5" :  PDM._PDM_MESH_NODAL_PYRAMID5,
        "PENTA_6":  PDM._PDM_MESH_NODAL_PRISM6,
        "HEXA_8" :  PDM._PDM_MESH_NODAL_HEXA8,
        "HEXA_20":    10,
        }
pdm_to_cgns = {val : key for key, val in cgns_to_pdm.items()}

def element_pdm_type(n):
  assert n < len(elements_properties)
  return cgns_to_pdm[elements_properties[n][0]]

def cgns_elt_name_to_pdm_element_type(name):
    return cgns_to_pdm[name]

def pdm_elt_name_to_cgns_element_type(pdm_id):
    return pdm_to_cgns[pdm_id]


