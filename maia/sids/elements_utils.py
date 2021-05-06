import Converter.Internal as I
from . import sids
import Pypdm.Pypdm as PDM

elements_properties = [
#CGNS_Id, ElementName        ,  dim, nVtx,nEdge,nFace, refElt,pdm_id
    ("ElementTypeNull"       , None, None, None, None,   None,  None),
    ("ElementTypeUserDefined", None, None, None, None,   None,  None),
    ("NODE"                  ,    0,    1,    1,    0,  "NODE",    0),
    ("BAR_2"                 ,    1,    2,    1,    0,   "BAR",    1),
    ("BAR_3"                 ,    1,    3,    1,    0,   "BAR", None),
    ("TRI_3"                 ,    2,    3,    3,    1,   "TRI",    2),
    ("TRI_6"                 ,    2,    6,    3,    1,   "TRI", None),
    ("QUAD_4"                ,    2,    4,    4,    1,  "QUAD",    3),
    ("QUAD_8"                ,    2,    8,    4,    1,  "QUAD",    9),
    ("QUAD_9"                ,    2,    9,    4,    1,  "QUAD", None),
    ("TETRA_4"               ,    3,    4,    6,    4, "TETRA",    5),
    ("TETRA_10"              ,    3,   10,    6,    4, "TETRA", None),
    ("PYRA_5"                ,    3,    5,    8,    5,  "PYRA",    6),
    ("PYRA_14"               ,    3,   14,    8,    5,  "PYRA", None),
    ("PENTA_6"               ,    3,    6,    9,    5, "PENTA",    7),
    ("PENTA_15"              ,    3,   15,    9,    5, "PENTA", None),
    ("PENTA_18"              ,    3,   18,    9,    5, "PENTA", None),
    ("HEXA_8"                ,    3,    8,   12,    6,  "HEXA",    8),
    ("HEXA_20"               ,    3,   20,   12,    6,  "HEXA",   10),
    ("HEXA_27"               ,    3,   27,   12,    6,  "HEXA", None),
    ("MIXED"                 , None, None, None, None,    None, None),
    ("PYRA_13"               ,    3,   13,    8,    5,  "PYRA", None),
    ("NGON_n"                ,    2, None, None, None,    None, None),
    ("NFACE_n"               ,    3, None, None, None,    None, None),
    ("BAR_4"                 ,    1,    4,    1,    0,   "BAR", None),
    ("TRI_9"                 ,    2,    9,    3,    1,   "TRI", None),
    ("TRI_10"                ,    2,   10,    3,    1,   "TRI", None),
    ("QUAD_12"               ,    2,   12,    4,    1,  "QUAD", None),
    ("QUAD_16"               ,    2,   16,    4,    1,  "QUAD", None),
    ("TETRA_16"              ,    3,   16,    6,    4, "TETRA", None),
    ("TETRA_20"              ,    3,   20,    6,    4, "TETRA", None),
    ("PYRA_21"               ,    3,   21,    8,    5,  "PYRA", None),
    ("PYRA_29"               ,    3,   29,    8,    5,  "PYRA", None),
    ("PYRA_30"               ,    3,   30,    8,    5,  "PYRA", None),
    ("PENTA_24"              ,    3,   24,    9,    5, "PENTA", None),
    ("PENTA_38"              ,    3,   38,    9,    5, "PENTA", None),
    ("PENTA_40"              ,    3,   40,    9,    5, "PENTA", None),
    ("HEXA_32"               ,    3,   32,   12,    6,  "HEXA", None),
    ("HEXA_56"               ,    3,   56,   12,    6,  "HEXA", None),
    ("HEXA_64"               ,    3,   64,   12,    6,  "HEXA", None),
    ("BAR_5"                 ,    1,    5,    1,    0,   "BAR", None),
    ("TRI_12"                ,    2,   12,    3,    1,   "TRI", None),
    ("TRI_15"                ,    2,   15,    3,    1,   "TRI", None),
    ("QUAD_P4_16"            ,    2,   16,    4,    1,  "QUAD", None),
    ("QUAD_25"               ,    2,   25,    4,    1,  "QUAD", None),
    ("TETRA_22"              ,    3,   22,    6,    4, "TETRA", None),
    ("TETRA_34"              ,    3,   34,    6,    4, "TETRA", None),
    ("TETRA_35"              ,    3,   35,    6,    4, "TETRA", None),
    ("PYRA_P4_29"            ,    3,   29,    8,    5,  "PYRA", None),
    ("PYRA_50"               ,    3,   50,    8,    5,  "PYRA", None),
    ("PYRA_55"               ,    3,   55,    8,    5,  "PYRA", None),
    ("PENTA_33"              ,    3,   33,    9,    5, "PENTA", None),
    ("PENTA_66"              ,    3,   66,    9,    5, "PENTA", None),
    ("PENTA_75"              ,    3,   75,    9,    5, "PENTA", None),
    ("HEXA_44"               ,    3,   44,   12,    6,  "HEXA", None),
    ("HEXA_98"               ,    3,   98,   12,    6,  "HEXA", None),
    ("HEXA_125"              ,    3,  125,   12,    6,  "HEXA", None),
    ]

def element_name(n):
  assert n < len(elements_properties)
  return elements_properties[n][0]

def element_dim(n):
  assert n < len(elements_properties)
  return elements_properties[n][1]

def element_number_of_nodes(n):
  assert n < len(elements_properties)
  return elements_properties[n][2]

def element_pdm_type(n):
  assert n < len(elements_properties)
  return elements_properties[n][6]
  
PDM_MESH_NODAL_POINT    = PDM._PDM_MESH_NODAL_POINT
PDM_MESH_NODAL_BAR2     = PDM._PDM_MESH_NODAL_BAR2
PDM_MESH_NODAL_TRIA3    = PDM._PDM_MESH_NODAL_TRIA3
PDM_MESH_NODAL_QUAD4    = PDM._PDM_MESH_NODAL_QUAD4
PDM_MESH_NODAL_POLY_2D  = PDM._PDM_MESH_NODAL_POLY_2D
PDM_MESH_NODAL_TETRA4   = PDM._PDM_MESH_NODAL_TETRA4
PDM_MESH_NODAL_PYRAMID5 = PDM._PDM_MESH_NODAL_PYRAMID5
PDM_MESH_NODAL_PRISM6   = PDM._PDM_MESH_NODAL_PRISM6
PDM_MESH_NODAL_HEXA8    = PDM._PDM_MESH_NODAL_HEXA8
PDM_MESH_NODAL_POLY_3D  = PDM._PDM_MESH_NODAL_POLY_3D
cgns_pdm_element_type = [
  ("NODE"    ,  PDM._PDM_MESH_NODAL_POINT    ),
  ("BAR_2"   ,  PDM._PDM_MESH_NODAL_BAR2     ),
  ("TRI_3"   ,  PDM._PDM_MESH_NODAL_TRIA3    ),
  ("QUAD_4"  ,  PDM._PDM_MESH_NODAL_QUAD4    ),
  ("TETRA_4" ,  PDM._PDM_MESH_NODAL_TETRA4   ),
  ("PYRA_5"  ,  PDM._PDM_MESH_NODAL_PYRAMID5 ),
  ("PENTA_6" ,  PDM._PDM_MESH_NODAL_PRISM6   ),
  ("HEXA_8"  ,  PDM._PDM_MESH_NODAL_HEXA8    ),
]

def get_range_of_ngon(zone):
  """
  Return the ElementRange array of the NGON elements
  """
  ngons = [elem for elem in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementType(elem) == 22]
  assert len(ngons) == 1
  return sids.ElementRange(ngons[0])
def cgns_elt_name_to_pdm_element_type(name):
  for cgns_name,pdm_name in cgns_pdm_element_type:
    if cgns_name==name: return pdm_name
  raise NameError("No PDM element associated to "+name)


pdm_cgns_element_type = [
  (PDM._PDM_MESH_NODAL_POINT   , "NODE"   ),
  (PDM._PDM_MESH_NODAL_BAR2    , "BAR_2"  ),
  (PDM._PDM_MESH_NODAL_TRIA3   , "TRI_3"  ),
  (PDM._PDM_MESH_NODAL_QUAD4   , "QUAD_4" ),
  (PDM._PDM_MESH_NODAL_TETRA4  , "TETRA_4"),
  (PDM._PDM_MESH_NODAL_PYRAMID5, "PYRA_5" ),
  (PDM._PDM_MESH_NODAL_PRISM6  , "PENTA_6"),
  (PDM._PDM_MESH_NODAL_HEXA8   , "HEXA_8" ),
]

def pdm_elt_name_to_cgns_element_type(name):
  for pdm_name,cgns_name in pdm_cgns_element_type:
    if pdm_name==name: return cgns_name
  raise NameError("No PDM element associated to "+name)
