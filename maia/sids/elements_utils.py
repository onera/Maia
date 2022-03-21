import Converter.Internal as I
from . import sids
from maia.utils import py_utils
import Pypdm.Pypdm as PDM
import itertools

elements_properties = [
#CGNS_Id, ElementName        ,  dim, nVtx,nEdge,nFace, refElt,pdm_id
    ("ElementTypeNull"       , None, None, None, None,   None,  None),
    ("ElementTypeUserDefined", None, None, None, None,   None,  None),
    ("NODE"                  ,    0,    1,    1,    0,  "NODE", PDM._PDM_MESH_NODAL_POINT),
    ("BAR_2"                 ,    1,    2,    1,    0,   "BAR", PDM._PDM_MESH_NODAL_BAR2),
    ("BAR_3"                 ,    1,    3,    1,    0,   "BAR", None),
    ("TRI_3"                 ,    2,    3,    3,    1,   "TRI", PDM._PDM_MESH_NODAL_TRIA3),
    ("TRI_6"                 ,    2,    6,    3,    1,   "TRI", None),
    ("QUAD_4"                ,    2,    4,    4,    1,  "QUAD", PDM._PDM_MESH_NODAL_QUAD4),
    ("QUAD_8"                ,    2,    8,    4,    1,  "QUAD",    9),
    ("QUAD_9"                ,    2,    9,    4,    1,  "QUAD", None),
    ("TETRA_4"               ,    3,    4,    6,    4, "TETRA", PDM._PDM_MESH_NODAL_TETRA4),
    ("TETRA_10"              ,    3,   10,    6,    4, "TETRA", None),
    ("PYRA_5"                ,    3,    5,    8,    5,  "PYRA", PDM._PDM_MESH_NODAL_PYRAMID5),
    ("PYRA_14"               ,    3,   14,    8,    5,  "PYRA", None),
    ("PENTA_6"               ,    3,    6,    9,    5, "PENTA", PDM._PDM_MESH_NODAL_PRISM6),
    ("PENTA_15"              ,    3,   15,    9,    5, "PENTA", None),
    ("PENTA_18"              ,    3,   18,    9,    5, "PENTA", None),
    ("HEXA_8"                ,    3,    8,   12,    6,  "HEXA", PDM._PDM_MESH_NODAL_HEXA8),
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

elements_dim_to_pdm_kind = [PDM._PDM_GEOMETRY_KIND_CORNER,
                            PDM._PDM_GEOMETRY_KIND_RIDGE,
                            PDM._PDM_GEOMETRY_KIND_SURFACIC,
                            PDM._PDM_GEOMETRY_KIND_VOLUMIC]


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

def cgns_elt_name_to_pdm_element_type(name):
  for elem in elements_properties:
    if elem[0] == name:
      return elem[6]
  raise NameError(f"CGNS Elem {name} is not a valid name")

def pdm_elt_name_to_cgns_element_type(pdm_id):
  for elem in elements_properties:
    if elem[6] == pdm_id:
      return elem[0]
  raise NameError(f"No PDM element associated to {pdm_id}")

def get_range_of_ngon(zone):
  """
  Return the ElementRange array of the NGON elements
  """
  return sids.ElementRange(sids.Zone.NGonNode(zone))

def get_ordered_elements(zone):
  """
  Return the elements nodes in increasing order wrt ElementRange
  """
  return sorted(I.getNodesFromType1(zone, 'Elements_t'),
                key = lambda item : sids.ElementRange(item)[0])

def get_ordered_elements_per_dim(zone):
  """
  Return a list of size 4 containing Element nodes belonging to each dimension.
  In addition, Element are sorted according to their ElementRange withing each dimension.
  """
  # TODO : how to prevent special case of range of elemt mixed in dim ?
  return py_utils.bucket_split(get_ordered_elements(zone), lambda e: sids.ElementDimension(e), size=4)

  return sorted_elts_by_dim

def get_elt_range_per_dim(zone):
  """
  Return a list of size 4 containing min & max element id of each dimension
  This function is relevant only if Element of same dimension have consecutive
  ElementRange
  """
  sorted_elts_by_dim = get_ordered_elements_per_dim(zone)

  range_by_dim = [[0,0], [0,0], [0,0], [0,0]]
  for i_dim, elt_dim in enumerate(sorted_elts_by_dim):
    # Element is sorted
    if(len(elt_dim) > 0):
      range_by_dim[i_dim][0] = sids.ElementRange(elt_dim[0 ])[0]
      range_by_dim[i_dim][1] = sids.ElementRange(elt_dim[-1])[1]

  # Check if element range were not interlaced
  for first, second in itertools.combinations(range_by_dim, 2):
    if py_utils.are_overlapping(first, second, strict=True):
      raise RuntimeError("ElementRange with different dimensions are interlaced")

  return range_by_dim

