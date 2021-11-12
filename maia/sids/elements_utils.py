import Converter.Internal as I
from mpi4py import MPI
import numpy as np
from . import sids
from maia       import npy_pdm_gnum_dtype as pdm_gnum_dtype
import Pypdm.Pypdm as PDM

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

elements_kind_by_dim = [PDM._PDM_GEOMETRY_KIND_CORNER,
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
  ngons = [elem for elem in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementType(elem) == 22]
  assert len(ngons) == 1
  return sids.ElementRange(ngons[0])

def get_ordered_elements_std(zone):
  """
  Return the elements nodes in inscreasing order wrt ElementRange
  """
  return sorted(I.getNodesFromType1(zone, 'Elements_t'),
                key = lambda item : sids.ElementRange(item)[0])

def get_ordered_elements_std_by_geom_kind(zone):
  """
  Return the elements nodes in inscreasing order wrt ElementRange
  """
  sorted_elts = get_ordered_elements_std(zone)
  next_dim = 0
  sorted_elts_by_dim = [[], [], [], []]
  for elt in sorted_elts:
    # TODO : how to prevent special case of range of elemt mixed in dim ?
    sorted_elts_by_dim[sids.ElementDimension(elt)].append(elt)

  return sorted_elts_by_dim

def get_range_elt_of_same_dim(zone):
  """
  """
  sorted_elts_by_dim = get_ordered_elements_std_by_geom_kind(zone)

  range_by_dim = [[0,0], [0,0], [0,0], [0,0]]

  for i_dim, elt_dim in enumerate(sorted_elts_by_dim):
    # Element is sorted
    if(len(elt_dim) > 0):
      range_by_dim[i_dim][0] = sids.ElementRange(elt_dim[0 ])[0]
      range_by_dim[i_dim][1] = sids.ElementRange(elt_dim[-1])[1]

  return range_by_dim

def split_point_list_by_dim(bc_point_lists, range_by_dim, comm):
  """
  """
  # print("range_by_dim --> ", range_by_dim)

  bc_point_lists_by_dim = [[], [], [], []]
  for pl in bc_point_lists:

    if(pl.shape[1] > 0):
      min_l_pl = np.amin(pl[0,:])
      max_l_pl = np.amax(pl[0,:])
    else:
      min_l_pl = np.iinfo(pdm_gnum_dtype).max
      max_l_pl = -1

    min_pl    = comm.allreduce(min_l_pl, op=MPI.MIN)
    max_pl    = comm.allreduce(max_l_pl, op=MPI.MAX)

    for i_dim in range(len(range_by_dim)):
      if(min_pl >= range_by_dim[i_dim][0] and max_pl <= range_by_dim[i_dim][1]):
        bc_point_lists_by_dim[i_dim].append(pl)

  return bc_point_lists_by_dim
