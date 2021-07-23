from typing import List, Tuple
from functools import wraps
import numpy as np

import Converter.Internal as I

import maia.sids.Internal_ext  as IE
import maia.sids.cgns_keywords as CGK

from maia.sids.cgns_keywords import Label as CGL
from maia.utils.py_utils     import list_or_only_elt
from . import elements_utils as EU

# --------------------------------------------------------------------------
class Zone:
  @staticmethod
  @IE.check_is_label("Zone_t")
  def VertexSize(zone_node):
    z_sizes = I.getValue(zone_node)
    return list_or_only_elt(z_sizes[:,0])

  @staticmethod
  @IE.check_is_label("Zone_t")
  def CellSize(zone_node):
    z_sizes = I.getValue(zone_node)
    return list_or_only_elt(z_sizes[:,1])

  @staticmethod
  @IE.check_is_label("Zone_t")
  def FaceSize(zone_node):

    def compute_nface_per_direction(d, dim, vtx_size, cell_size):
      n_face_per_direction = vtx_size[d%dim]
      if dim >= 2:
        n_face_per_direction *= cell_size[(d+1)%dim]
      if dim == 3:
        n_face_per_direction *= cell_size[(d+2)%dim]
      return n_face_per_direction

    # Find face number
    vtx_size  = Zone.VertexSize(zone_node)
    cell_size = Zone.CellSize(zone_node)
    if Zone.Type(zone_node) == "Structured":
      dim = len(vtx_size)
      print(f"dim [S] = {dim}")
      # n_face = np.sum([vtx_size[(0+d)%dim]*cell_size[(1+d)%dim]*cell_size[(2+d)%dim] for d in range(dim)])
      n_face = [compute_nface_per_direction(d, dim, vtx_size, cell_size) for d in range(dim)]
      print(f"n_face [S] = {n_face}")
    elif Zone.Type(zone_node) == "Unstructured":
      element_node = I.getNodeFromType1(zone_node, CGL.Elements_t.name)
      if ElementType(element_node) == CGK.ElementType.NGON_n.value:
        face_vtx, face_vtx_idx, ngon_pe = face_connectivity(element_node)
        n_face = [ngon_pe.shape[0]]
      else:
        raise NotImplementedError(f"Unstructured Zone {I.getName(zone_node)} with {ElementCGNSName(element_node)} not yet implemented.")
      print(f"n_face [U] = {n_face}")
    else:
      raise TypeError(f"Unable to determine the ZoneType for Zone {I.getName(zone_node)}")
    return list_or_only_elt(n_face)

  @staticmethod
  @IE.check_is_label("Zone_t")
  def VertexBoundarySize(zone_node):
    z_sizes = I.getValue(zone_node)
    return list_or_only_elt(z_sizes[:,2])

  @staticmethod
  @IE.check_is_label("Zone_t")
  def Type(zone_node):
    zone_type_node = IE.getChildFromLabel1(zone_node, CGL.ZoneType_t.name)
    return I.getValue(zone_type_node)

  @staticmethod
  @IE.check_is_label("Zone_t")
  def getBCsFromFamily(zone_node, families):
    for bc_node in IE.getNodesByMatching(zone_node, ['ZoneBC_t', 'BC_t']):
      bctype = I.getValue(bc_node)
      if bctype == 'FamilySpecified':
        family_name_node = IE.getChildFromLabel1(bc_node, CGL.FamilyName_t.name)
        if I.getValue(family_name_node) in families:
          yield bc_node

  @staticmethod
  def n_vtx(zone_node):
    return np.prod(Zone.VertexSize(zone_node))

  @staticmethod
  def n_cell(zone_node):
    return np.prod(Zone.CellSize(zone_node))

  @staticmethod
  def n_face(zone_node):
    return np.sum(Zone.FaceSize(zone_node))

  @staticmethod
  def n_vtx_bnd(zone_node):
    return np.prod(Zone.VertexBoundarySize(zone_node))

  @staticmethod
  @IE.check_is_label("Zone_t")
  def get_ln_to_gn(zone_node: List) -> Tuple:
    """
    Args:
        zone_node (List): CGNS Zone_t node

    Returns:
        Tuple: Return local to global numerotation of vtx, cell and face
    """
    pdm_nodes = IE.getChildFromName1(zone_node, ":CGNS#Ppart")
    vtx_ln_to_gn  = I.getVal(IE.getGlobalNumbering(zone_node, 'Vertex'))
    cell_ln_to_gn = I.getVal(IE.getGlobalNumbering(zone_node, 'Cell'))
    face_ln_to_gn = I.getVal(IE.getChildFromName1(pdm_nodes, "np_face_ln_to_gn"))
    return vtx_ln_to_gn, cell_ln_to_gn, face_ln_to_gn

  @staticmethod
  @IE.check_is_label("Zone_t")
  def get_infos(zone_node: List) -> Tuple:
    """
    Args:
        zone_node (List): CGNS Zone_t

    Returns:
        Tuple: Return local to global numerotation of vtx, cell and face
    """
    pdm_nodes = IE.getChildFromName1(zone_node, ":CGNS#Ppart")
    # Vertex coordinates
    vtx_coords    = I.getVal(IE.getChildFromName1(pdm_nodes, "np_vtx_coord"))
    vtx_ln_to_gn  = I.getVal(IE.getGlobalNumbering(zone_node, 'Vertex'))
    # vtx_ln_to_gn  = I.getVal(I.getNodeFromName1(pdm_nodes, "np_vtx_ln_to_gn"))
    # Cell<->Face connectivity
    cell_face_idx = I.getVal(IE.getChildFromName1(pdm_nodes, "np_cell_face_idx"))
    cell_face     = I.getVal(IE.getChildFromName1(pdm_nodes, "np_cell_face"))
    # cell_ln_to_gn = I.getVal(I.requireNodeFromName1(pdm_nodes, "np_cell_ln_to_gn"))
    cell_ln_to_gn = I.getVal(IE.getGlobalNumbering(zone_node, 'Cell'))
    # Face<->Vtx connectivity
    face_vtx_idx  = I.getVal(IE.getChildFromName1(pdm_nodes, "np_face_vtx_idx"))
    face_vtx      = I.getVal(IE.getChildFromName1(pdm_nodes, "np_face_vtx"))
    face_ln_to_gn = I.getVal(IE.getChildFromName1(pdm_nodes, "np_face_ln_to_gn"))
    return vtx_coords, vtx_ln_to_gn, \
           cell_face_idx, cell_face, cell_ln_to_gn, \
           face_vtx_idx, face_vtx, face_ln_to_gn

# --------------------------------------------------------------------------
@IE.check_is_label("Elements_t")
def ElementRange(elements):
  return I.getNodeFromName(elements,"ElementRange")[1]

@IE.check_is_label("Elements_t")
def ElementType(elements):
  return elements[1][0]

@IE.check_is_label("Elements_t")
def ElementSize(elements):
  er = I.getNodeFromName(elements,"ElementRange")[1]
  return er[1] - er[0] + 1

def ElementCGNSName(element):
  return EU.element_name(ElementType(element))

def ElementDimension(element):
  return EU.element_dim(ElementType(element))

def ElementNVtx(element):
  return EU.element_number_of_nodes(ElementType(element))


# --------------------------------------------------------------------------
# @IE.check_is_label("IndexRange_t")
# def point_range_sizes(point_range_node):
#   """Allow point_range to be inverted (PR[:,1] < PR[:,0])
#   as it can occurs in struct GCs
#   """
#   pr_values = point_range_node[1]
#   return np.abs(pr_values[:,1] - pr_values[:,0]) + 1

class PointRange:
  @staticmethod
  @IE.check_is_label("IndexRange_t")
  def VertexSize(point_range_node):
    """Allow point_range to be inverted (PR[:,1] < PR[:,0])
    as it can occurs in struct GCs
    """
    pr_values = point_range_node[1]
    return np.abs(pr_values[:,1] - pr_values[:,0]) + 1

  @staticmethod
  @IE.check_is_label("IndexRange_t")
  def FaceSize(point_range_node):
    return np.subtract(PointRange.VertexSize(point_range_node), 1)

  @staticmethod
  def n_vtx(point_range_node):
    return np.prod(PointRange.VertexSize(point_range_node))

  @staticmethod
  def n_face(point_range_node):
    return np.prod(list(filter(lambda i:i > 0, PointRange.FaceSize(point_range_node))))


# --------------------------------------------------------------------------
# @IE.check_is_label("IndexArray_t")
# def point_list_sizes(point_list_node):
#   pl_values = point_list_node[1]
#   return pl_values.shape

class PointList:
  @staticmethod
  @IE.check_is_label("IndexArray_t")
  def FaceSize(point_list_node):
    pl_values = point_list_node[1]
    return pl_values.shape

  @staticmethod
  def n_face(point_range_node):
    return PointList.FaceSize(point_range_node)[1]


# --------------------------------------------------------------------------
# def zone_n_vtx( zone ):
#   return np.prod(Zone.VertexSize(zone))

# def zone_n_cell( zone ):
#   return np.prod(Zone.CellSize(zone))

# def zone_n_vtx_bnd( zone ):
#   return np.prod(Zone.VertexBoundarySize(zone))

# def point_range_n_vtx(point_range_node):
#   return np.prod(PointRange.VertexSize(point_range_node))

# def point_range_n_face(point_range_node):
#   pr_n_vtx_m1 = np.subtract(PointRange.VertexSize(point_range_node), 1)
#   return np.prod(list(filter(lambda i:i > 0, pr_n_vtx_m1)))

# def point_list_n_face(point_range_node):
#   return point_list_sizes(point_range_node)[1]

# --------------------------------------------------------------------------
def GridLocation(node):
  grid_loc_n = I.getNodeFromType1(node, 'GridLocation_t')
  return I.getValue(grid_loc_n) if grid_loc_n else 'Vertex'

# --------------------------------------------------------------------------
def newDataArrayFromName(parent, data_name):
  data_node = I.getNodeFromNameAndType(parent, data_name, CGL.DataArray_t.name)
  if not data_node:
    data_node = I.newDataArray(data_name, parent=parent)
  return data_node

# --------------------------------------------------------------------------
def coordinates(node, name=None):
  def get_children(grid_coord_node, name):
    coord_node = I.getNodeFromName1(grid_coord_node, name)
    if coord_node is None:
      raise RuntimeError(f"Unable to find '{name}' node in {I.getName(grid_coord_node)}.")
    return coord_node

  if name:
    grid_coord_node = I.getNodeFromNameAndType(node, name, "GridCoordinates_t")
  else:
    grid_coord_node = I.getNodeFromType(node, "GridCoordinates_t")

  if grid_coord_node is None:
    raise RuntimeError(f"Unable to find GridCoordinates_t node in {I.getName(node)}.")
  x = I.getVal(get_children(grid_coord_node, "CoordinateX"))
  y = I.getVal(get_children(grid_coord_node, "CoordinateY"))
  z = I.getVal(get_children(grid_coord_node, "CoordinateZ"))

  return x, y, z

def face_connectivity(node):
  def get_children(element_node, name):
    elt_node = I.getNodeFromName1(element_node, name)
    if elt_node is None:
      raise RuntimeError(f"Unable to find '{name}' node in {I.getName(element_node)}.")
    return elt_node

  count = 0
  for element_node in I.getNodesFromType(node, CGL.Elements_t.name):
    if ElementType(element_node) == CGK.ElementType.NGON_n.value:
      if count > 0:
        raise RuntimeError(f"Several NGON_n Elements_t node is not allowed in {I.getName(node)}.")
      face_vtx     = I.getVal(get_children(element_node, "ElementConnectivity"))
      face_vtx_idx = I.getVal(get_children(element_node, "ElementStartOffset"))
      ngon_pe      = I.getVal(get_children(element_node, "ParentElements"))
      count += 1

  if count == 0:
    raise RuntimeError(f"Unable to find NGon_n Elements_t node in {I.getName(node)}.")
  return face_vtx, face_vtx_idx, ngon_pe

def cell_connectivity(node):
  def get_children(element_node, name):
    elt_node = I.getNodeFromName1(element_node, name)
    if elt_node is None:
      raise RuntimeError(f"Unable to find '{name}' node in {I.getName(element_node)}.")
    return elt_node

  count = 0
  for element_node in I.getNodesFromType(node, CGL.Elements_t.name):
    if ElementType(element_node) == CGK.ElementType.NGON_n.value:
      if count > 0:
        raise RuntimeError(f"Several NGON_n Elements_t node is not allowed in {I.getName(node)}.")
      face_vtx     = I.getVal(get_children(element_node, "ElementConnectivity"))
      face_vtx_idx = I.getVal(get_children(element_node, "ElementStartOffset"))
      ngon_pe      = I.getVal(get_children(element_node, "ParentElements"))
      count += 1

  if count == 0:
    raise RuntimeError(f"Unable to find NGon_n Elements_t node in {I.getName(node)}.")
  return face_vtx, face_vtx_idx, ngon_pe

if __name__ == "__main__":
    import Converter.PyTree as C

    import maia.sids.cgns_keywords as CGK
    import maia.sids.sids          as SIDS
    from maia.sids.cgns_keywords import Label as CGL
    t = C.convertFile2PyTree("geometry/cubeU_join_bnd-new.hdf")
    I.printTree(t)

    for zone_node in I.getZones(t):
        zonetype_node = I.getNodeFromType1(zone_node, CGL.ZoneType_t.name)
        I._rmNode(t, zonetype_node)
    for zone_node in I.getZones(t):
        if SIDS.Zone.Type(zone_node) == "Unstructured":
            pass

    for zone_node in I.getZones(t):
        element_node = I.getNodeFromType1(zone_node, CGL.Elements_t.name)
        # NGon elements
        if SIDS.ElementType(element_node) == CGK.ElementType.NFACE_n.value:
            pass
        else:
            raise NotImplementedForElementError(zone_node, element_node)
