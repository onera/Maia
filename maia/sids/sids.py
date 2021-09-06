from typing import List, Tuple
import numpy as np

import Converter.Internal as I

import maia.sids.Internal_ext  as IE
import maia.sids.cgns_keywords as CGK

import maia
from maia.sids.cgns_keywords import Label as CGL
from maia.utils.py_utils     import list_or_only_elt
from . import elements_utils as EU

# --------------------------------------------------------------------------
@maia.for_all_methods(IE.check_is_label("Zone_t"))
class Zone:
  @staticmethod
  def VertexSize(zone_node):
    z_sizes = I.getValue(zone_node)
    return list_or_only_elt(z_sizes[:,0])

  @staticmethod
  def CellSize(zone_node):
    z_sizes = I.getValue(zone_node)
    return list_or_only_elt(z_sizes[:,1])

  @staticmethod
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
  def VertexBoundarySize(zone_node):
    z_sizes = I.getValue(zone_node)
    return list_or_only_elt(z_sizes[:,2])

  @staticmethod
  def Type(zone_node):
    zone_type_node = I.getNodeFromType1(zone_node, CGL.ZoneType_t.name)
    return I.getValue(zone_type_node)

  #Todo : this one should go in IE
  @staticmethod
  def getBCsFromFamily(zone_node, families):
    bc_query = lambda n : I.getType(n) == 'BC_t' and I.getValue(n) == 'FamilySpecified' and \
      I.getValue(I.getNodeFromType1(n, CGL.FamilyName_t.name)) in families
    return IE.iterNodesByMatching(zone_node, ['ZoneBC_t', bc_query])

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
@maia.for_all_methods(IE.check_is_label("IndexRange_t"))
class PointRange:
  @staticmethod
  def VertexSize(point_range_node):
    """
    Allow point_range to be inverted (PR[:,1] < PR[:,0]) as it can occurs in struct GCs
    """
    pr_values = point_range_node[1]
    return np.abs(pr_values[:,1] - pr_values[:,0]) + 1

  @staticmethod
  def FaceSize(point_range_node):
    return np.subtract(PointRange.VertexSize(point_range_node), 1)

  @staticmethod
  def n_vtx(point_range_node):
    return np.prod(PointRange.VertexSize(point_range_node))

  @staticmethod
  def n_face(point_range_node):
    return np.prod([f for f in PointRange.FaceSize(point_range_node) if f > 0])


# --------------------------------------------------------------------------
@maia.for_all_methods(IE.check_is_label("IndexArray_t"))
class PointList:
  @staticmethod
  def FaceSize(point_list_node):
    pl_values = point_list_node[1]
    return pl_values.shape

  @staticmethod
  def n_face(point_range_node):
    return PointList.FaceSize(point_range_node)[1]


# --------------------------------------------------------------------------
def GridLocation(node):
  grid_loc_n = I.getNodeFromType1(node, 'GridLocation_t')
  return I.getValue(grid_loc_n) if grid_loc_n else 'Vertex'

# --------------------------------------------------------------------------
#todo Should go in IE
def newDataArrayFromName(parent, data_name):
  data_node = I.getNodeFromNameAndType(parent, data_name, CGL.DataArray_t.name)
  if not data_node:
    data_node = I.newDataArray(data_name, parent=parent)
  return data_node

# --------------------------------------------------------------------------
#todo : IE ?
def coordinates(node, name=None):

  grid_coord_node = I.getNodeFromType(node, "GridCoordinates_t") if name is None \
      else I.getNodeFromNameAndType(node, name, "GridCoordinates_t")
  if grid_coord_node is None:
    raise RuntimeError(f"Unable to find GridCoordinates_t node in {I.getName(node)}.")

  x = I.getVal(I.getNodeFromName1(grid_coord_node, "CoordinateX"))
  y = I.getVal(I.getNodeFromName1(grid_coord_node, "CoordinateY"))
  z = I.getVal(I.getNodeFromName1(grid_coord_node, "CoordinateZ"))

  return x, y, z

#todo : IE ?
def ngon_connectivity(node):
  count = 0
  for element_node in I.getNodesFromType(node, CGL.Elements_t.name):
    if ElementCGNSName(element_node) == "NGON_n":
      if count > 0:
        raise RuntimeError(f"Several NGON_n Elements_t node is not allowed in {I.getName(node)}.")
      face_vtx     = I.getVal(I.getNodeFromName1(element_node, "ElementConnectivity"))
      face_vtx_idx = I.getVal(I.getNodeFromName1(element_node, "ElementStartOffset"))
      ngon_pe      = I.getVal(I.getNodeFromName1(element_node, "ParentElements"))
      count += 1

  if count == 0:
    raise RuntimeError(f"Unable to find NGon_n Elements_t node in {I.getName(node)}.")
  return face_vtx, face_vtx_idx, ngon_pe

