import numpy as np
import itertools

import Converter.Internal as I

from maia.utils              import py_utils
from maia.utils.meta         import for_all_methods

from maia.pytree.compare import check_is_label, check_in_labels
from . import elements_utils as EU

# --------------------------------------------------------------------------
@for_all_methods(check_is_label("Zone_t"))
class Zone:
  @staticmethod
  def VertexSize(zone_node):
    z_sizes = I.getValue(zone_node)
    return py_utils.list_or_only_elt(z_sizes[:,0])

  @staticmethod
  def CellSize(zone_node):
    z_sizes = I.getValue(zone_node)
    return py_utils.list_or_only_elt(z_sizes[:,1])

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
      # n_face = np.sum([vtx_size[(0+d)%dim]*cell_size[(1+d)%dim]*cell_size[(2+d)%dim] for d in range(dim)])
      n_face = [compute_nface_per_direction(d, dim, vtx_size, cell_size) for d in range(dim)]
    elif Zone.Type(zone_node) == "Unstructured":
      ngon_node = Zone.NGonNode(zone_node)
      er = I.getNodeFromName1(ngon_node, 'ElementRange')[1]
      n_face = [er[1] - er[0] + 1]
    else:
      raise TypeError(f"Unable to determine the ZoneType for Zone {I.getName(zone_node)}")
    return py_utils.list_or_only_elt(n_face)

  @staticmethod
  def NGonNode(zone_node):
    ngons = [e for e in I.getNodesFromType1(zone_node, "Elements_t") if Element.CGNSName(e) == 'NGON_n']
    return py_utils.expects_one(ngons, ("NGon node", f"zone {I.getName(zone_node)}"))

  @staticmethod
  def NFaceNode(zone_node):
    nfaces = [e for e in I.getNodesFromType1(zone_node, "Elements_t") if Element.CGNSName(e) == 'NFACE_n']
    return py_utils.expects_one(nfaces, ("NFace node", f"zone {I.getName(zone_node)}"))

  @staticmethod
  def VertexBoundarySize(zone_node):
    z_sizes = I.getValue(zone_node)
    return py_utils.list_or_only_elt(z_sizes[:,2])

  @staticmethod
  def Type(zone_node):
    zone_type_node = I.getNodeFromType1(zone_node, "ZoneType_t")
    return I.getValue(zone_type_node)

  #Todo : this one should go elsewhere
  @staticmethod
  def getBCsFromFamily(zone_node, families):
    from maia.pytree import iter_children_from_predicates
    bc_query = lambda n : I.getType(n) == 'BC_t' and I.getValue(n) == 'FamilySpecified' and \
      I.getValue(I.getNodeFromType1(n, "FamilyName_t")) in families
    return iter_children_from_predicates(zone_node, ['ZoneBC_t', bc_query])

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
  def coordinates(zone_node, name=None):
    grid_coord_node = I.getNodeFromType1(zone_node, "GridCoordinates_t") if name is None \
        else I.getNodeFromNameAndType(zone_node, name, "GridCoordinates_t")
    if grid_coord_node is None:
      raise RuntimeError(f"Unable to find GridCoordinates_t node in {I.getName(node)}.")

    x = I.getVal(I.getNodeFromName1(grid_coord_node, "CoordinateX"))
    y = I.getVal(I.getNodeFromName1(grid_coord_node, "CoordinateY"))
    z = I.getVal(I.getNodeFromName1(grid_coord_node, "CoordinateZ"))

    return x, y, z

  @staticmethod
  def ngon_connectivity(zone_node):
    ngon_node = Zone.NGonNode(zone_node)
    face_vtx_idx = I.getVal(I.getNodeFromName1(ngon_node, "ElementStartOffset"))
    face_vtx     = I.getVal(I.getNodeFromName1(ngon_node, "ElementConnectivity"))
    ngon_pe      = I.getVal(I.getNodeFromName1(ngon_node, "ParentElements"))
    return face_vtx_idx, face_vtx, ngon_pe

  @staticmethod
  def get_range_of_ngon(zone):
    """
    Return the ElementRange array of the NGON elements
    """
    return Element.Range(Zone.NGonNode(zone))

  @staticmethod
  def get_ordered_elements(zone):
    """
    Return the elements nodes in increasing order wrt ElementRange
    """
    return sorted(I.getNodesFromType1(zone, 'Elements_t'),
                  key = lambda item : Element.Range(item)[0])

  @staticmethod
  def get_ordered_elements_per_dim(zone):
    """
    Return a list of size 4 containing Element nodes belonging to each dimension.
    In addition, Element are sorted according to their ElementRange withing each dimension.
    """
    # TODO : how to prevent special case of range of elemt mixed in dim ?
    return py_utils.bucket_split(Zone.get_ordered_elements(zone), lambda e: Element.Dimension(e), size=4)

    return sorted_elts_by_dim

  @staticmethod
  def get_elt_range_per_dim(zone):
    """
    Return a list of size 4 containing min & max element id of each dimension
    This function is relevant only if Element of same dimension have consecutive
    ElementRange
    """
    sorted_elts_by_dim = Zone.get_ordered_elements_per_dim(zone)

    range_by_dim = [[0,0], [0,0], [0,0], [0,0]]
    for i_dim, elt_dim in enumerate(sorted_elts_by_dim):
      # Element is sorted
      if(len(elt_dim) > 0):
        range_by_dim[i_dim][0] = Element.Range(elt_dim[0 ])[0]
        range_by_dim[i_dim][1] = Element.Range(elt_dim[-1])[1]

    # Check if element range were not interlaced
    for first, second in itertools.combinations(range_by_dim, 2):
      if py_utils.are_overlapping(first, second, strict=True):
        raise RuntimeError("ElementRange with different dimensions are interlaced")

    return range_by_dim

  @staticmethod
  def elt_ordering_by_dim(zone):
    """Returns 1 if lower dimension elements have lower element range, -1 if
    lower dim. elements have higher element range, and 0 if order can not be determined"""
    status = 0
    sect_start = [r[0] for r in Zone.get_elt_range_per_dim(zone) if r[0] > 0]
    if len(sect_start) >= 2:
      if sect_start[0] < sect_start[-1]:
        status = 1
      elif sect_start[0] > sect_start[-1]:
        status = -1
    return status


# --------------------------------------------------------------------------
@for_all_methods(check_is_label("Elements_t"))
class Element:

  @staticmethod
  def Type(elt_node):
    return elt_node[1][0]

  @staticmethod
  def CGNSName(elt_node):
    return EU.element_name(Element.Type(elt_node))

  @staticmethod
  def Dimension(elt_node):
    return EU.element_dim(Element.Type(elt_node))

  @staticmethod
  def NVtx(elt_node):
     return EU.element_number_of_nodes(Element.Type(elt_node))

  @staticmethod
  def Range(elt_node):
    return I.getNodeFromName(elt_node,"ElementRange")[1]

  @staticmethod
  def Size(elt_node):
    er = Element.Range(elt_node)
    return er[1] - er[0] + 1



@for_all_methods(check_in_labels(["GridConnectivity_t", "GridConnectivity1to1_t"]))
class GridConnectivity:

  @staticmethod
  def Type(gc):
    if I.getType(gc) == 'GridConnectivity1to1_t':
      return 'Abutting1to1'
    elif I.getType(gc) == 'GridConnectivity_t':
      gc_type_n = I.getNodeFromName(gc, 'GridConnectivityType')
      return I.getValue(gc_type_n) if gc_type_n is not None else 'Overset'

  @staticmethod
  def is1to1(gc):
    return GridConnectivity.Type(gc) == 'Abutting1to1'

@for_all_methods(check_in_labels(["FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", \
        "BC_t", "BCDataSet_t", "GridConnectivity_t", "GridConnectivity1to1_t"]))
class Subset:
  """
  A subset is a node having a PointList or a PointRange
  """
  
  def getPatch(subset):
    pl = I.getNodeFromName1(subset, 'PointList')
    pr = I.getNodeFromName1(subset, 'PointRange')
    assert (pl is None) ^ (pr is None)
    return pl if pl is not None else pr

  def n_elem(subset):
    patch = Subset.getPatch(subset)
    return PointList.n_elem(patch) if I.getType(patch) == 'IndexArray_t' else PointRange.n_elem(patch)

  def GridLocation(subset):
    grid_loc_n = I.getNodeFromType1(subset, 'GridLocation_t')
    return I.getValue(grid_loc_n) if grid_loc_n else 'Vertex'

# --------------------------------------------------------------------------
@for_all_methods(check_is_label("IndexRange_t"))
class PointRange:

  @staticmethod
  def SizePerIndex(point_range_node):
    """
    Allow point_range to be inverted (PR[:,1] < PR[:,0]) as it can occurs in struct GCs
    """
    pr_values = point_range_node[1]
    return np.abs(pr_values[:,1] - pr_values[:,0]) + 1

  @staticmethod
  def n_elem(point_range_node):
    return PointRange.SizePerIndex(point_range_node).prod()


# --------------------------------------------------------------------------
@for_all_methods(check_is_label("IndexArray_t"))
class PointList:

  @staticmethod
  def n_elem(point_list_node):
    return I.getVal(point_list_node).shape[1]

