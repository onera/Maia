import numpy as np
import itertools


from maia.pytree.compare import check_is_label, check_in_labels
from maia.pytree         import node as N
from maia.pytree         import walk as W
from . import elements_utils as EU
from . import utils
from .utils import for_all_methods

def _list_or_only_elt(l):
  return l[0] if len(l) == 1 else l

# --------------------------------------------------------------------------
@for_all_methods(check_is_label("Zone_t"))
class Zone:
  @staticmethod
  def VertexSize(zone_node):
    z_sizes = N.get_value(zone_node)
    return _list_or_only_elt(z_sizes[:,0])

  @staticmethod
  def CellSize(zone_node):
    z_sizes = N.get_value(zone_node)
    return _list_or_only_elt(z_sizes[:,1])

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
      er = W.get_child_from_name(ngon_node, 'ElementRange')[1]
      n_face = [er[1] - er[0] + 1]
    else:
      raise TypeError(f"Unable to determine the ZoneType for Zone {N.get_name(zone_node)}")
    return _list_or_only_elt(n_face)

  @staticmethod
  def NGonNode(zone_node):
    predicate = lambda n: N.get_label(n) == "Elements_t" and Element.CGNSName(n) == 'NGON_n'
    ngons = W.get_children_from_predicate(zone_node, predicate)
    return utils.expects_one(ngons, ("NGon node", f"zone {N.get_name(zone_node)}"))

  @staticmethod
  def NFaceNode(zone_node):
    predicate = lambda n: N.get_label(n) == "Elements_t" and Element.CGNSName(n) == 'NFACE_n'
    nfaces = W.get_children_from_predicate(zone_node, predicate)
    return utils.expects_one(nfaces, ("NFace node", f"zone {N.get_name(zone_node)}"))

  @staticmethod
  def VertexBoundarySize(zone_node):
    z_sizes = N.get_value(zone_node)
    return _list_or_only_elt(z_sizes[:,2])

  @staticmethod
  def Type(zone_node):
    zone_type_node = W.get_child_from_label(zone_node, "ZoneType_t")
    return N.get_value(zone_type_node)

  #Todo : this one should go elsewhere
  @staticmethod
  def getBCsFromFamily(zone_node, families):
    from maia.pytree import iter_children_from_predicates
    bc_query = lambda n : N.get_label(n) == 'BC_t' and N.get_value(n) == 'FamilySpecified' and \
      N.get_value(W.get_child_from_label(n, "FamilyName_t")) in families
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
  def has_ngon_elements(zone_node):
    predicate = lambda n: N.get_label(n) == "Elements_t" and Element.CGNSName(n) == 'NGON_n'
    return W.get_child_from_predicate(zone_node, predicate) is not None

  @staticmethod
  def has_nface_elements(zone_node):
    predicate = lambda n: N.get_label(n) == "Elements_t" and Element.CGNSName(n) == 'NFACE_n'
    return W.get_child_from_predicate(zone_node, predicate) is not None

  @staticmethod
  def coordinates(zone_node, name=None):
    grid_coord_node = W.get_child_from_label(zone_node, "GridCoordinates_t") if name is None \
        else W.get_child_from_name_and_label(zone_node, name, "GridCoordinates_t")
    if grid_coord_node is None:
      raise RuntimeError(f"Unable to find GridCoordinates_t node in {N.get_name(node)}.")

    x = N.get_value(W.get_child_from_name(grid_coord_node, "CoordinateX"))
    y = N.get_value(W.get_child_from_name(grid_coord_node, "CoordinateY"))
    z = N.get_value(W.get_child_from_name(grid_coord_node, "CoordinateZ"))

    return x, y, z

  @staticmethod
  def ngon_connectivity(zone_node):
    ngon_node = Zone.NGonNode(zone_node)
    face_vtx_idx = N.get_value(W.get_child_from_name(ngon_node, "ElementStartOffset"))
    face_vtx     = N.get_value(W.get_child_from_name(ngon_node, "ElementConnectivity"))
    ngon_pe      = N.get_value(W.get_child_from_name(ngon_node, "ParentElements"))
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
    return sorted(W.get_children_from_label(zone, 'Elements_t'),
                  key = lambda item : Element.Range(item)[0])

  @staticmethod
  def get_ordered_elements_per_dim(zone):
    """
    Return a list of size 4 containing Element nodes belonging to each dimension.
    In addition, Element are sorted according to their ElementRange withing each dimension.
    """
    # TODO : how to prevent special case of range of elemt mixed in dim ?
    return utils.bucket_split(Zone.get_ordered_elements(zone), lambda e: Element.Dimension(e), size=4)

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
      if utils.are_overlapping(first, second, strict=True):
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
    return int(elt_node[1][0])

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
    return W.get_child_from_name(elt_node,"ElementRange")[1]

  @staticmethod
  def Size(elt_node):
    er = Element.Range(elt_node)
    return er[1] - er[0] + 1



@for_all_methods(check_in_labels(["GridConnectivity_t", "GridConnectivity1to1_t"]))
class GridConnectivity:

  @staticmethod
  def Type(gc):
    if N.get_label(gc) == 'GridConnectivity1to1_t':
      return 'Abutting1to1'
    elif N.get_label(gc) == 'GridConnectivity_t':
      gc_type_n = W.get_child_from_name(gc, 'GridConnectivityType')
      return N.get_value(gc_type_n) if gc_type_n is not None else 'Overset'

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
    pl = W.get_child_from_name(subset, 'PointList')
    pr = W.get_child_from_name(subset, 'PointRange')
    assert (pl is None) ^ (pr is None)
    return pl if pl is not None else pr

  def n_elem(subset):
    patch = Subset.getPatch(subset)
    return PointList.n_elem(patch) if N.get_label(patch) == 'IndexArray_t' else PointRange.n_elem(patch)

  def GridLocation(subset):
    grid_loc_n = W.get_child_from_label(subset, 'GridLocation_t')
    return N.get_value(grid_loc_n) if grid_loc_n else 'Vertex'

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
    return N.get_value(point_list_node).shape[1]

