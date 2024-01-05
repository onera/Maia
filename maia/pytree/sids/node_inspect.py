import numpy as np
import itertools

from maia.pytree.typing import *

from maia.pytree.compare import check_is_label, check_in_labels
from maia.pytree         import node as N
from maia.pytree         import walk as W
from . import elements_utils as EU
from . import utils
from .utils import for_all_methods

# Custom NamedTuple list
class PeriodicValues(NamedTuple):
  RotationCenter:ArrayLike
  RotationAngle:ArrayLike
  Translation:ArrayLike
  def asdict(self, snake_case=False):
    if snake_case:
      return  {'rotation_center' : self.RotationCenter,
              'rotation_angle' : self.RotationAngle,
               'translation' : self.Translation}
    else:
      return self._asdict()


# --------------------------------------------------------------------------
@for_all_methods(check_is_label("CGNSTree_t"))
class Tree:
  """The following functions apply to the top level CGNSTree_t"""

  @staticmethod
  def find_connected_zones(tree:CGNSTree) -> List[List[str]]:
    """ Gather the zones of the tree into groups connected by 1to1 connectivities

    Only non periodic joins are considered. Output paths start at tree level
    (Basename/Zonename).

    Args:
      tree (CGNSTree): Input CGNSTree_t node
    Returns:
      List[List[str]]: grouped paths of zones
    Example:
      >>> tree = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
      ... Base CGNSBase_t:
      ...   Zone1 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone3":
      ...   Zone2 Zone_t:
      ...   Zone3 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone1":
      ... ''')
      >>> PT.Tree.find_connected_zones(tree)
      [['Base/Zone2'], ['Base/Zone1', 'Base/Zone3']]

    """
    import maia.pytree as PT
    connected_zones = []
    matching_gcs_u = lambda n : PT.get_label(n) == 'GridConnectivity_t' and PT.GridConnectivity.is1to1(n)
    matching_gcs_s = lambda n : PT.get_label(n) == 'GridConnectivity1to1_t'
    matching_gcs = lambda n : (matching_gcs_u(n) or matching_gcs_s(n)) \
                            and PT.get_child_from_label(n, 'GridConnectivityProperty_t') is None
    
    for zone_path in PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t'):
      group     = [zone_path]
      zone = PT.get_node_from_path(tree, zone_path)
      for gc in PT.iter_children_from_predicates(zone, ['ZoneGridConnectivity_t', matching_gcs]):
        opp_zone_path = GridConnectivity.ZoneDonorPath(gc, PT.path_head(zone_path))
        utils.append_unique(group, opp_zone_path)
      connected_zones.append(group)

    for zone_path in PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t'):
      groups_to_merge = []
      for i, group in enumerate(connected_zones):
        if zone_path in group:
          groups_to_merge.append(i)
      if groups_to_merge != []:
        new_group = []
        for i in groups_to_merge[::-1]: #Reverse loop to pop without changing idx
          zones_paths = connected_zones.pop(i)
          for z_p in zones_paths:
            utils.append_unique(new_group, z_p)
        connected_zones.append(new_group)
    return [sorted(zones) for zones in connected_zones]

  @staticmethod
  def find_periodic_jns(tree: CGNSTree, rtol=1e-5, atol=0.) -> Tuple[List[PeriodicValues], List[List[str]]]:
    """ Gather the periodic joins of the tree according to their periodicity values

    Returned paths starts at tree level.

    Args:
      tree (CGNSTree): Input CGNSTree_t node
      rtol (float, optional): relative tolerance for periodicity comparaison
      atol (float, optional): absolute tolerance for periodicity comparaison
    Returns:
       Pair of lists: at each indice,

      - first list contains the periodic arrays for the group of joins
      - second list contains the paths of joins related to this value
    """
    import maia.pytree as PT
    perio_values = list()
    perio_jns = list()
    is_jn_perio = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
            and PT.GridConnectivity.isperiodic(n)
    for jn_path in PT.predicates_to_paths(tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', is_jn_perio]):
      jn = PT.get_node_from_path(tree, jn_path)
      perio = PT.GridConnectivity.periodic_values(jn)
      for i_perio, perio_value in enumerate(perio_values):
        if all([np.allclose(a,b,rtol,atol) for a,b in zip(perio_value, perio)]):
          perio_jns[i_perio].append(jn_path)
          break
      else:
        perio_values.append(perio)
        perio_jns.append([jn_path])

    return perio_values, perio_jns


# --------------------------------------------------------------------------
@for_all_methods(check_is_label("Zone_t"))
class Zone:
  """The following functions apply to any Zone_t node"""

  @staticmethod
  def IndexDimension(zone_node:CGNSTree) -> int:
    """
    Return the IndexDimension of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int : IndexDimension (1,2 or 3)
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[11,10,0]])
      >>> PT.Zone.IndexDimension(zone)
      1
    """
    z_sizes = N.get_value(zone_node)
    return (z_sizes[:,0]).size

  @staticmethod
  def VertexSize(zone_node:CGNSTree) -> Union[int, np.ndarray]:
    """
    Return the number of vertices per direction of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int or ndarray of int : number of vertices in each direction 
      for structured zones, total number of vertices for unstructured zones
    Example:
      >>> zone = PT.new_Zone(type='Structured', size=[[11,10,0], [6,5,0], [2,1,0]])
      >>> PT.Zone.VertexSize(zone)
      array([11, 6, 2], dtype=int32)
    """
    sizes = N.get_value(zone_node)[:,0]
    return sizes[0] if Zone.Type(zone_node) == 'Unstructured' else sizes

  @staticmethod
  def CellSize(zone_node:CGNSTree) -> Union[int, np.ndarray]:
    """
    Return the number of cells per direction of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int or ndarray of int : number of cells in each direction 
      for structured zones, total number of cells for unstructured zones
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[11,10,0]])
      >>> PT.Zone.CellSize(zone)
      10
    """
    sizes = N.get_value(zone_node)[:,1]
    return sizes[0] if Zone.Type(zone_node) == 'Unstructured' else sizes

  @staticmethod
  def FaceSize(zone_node:CGNSTree) -> Union[int, np.ndarray]:
    """
    Return the number of faces per direction of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int or list of int : number of faces in each direction 
      for structured zones, total number of faces for unstructured zones
    Example:
      >>> zone = PT.new_Zone(type='Structured', size=[[11,10,0], [6,5,0]])
      >>> PT.Zone.FaceSize(zone)
      [55, 60]

    Warning:
      Unstructured zones are supported only if they have a NGon connectivity
    """
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
      n_face = er[1] - er[0] + 1
    else:
      raise TypeError(f"Unable to determine the ZoneType for Zone {N.get_name(zone_node)}")
    return n_face

  @staticmethod
  def NGonNode(zone_node:CGNSTree) -> CGNSTree:
    """Return the Element_t node of kind ``NGON_n`` of a Zone_t node
    
    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      CGNSTree : NGon node
    Raises:
      RuntimeError: if not exactly one ``NGON_n`` element node exists in zone
    """
    predicate = lambda n: N.get_label(n) == "Elements_t" and Element.CGNSName(n) == 'NGON_n'
    ngons = W.get_children_from_predicate(zone_node, predicate)
    return utils.expects_one(ngons, ("NGon node", f"zone {N.get_name(zone_node)}"))

  @staticmethod
  def NFaceNode(zone_node:CGNSTree) -> CGNSTree:
    """Return the Element_t node of kind ``NFACE_n`` of a Zone_t node
    
    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      CGNSTree : NFace node
    Raises:
      RuntimeError: if not exactly one ``NFACE_n`` element node exists in zone
    """
    predicate = lambda n: N.get_label(n) == "Elements_t" and Element.CGNSName(n) == 'NFACE_n'
    nfaces = W.get_children_from_predicate(zone_node, predicate)
    return utils.expects_one(nfaces, ("NFace node", f"zone {N.get_name(zone_node)}"))

  @staticmethod
  def VertexBoundarySize(zone_node:CGNSTree) -> Union[int, np.ndarray]:
    """
    Return the number of boundary vertices per direction of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int or ndarray of int : number of boundary vtx in each direction 
      for structured zones, total number of boundary vtx for unstructured zones
    Example:
      >>> zone = PT.new_Zone(type='Structured', size=[[11,10,0],[6,5,0]])
      >>> PT.Zone.VertexBoundarySize(zone)
      array([0, 0], dtype=int32)
    """
    sizes = N.get_value(zone_node)[:,2]
    return sizes[0] if Zone.Type(zone_node) == 'Unstructured' else sizes

  @staticmethod
  def Type(zone_node:CGNSTree) -> str:
    """
    Return the kind of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      str : One of 'Structured', 'Unstructured', 'UserDefined' or 'Null'
    Example:
      >>> zone = PT.new_Zone(type='Unstructured')
      >>> PT.Zone.Type(zone)
      'Unstructured'
    """
    zone_type_node = W.get_child_from_label(zone_node, "ZoneType_t")
    return N.get_value(zone_type_node)

  @staticmethod
  def n_vtx(zone_node:CGNSTree) -> int:
    """
    Return the total number of vertices of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int : number of vertices
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[11,10,0]])
      >>> PT.Zone.n_vtx(zone)
      11
    """
    return np.prod(Zone.VertexSize(zone_node))

  @staticmethod
  def n_cell(zone_node:CGNSTree) -> int:
    """
    Return the total number of cells of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int : number of cells
    Example:
      >>> zone = PT.new_Zone(type='Structured', size=[[11,10,0], [6,5,0]])
      >>> PT.Zone.n_cell(zone)
      50
    """
    return np.prod(Zone.CellSize(zone_node))

  @staticmethod
  def n_face(zone_node:CGNSTree) -> int:
    """
    Return the total number of faces of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int : number of faces
    Example:
      >>> zone = PT.new_Zone(type='Structured', size=[[11,10,0], [6,5,0]])
      >>> PT.Zone.n_face(zone)
      115

    Warning:
      Unstructured zones are supported only if they have a NGon connectivity
    """
    return np.sum(Zone.FaceSize(zone_node))

  @staticmethod
  def n_vtx_bnd(zone_node:CGNSTree) -> int:
    """
    Return the total number of boundary vertices of a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int : number of boundary vertices
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[11,10,0]])
      >>> PT.Zone.n_vtx_bnd(zone)
      0
    """
    return np.prod(Zone.VertexBoundarySize(zone_node))

  @staticmethod
  def has_ngon_elements(zone_node: CGNSTree) -> bool:
    """ Return True if some Element_t node of kind ``NGON_n`` exists in the Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      bool
    Example:
      >>> zone = PT.new_Zone(type='Structured', size=[[11,10,0], [6,5,0]])
      >>> PT.new_NGonElements(parent=zone)
      >>> PT.Zone.has_ngon_elements(zone)
      True
    """
    predicate = lambda n: N.get_label(n) == "Elements_t" and Element.CGNSName(n) == 'NGON_n'
    return W.get_child_from_predicate(zone_node, predicate) is not None

  @staticmethod
  def has_nface_elements(zone_node:CGNSTree) -> bool:
    """ Return True if some Element_t node of kind ``NFACE_n`` exists in the Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      bool
    Example:
      >>> zone = PT.new_Zone(type='Structured', size=[[11,10,0], [6,5,0]])
      >>> PT.Zone.has_nface_elements(zone)
      False
    """
    predicate = lambda n: N.get_label(n) == "Elements_t" and Element.CGNSName(n) == 'NFACE_n'
    return W.get_child_from_predicate(zone_node, predicate) is not None

  @staticmethod
  def coordinates(zone_node:CGNSTree, name:str=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """ Return the coordinate arrays of the Zone_t node

    Only cartesian coordinates are supported.

    Args:
      zone_node (CGNSTree): Input Zone_t node
      name (str, optional): Name of the GridCoordinates node from which coordinates are taken.
        If not specified, first container found is used.
    Returns:
      Triplet of ndarray or None: for each direction, corresponding coordinate array or ``None``
      if physicalDimension is != 3
    Example:
      >>> zone = PT.new_Zone(type='Unstructured')
      >>> PT.new_GridCoordinates(fields={'CoordinateX' : [0., 0.5, 1.], 
      ...                                'CoordinateY' : [.5, .5, .5]},
      ...                        parent=zone)
      >>> PT.Zone.coordinates(zone)
      (array([0.,0.5,1.], dtype=float32), array([0.5,0.5,0.5], dtype=float32), None)
    """
    grid_coord_node = W.get_child_from_label(zone_node, "GridCoordinates_t") if name is None \
        else W.get_child_from_name_and_label(zone_node, name, "GridCoordinates_t")
    if grid_coord_node is None:
      raise RuntimeError(f"Unable to find GridCoordinates_t node in {N.get_name(zone_node)}.")

    x_node = W.get_child_from_name(grid_coord_node, "CoordinateX")
    y_node = W.get_child_from_name(grid_coord_node, "CoordinateY")
    z_node = W.get_child_from_name(grid_coord_node, "CoordinateZ")
    x = N.get_value(x_node) if x_node else None
    y = N.get_value(y_node) if y_node else None
    z = N.get_value(z_node) if z_node else None

    return x, y, z

  @staticmethod
  def ngon_connectivity(zone_node:CGNSTree) -> List[np.ndarray]:
    ngon_node = Zone.NGonNode(zone_node)
    face_vtx_idx = N.get_value(W.get_child_from_name(ngon_node, "ElementStartOffset"))
    face_vtx     = N.get_value(W.get_child_from_name(ngon_node, "ElementConnectivity"))
    ngon_pe      = N.get_value(W.get_child_from_name(ngon_node, "ParentElements"))
    return face_vtx_idx, face_vtx, ngon_pe

  @staticmethod
  def get_ordered_elements(zone_node:CGNSTree) -> List[CGNSTree]:
    """ Return the Elements under a Zone_t node, sorted according to their ElementRange
    
    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      list of CGNSTree : Elements_t nodes
    Example:
      >>> zone = PT.new_Zone(type='Unstructured')
      >>> PT.new_Elements('TETRA', 'TETRA_4', erange=[1,10], parent=zone)
      >>> PT.new_Elements('TRI', 'TRI_3', erange=[11,30], parent=zone)
      >>> [PT.get_name(node) for node in PT.Zone.get_ordered_elements(zone)]
      ['TETRA', 'TRI']
    """
    return sorted(W.get_children_from_label(zone_node, 'Elements_t'),
                  key = lambda item : Element.Range(item)[0])

  @staticmethod
  def get_ordered_elements_per_dim(zone_node:CGNSTree) -> List[List[CGNSTree]]:
    """Return the Elements under a Zone_t node, gathered according to their dimension

    Within each dimension, Elements are sorted according to their ElementRange

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      list of 4 list of CGNSTree : Elements_t nodes
    Example:
      >>> zone = PT.new_Zone(type='Unstructured')
      >>> PT.new_Elements('PYRA', 'PYRA_5', erange=[1,10],  parent=zone)
      >>> PT.new_Elements('TRI',  'TRI_3',  erange=[11,30], parent=zone)
      >>> PT.new_Elements('QUAD', 'QUAD_4', erange=[31,40], parent=zone)
      >>> [len(elts) for elts in PT.Zone.get_ordered_elements_per_dim(zone)]
      [0, 0, 2, 1]
    """
    # TODO : how to prevent special case of range of elemt mixed in dim ?
    return utils.bucket_split(Zone.get_ordered_elements(zone_node), lambda e: Element.Dimension(e), size=4)

  @staticmethod
  def get_elt_range_per_dim(zone_node:CGNSTree) -> List[List[int]]:
    """ Return the min & max element number of each dimension found in a Zone_t node

    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      list of 4 pairs : min and max element id for each dimension
    Raises:
      RuntimeError: if elements of different dimension are interlaced
    Example:
      >>> zone = PT.new_Zone(type='Unstructured')
      >>> PT.new_Elements('PYRA', 'PYRA_5', erange=[1,10],  parent=zone)
      >>> PT.new_Elements('TRI',  'TRI_3',  erange=[11,30], parent=zone)
      >>> PT.new_Elements('BAR',  'BAR_2',  erange=[31,40], parent=zone)
      >>> PT.Zone.get_elt_range_per_dim(zone)
      [[0, 0], [31, 40], [11, 30], [1, 10]]
    """
    sorted_elts_by_dim = Zone.get_ordered_elements_per_dim(zone_node)

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
  def elt_ordering_by_dim(zone_node:CGNSTree):
    """Return a flag indicating if elements belonging to a Zone_t node are sorted
    
    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      int : Flag indicating how elements are sorted:
      
      - 1 if elements of lower dimension have lower ElementRange
      - \- 1 if elements of lower dimension have higher ElementRange
      - 0 if elements are not sorted
    Example:
      >>> zone = PT.new_Zone(type='Unstructured')
      >>> PT.new_Elements('TETRA', 'TETRA_4', erange=[1,10], parent=zone)
      >>> PT.new_Elements('TRI', 'TRI_3', erange=[11,30], parent=zone)
      >>> PT.Zone.elt_ordering_by_dim(zone)
      -1
    """
    status = 0
    sect_start = [r[0] for r in Zone.get_elt_range_per_dim(zone_node) if r[0] > 0]
    if len(sect_start) >= 2:
      if sect_start[0] < sect_start[-1]:
        status = 1
      elif sect_start[0] > sect_start[-1]:
        status = -1
    return status


# --------------------------------------------------------------------------
@for_all_methods(check_is_label("Elements_t"))
class Element:

  """The following functions apply to any Element_t node"""

  @staticmethod
  def Type(elt_node:CGNSTree) -> int:
    """ Return the type of an Element_t node

    Args:
      elt_node (CGNSTree): Input Element_t node
    Returns:
      int : CGNS code corresponding to this element kind
    Example:
      >>> elt = PT.new_Elements(type='TETRA_4')
      >>> PT.Element.Type(elt)
      10
    """
    return int(elt_node[1][0])

  @staticmethod
  def CGNSName(elt_node:CGNSTree) -> str:
    """ Return the generic name of an Element_t node

    Args:
      elt_node (CGNSTree): Input Element_t node
    Returns:
      str : CGNS name corresponding to this element kind
    Example:
      >>> elt = PT.new_NFaceElements('MyElements')
      >>> PT.Element.CGNSName(elt)
      'NFACE_n'
    """
    return EU.element_name(Element.Type(elt_node))

  @staticmethod
  def Dimension(elt_node:CGNSTree) -> int:
    """ Return the dimension of an Element_t node

    Args:
      elt_node (CGNSTree): Input Element_t node
    Returns:
      int : Dimension of this element kind (0, 1, 2 or 3)
    Example:
      >>> elt = PT.new_Elements(type='TRI_3')
      >>> PT.Element.Dimension(elt)
      2
    """
    return EU.element_dim(Element.Type(elt_node))

  @staticmethod
  def NVtx(elt_node:CGNSTree) -> int:
    """ Return the number of vertices of an Element_t node

    Args:
      elt_node (CGNSTree): Input Element_t node
    Returns:
      int : Number of vertices for this element kind
    Example:
      >>> elt = PT.new_Elements(type='PYRA_5')
      >>> PT.Element.NVtx(elt)
      5
    """
    return EU.element_number_of_nodes(Element.Type(elt_node))

  @staticmethod
  def Range(elt_node:CGNSTree) -> np.ndarray:
    """ Return the value of the ElementRange of an Element_t node

    Args:
      elt_node (CGNSTree): Input Element_t node
    Returns:
      ndarray : ElementRange of the node
    Example:
      >>> elt = PT.new_Elements(type='PYRA_5', erange=[21,40])
      >>> PT.Element.Range(elt)
      array([21, 40], dtype=int32)
    """
    return W.get_child_from_name(elt_node,"ElementRange")[1]

  @staticmethod
  def Size(elt_node:CGNSTree) -> int:
    """ Return the size (number of elements) of an Element_t node

    Args:
      elt_node (CGNSTree): Input Element_t node
    Returns:
      int : Number of elements described by the node
    Example:
      >>> elt = PT.new_Elements(type='PYRA_5', erange=[21,40])
      >>> PT.Element.Size(elt)
      20
    """
    er = Element.Range(elt_node)
    return er[1] - er[0] + 1



@for_all_methods(check_in_labels(["GridConnectivity_t", "GridConnectivity1to1_t"]))
class GridConnectivity:

  @staticmethod
  def Type(gc_node:CGNSTree) -> str:
    """ Return the type of a GridConnectivity node

    Args:
      gc_node (CGNSTree): Input GridConnectivity node
    Returns:
      str : One of 'Null', 'UserDefined', 'Overset', 'Abutting' or 'Abutting1to1'
    Example:
      >>> gc = PT.new_GridConnectivity1to1('GC')
      >>> PT.GridConnectivity.Type(gc)
      'Abutting1to1'
    """
    if N.get_label(gc_node) == 'GridConnectivity1to1_t':
      return 'Abutting1to1'
    elif N.get_label(gc_node) == 'GridConnectivity_t':
      gc_type_n = W.get_child_from_name(gc_node, 'GridConnectivityType')
      return N.get_value(gc_type_n) if gc_type_n is not None else 'Overset'

  @staticmethod
  def is1to1(gc_node:CGNSTree) -> bool:
    """ Return True if the GridConnectivity node is of type 'Abutting1to1'

    Args:
      gc_node (CGNSTree): Input GridConnectivity node
    Returns:
      bool
    Example:
      >>> gc = PT.new_GridConnectivity('GC', type='Overset')
      >>> PT.GridConnectivity.is1to1(gc)
      False
    """
    return GridConnectivity.Type(gc_node) == 'Abutting1to1'

  @staticmethod
  def isperiodic(gc_node:CGNSTree) -> bool:
    """ Return True if the GridConnectivity node is periodic

    Args:
      gc_node (CGNSTree): Input GridConnectivity node
    Returns:
      bool
    Example:
      >>> gc = PT.new_GridConnectivity('GC', type='Overset')
      >>> PT.GridConnectivity.isperiodic(gc)
      False
    """
    return W.get_node_from_label(gc_node, 'Periodic_t', depth=[2,2]) is not None

  @staticmethod
  def periodic_values(gc_node:CGNSTree) -> PeriodicValues:
    """ Return the periodic transformation of a GridConnectivity node

    Args:
      gc_node (CGNSTree): Input GridConnectivity node
    Returns:
      Triplet of ndarray or None : values of RotationCenter, RotationAngle and Translation, stored
      in a named tuple
    Example:
      >>> gc = PT.new_GridConnectivity('GC')
      >>> PT.new_GridConnectivityProperty({'translation' : [1., 0, 0]}, parent=gc)
      >>> PT.GridConnectivity.periodic_values(gc)
      PeriodicValues(RotationCenter=array([0., 0., 0.], dtype=float32), 
                     RotationAngle=array([0., 0., 0.], dtype=float32),
                     Translation=array([1., 0., 0.], dtype=float32))
    """
    perio_node = W.get_node_from_label(gc_node, "Periodic_t", depth=[2,2])
    return PeriodicValues._make([W.get_child_from_name(perio_node, key)[1] if perio_node else None
                                 for key in PeriodicValues._fields])

  @staticmethod
  def ZoneDonorPath(gc_node:CGNSTree, cur_base_name:str) -> str:
    """ Return the path of the opposite zone of a GridConnectivity node

    Args:
      gc_node (CGNSTree): Input GridConnectivity node
      cur_base_name (str): Name of the parent base of this node
    Returns:
      str : Path (Basename/Zonename) of the opposite zone
    Example:
      >>> gc = PT.new_GridConnectivity('GC', donor_name='OtherBase/OppZone')
      >>> PT.GridConnectivity.ZoneDonorPath(gc, 'Base')
      'OtherBase/OppZone'
    """
    opp = N.get_value(gc_node)
    return opp if '/' in opp else cur_base_name + '/' + opp


@for_all_methods(check_in_labels(["FlowSolution_t", "DiscreteData_t", "ZoneSubRegion_t", \
        "BC_t", "BCDataSet_t", "GridConnectivity_t", "GridConnectivity1to1_t"]))
class Subset:
  """
  A subset is a node having a PointList or a PointRange
  """
  
  @staticmethod
  def getPatch(subset_node:CGNSTree) -> CGNSTree:
    """ Return the PointList or PointRange node defining the Subset node

    Args:
      subset_node (CGNSTree): Input Subset node
    Returns:
      CGNSTree : PointList or PointRange node
    Example:
      >>> bc = PT.new_BC('BC', loc='FaceCenter', point_list=[[1,2,3,4]])
      >>> PT.Subset.getPatch(bc)
      ['PointList', array([[1, 2, 3, 4]], dtype=int32), [], 'IndexArray_t']
    """
    pl = W.get_child_from_name(subset_node, 'PointList')
    pr = W.get_child_from_name(subset_node, 'PointRange')
    assert (pl is None) ^ (pr is None)
    return pl if pl is not None else pr

  @staticmethod
  def n_elem(subset_node:CGNSTree) -> int:
    """ Return the number of mesh elements included in a Subset node

    Args:
      subset_node (CGNSTree): Input Subset node
    Returns:
      int : Number of elements
    Example:
      >>> gc = PT.new_GridConnectivity1to1('GC', point_range=[[10,1],[1,10]])
      >>> PT.Subset.n_elem(gc)
      100
      >>> bc = PT.new_BC('BC', loc='FaceCenter', point_list=[[1,2,3,4]])
      >>> PT.Subset.n_elem(bc)
      4
    """
    patch = Subset.getPatch(subset_node)
    return PointList.n_elem(patch) if N.get_label(patch) == 'IndexArray_t' else PointRange.n_elem(patch)

  @staticmethod
  def GridLocation(subset_node:CGNSTree) -> str:
    """ Return the GridLocation value of a Subset node

    Args:
      subset_node (CGNSTree): Input Subset node
    Returns:
      str : One of 'Null', 'UserDefined', 'Vertex', 'CellCenter', 'FaceCenter',
      'IFaceCenter', 'JFaceCenter', 'KFaceCenter', or 'EdgeCenter'
    Example:
      >>> bc = PT.new_BC('BC', loc='FaceCenter')
      >>> PT.Subset.GridLocation(bc)
      'FaceCenter'
    """
    grid_loc_n = W.get_child_from_label(subset_node, 'GridLocation_t')
    return N.get_value(grid_loc_n) if grid_loc_n else 'Vertex'

  @staticmethod
  def ZSRExtent(zsr_node:CGNSTree, zone_node:CGNSTree) -> str:
    """
    Return the path of the node to which the ZoneSubRegion node maps

    Path start from zone_node and can point to a BC, a GC or the ZSR itself.
    This function only make sense for ZoneSubRegion_t nodes.

    Args:
      zsr_node (CGNSTree): Input ZoneSubRegion_t node
      zone_node (CGNSTree): Parent Zone_t node
    Returns:
      str : path of zone defining the ZSR extent
    Example:
      >>> zone = PT.new_Zone('Zone')
      >>> bc   = PT.new_BC('RelevantBC', parent=PT.new_ZoneBC(parent=zone))
      >>> zsr  = PT.new_ZoneSubRegion('ZSR', bc_name='RelevantBC')
      >>> PT.Subset.ZSRExtent(zsr, zone)
      'ZoneBC/RelevantBC'
    """
    import maia.pytree as PT
    bc_region_name = W.get_child_from_name(zsr_node, "BCRegionName")
    gc_region_name = W.get_child_from_name(zsr_node, "GridConnectivityRegionName")
    if bc_region_name is not None:
      is_bc = lambda n: N.get_label(n) == "BC_t" and \
                        N.get_name(n) == N.get_value(bc_region_name)
      paths = PT.predicates_to_paths(zone_node, ['ZoneBC_t', is_bc])
    elif gc_region_name is not None:
      is_gc = lambda n: N.get_label(n) in ["GridConnectivity1to1_t", "GridConnectivity_t"] and \
                        N.get_name(n) == N.get_value(gc_region_name)
      paths = PT.predicates_to_paths(zone_node, ['ZoneGridConnectivity_t', is_gc])
    else:
      paths = [N.get_name(zsr_node)]

    try:
      return utils.expects_one(paths)
    except RuntimeError:
      raise ValueError("ZoneSubRegion {0} has no valid extent".format(N.get_name(zsr_node)))

# --------------------------------------------------------------------------
@for_all_methods(check_is_label("IndexRange_t"))
class PointRange:

  @staticmethod
  def SizePerIndex(point_range_node:CGNSTree) -> np.ndarray:
    """
    Allow point_range to be inverted (PR[:,1] < PR[:,0]) as it can occurs in struct GCs
    """
    pr_values = point_range_node[1]
    return np.abs(pr_values[:,1] - pr_values[:,0]) + 1

  @staticmethod
  def n_elem(point_range_node:CGNSTree) -> int:
    return PointRange.SizePerIndex(point_range_node).prod()


# --------------------------------------------------------------------------
@for_all_methods(check_is_label("IndexArray_t"))
class PointList:

  @staticmethod
  def n_elem(point_list_node:CGNSTree) -> int:
    return N.get_value(point_list_node).shape[1]

