import typing

import maia.pytree as PT
from   maia.pytree.typing import *
from maia.typing import MPIComm

from maia.utils import vstride as vs
from maia.utils import par_utils

from . import search
from . conventions import GLBNUM_NAME

__all__ = ['Zone', 'Element', 'Subset', 'Container']

# --------------------------------------------------------------------------
def is_single_node(X:Union[CGNSTree, List[CGNSTree]]) -> bool:
  if len(X) != 4:
    return False
  else:
    return isinstance(X[0], str)

def _n_entity(input:Union[CGNSTree, List[CGNSTree]], comm:Optional[MPIComm], name:str) -> int:
  if is_single_node(input):
    # Distributed implementation
    node = typing.cast(CGNSTree, input)
    distri = PT.get_np_value(search.find_Distribution(node, name))
    return int(distri[2])
  else:
    # Partitioned implementation
    nodes = typing.cast(List[CGNSTree], input)
    assert comm is not None
    gnum_l = [PT.get_np_value(search.find_GlobalNumbering(n, name)) for n in nodes]
    return int(par_utils.arrays_max(gnum_l, comm))

class Zone:
  """ The following functions apply to Zone_t nodes """

  @staticmethod
  def dn_cell(zone_node:CGNSTree) -> int:
    """ Return the local number of cells of a **distributed** zone

    Args:
      zone_node (CGNSDistTree): Input Zone_t node
    Returns:
      int : local number of cells
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[77, 60, 0]])
      >>> MT.new_Distribution({'Cell' : [30, 45, 60]}, parent=zone)
      >>> MT.Zone.dn_cell(zone)
      15
    """
    # Return the local number of cells (only for distributed zones)
    distri = Zone.cell_distribution(zone_node)
    return int(distri[1] - distri[0])

  @staticmethod
  def pn_cell(zone_node:CGNSTree) -> int:
    """ Return the local number of cells of a **partitioned** zone

    Args:
      zone_node (CGNSPartTree): Input Zone_t node
    Returns:
      int : local number of cells
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[14, 6, 0]])
      >>> MT.new_GlobalNumbering({'Cell' : [21,61,41,51,11,31]}, parent=zone)
      >>> MT.Zone.pn_cell(zone)
      6
    """
    gnum = PT.get_np_value(search.find_GlobalNumbering(zone_node, 'Cell'))
    return gnum.size

  @staticmethod
  def n_cell(zone_node:Union[CGNSTree, List[CGNSTree]], comm:Optional[MPIComm]=None) -> int:
    """ Return the **total** number of cells of a zone.

    The input zone can be either distributed (a single node is expected) or partitioned
    (the whole list of partitions is expected).

    Args:
      zone_node (CGNSDistTree or List[CGNSPartTree]): Input Zone_t node(s)
      comm (MPIComm): MPI communicator, mandatory for partitioned zones
    Returns:
      int : total number of cells
    Examples:
      >>> zone = PT.new_Zone('Zone', type='Unstructured', size=[[14, 6, 0]])
      >>> MT.new_Distribution({'Cell' : [3,6,6]}, parent=zone)
      >>> MT.Zone.n_cell(zone)
      6

      >>> zones = [PT.new_Zone('Zone.P1.N0', type='Unstructured', size=[[6, 2, 0]]),
      ...          PT.new_Zone('Zone.P1.N1', type='Unstructured', size=[[10, 4, 0]])]
      >>> MT.new_GlobalNumbering({'Cell' : [4,1]},     parent=zones[0])
      >>> MT.new_GlobalNumbering({'Cell' : [2,6,3,5]}, parent=zones[1])
      >>> MT.Zone.n_cell(zones, comm)
      6
    """
    return _n_entity(zone_node, comm, 'Cell')

  @staticmethod
  def dn_vtx(zone_node:CGNSTree) -> int:
    """ Return the local number of vertices of a **distributed** zone

    Args:
      zone_node (CGNSDistTree): Input Zone_t node
    Returns:
      int : local number of vertices
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[77, 60, 0]])
      >>> MT.new_Distribution({'Vertex' : [39, 58, 77]}, parent=zone)
      >>> MT.Zone.dn_vtx(zone)
      19
    """
    distri = Zone.vtx_distribution(zone_node)
    return int(distri[1] - distri[0])

  @staticmethod
  def pn_vtx(zone_node:CGNSTree) -> int:
    """ Return the local number of vertices of a **partitioned** zone

    Args:
      zone_node (CGNSPartTree): Input Zone_t node
    Returns:
      int : local number of vertices
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[8, 3, 0]])
      >>> MT.new_GlobalNumbering({'Vertex' : [9,11,13,14,16,10,12,15]}, parent=zone)
      >>> MT.Zone.pn_vtx(zone)
      8
    """
    gnum = PT.get_np_value(search.find_GlobalNumbering(zone_node, 'Vertex'))
    return gnum.size


  @staticmethod
  def n_vtx(zone_node:Union[CGNSTree, List[CGNSTree]], comm:Optional[MPIComm]=None) -> int:
    """ Return the **total** number of vertices of a zone.

    The input zone can be either distributed (a single node is expected) or partitioned
    (the whole list of partitions is expected).

    Args:
      zone_node (CGNSDistTree or List[CGNSPartTree]): Input Zone_t node(s)
      comm (MPIComm): MPI communicator, mandatory for partitioned zones
    Returns:
      int : total number of vertices
    Examples:
      >>> zone = PT.new_Zone('Zone', type='Unstructured', size=[[5, 4, 0]])
      >>> MT.new_Distribution({'Vertex' : [3,5,5]}, parent=zone)
      >>> MT.Zone.n_vtx(zone)
      5

      >>> zones = [PT.new_Zone('Zone.P1.N0', type='Unstructured', size=[[3, 2, 0]]),
      ...          PT.new_Zone('Zone.P1.N1', type='Unstructured', size=[[3, 2, 0]])]
      >>> MT.new_GlobalNumbering({'Vertex' : [1,3,2]}, parent=zones[0])
      >>> MT.new_GlobalNumbering({'Vertex' : [3,4,5]}, parent=zones[1])
      >>> MT.Zone.n_vtx(zones, comm)
      5
    """
    return _n_entity(zone_node, comm, 'Vertex')

  @staticmethod
  def vtx_distribution(zone_node:CGNSTree) -> NDArray:
    """ Return the vertices distribution array of a **distributed** zone

    Args:
      zone_node (CGNSDistTree): Input Zone_t node
    Returns:
      NDArray : distribution array for vertices
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[77, 60, 0]])
      >>> MT.new_Distribution({'Vertex' : [39, 58, 77]}, parent=zone)
      >>> MT.Zone.vtx_distribution(zone)
      array([39, 58, 77], dtype=int32)
    """
    return PT.get_np_value(search.find_Distribution(zone_node, 'Vertex'))
  @staticmethod
  def face_distribution(zone_node:CGNSTree) -> NDArray:
    return PT.get_np_value(search.find_Distribution(zone_node, 'Face'))
  @staticmethod
  def face_globalnumbering(zone_node:CGNSTree) -> NDArray:
    return PT.get_np_value(search.find_GlobalNumbering(zone_node, 'Face'))
  @staticmethod
  def cell_distribution(zone_node:CGNSTree) -> NDArray:
    """ Return the cells distribution array of a **distributed** zone

    Args:
      zone_node (CGNSDistTree): Input Zone_t node
    Returns:
      NDArray : distribution array for cells
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[77, 60, 0]])
      >>> MT.new_Distribution({'Cell' : [45, 60, 60]}, parent=zone)
      >>> MT.Zone.cell_distribution(zone)
      array([45, 60, 60], dtype=int32)
    """
    return PT.get_np_value(search.find_Distribution(zone_node, 'Cell'))

  @staticmethod
  def cell_globalnumbering(zone_node:CGNSTree) -> NDArray:
    """ Return the cells absolute numbering array of a **partitioned** zone

    Args:
      zone_node (CGNSPartTree): Input Zone_t node
    Returns:
      NDArray : global numbering array for cells
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[8, 3, 0]])
      >>> MT.new_GlobalNumbering({'Cell' : [5,1,9]}, parent=zone)
      >>> MT.Zone.cell_globalnumbering(zone)
      array([5, 1, 9], dtype=int32)
    """
    return PT.get_np_value(search.find_GlobalNumbering(zone_node, 'Cell'))
  @staticmethod
  def vtx_globalnumbering(zone_node:CGNSTree) -> NDArray:
    """ Return the vertices absolute numbering array of a **partitioned** zone

    Args:
      zone_node (CGNSPartTree): Input Zone_t node
    Returns:
      NDArray : global numbering array for vertices
    Example:
      >>> zone = PT.new_Zone(type='Unstructured', size=[[8, 3, 0]])
      >>> MT.new_GlobalNumbering({'Vertex' : [9,11,13,14,16,10,12,15]}, parent=zone)
      >>> MT.Zone.vtx_globalnumbering(zone)
      array([ 9, 11, 13, 14, 16, 10, 12, 15], dtype=int32)
    """
    return PT.get_np_value(search.find_GlobalNumbering(zone_node, 'Vertex'))

  @staticmethod
  def EdgeNode(zone_node:CGNSTree) -> CGNSTree:
    """Return the Elements_t node of kind ``BAR_2`` of a Zone_t node
    
    This function aims to be the counterpart of :func:`~maia.pytree.Zone.NGonNode`
    for 2D polyedric zones.
    
    Args:
      zone_node (CGNSTree): Input Zone_t node
    Returns:
      CGNSTree : BAR_2 node
    Raises:
      RuntimeError: if not exactly one ``BAR_2`` element node exists in zone
    """
    #TODO :: maybe in pytree directly ?
    is_edge = lambda n : PT.get_label(n) == 'Elements_t' and PT.Element.Type(n) == 'BAR_2'
    edge_elts_nodes = PT.get_children_from_predicate(zone_node, is_edge)
    if len(edge_elts_nodes) != 1:
      raise RuntimeError("Exactly one EdgeElements_t node must be defined")
    return edge_elts_nodes[0]

class Element:
  """ The following functions apply to Elements_t nodes """

  @staticmethod
  def distribution(elt_node:CGNSTree) -> NDArray:
    """ Return the distribution array of a **distributed** element section

    Args:
      elt_node (CGNSTree): Input Elements_t node, distributed
    Returns:
      NDArray : distribution array
    Example:
      >>> elt = PT.new_Elements(type='TRI_3', erange=[1,10], econn=[1,5,4, 2,8,12])
      >>> MT.new_Distribution({'Element' : [4, 6, 10]}, parent=elt)
      >>> MT.Element.distribution(elt)
      array([ 4,  6, 10], dtype=int32)
    """
    return PT.get_np_value(search.find_Distribution(elt_node, 'Element'))
  @staticmethod
  def globalnumbering(elt_node:CGNSTree) -> NDArray:
    """ Return the absolute numbering array of a **partitioned** element section

    Args:
      elt_node (CGNSTree): Input Elements_t node, partitioned
    Returns:
      NDArray : global numbering array
    Example:
      >>> elt = PT.new_Elements(type='TRI_3', erange=[1,3])
      >>> MT.new_GlobalNumbering({'Element' : [4, 9, 3]}, parent=elt)
      >>> MT.Element.globalnumbering(elt)
      array([4, 9, 3], dtype=int32)
    """
    return PT.get_np_value(search.find_GlobalNumbering(elt_node, 'Element'))


  @staticmethod
  def dn_elt(elt_node:CGNSTree) -> int:
    """ Return the local number of elements of a **distributed** element section

    Args:
      elt_node (CGNSTree): Input Elements_t node, distributed
    Returns:
      int : local number of elements
    Example:
      >>> elt = PT.new_Elements(type='TRI_3', erange=[1,10], econn=[1,5,4, 2,8,12])
      >>> MT.new_Distribution({'Element' : [4, 6, 10]}, parent=elt)
      >>> MT.Element.dn_elt(elt)
      2
    """
    distri = Element.distribution(elt_node)
    return int(distri[1] - distri[0])

  @staticmethod
  def pn_elt(elt_node:CGNSTree) -> int:
    """ Return the local number of elements of a **partitioned** element section

    Args:
      elt_node (CGNSTree): Input Elements_t node, partitioned
    Returns:
      int : local number of elements
    Example:
      >>> elt = PT.new_Elements(type='TRI_3', erange=[1,3])
      >>> MT.new_GlobalNumbering({'Element' : [4, 9, 3]}, parent=elt)
      >>> MT.Element.pn_elt(elt)
      3
    """
    
    # Return the local number of elements (only for partitioned zones)
    gnum = PT.get_np_value(search.find_GlobalNumbering(elt_node, 'Element'))
    return gnum.size

  @staticmethod
  def n_elt(elt_node:Union[CGNSTree, List[CGNSTree]], comm:Optional[MPIComm]=None) -> int:
    """ Return the **total** number of elements of a element section.

    The input Elements_t can be either distributed (a single node is expected) or partitioned
    (the whole list of related nodes is expected).

    Args:
      elt_node (CGNSTree or List[CGNSTree]): Input Elements_t node(s)
      comm (MPIComm): MPI communicator, mandatory for partitioned elements
    Returns:
      int : total number of elements
    Examples:
      >>> tris = [PT.new_Elements(type='TRI_3', erange=[1,3]),
      ...         PT.new_Elements(type='TRI_3', erange=[1,4])]
      >>> MT.new_GlobalNumbering({'Element' : [1,5,4]}, parent=tris[0])
      >>> MT.new_GlobalNumbering({'Element' : [4,3,5,2]}, parent=tris[1])
      >>> MT.Element.n_elt(tris, comm)
      5
    """
    return _n_entity(elt_node, comm, 'Element')

  @staticmethod
  def connectivity(elt_node:CGNSTree) -> vs.VStrideArray:  
    """ Return a :class:`~maia.utils.ndarray.vstride.VStrideArray` describing
    the connectivity of the provided Elements_t node

    For distributed polyedric or mixed sections, the ElementStartOffset is
    automatticaly shifted to obtain the ``displs`` array.

    Args:
      elt_node (CGNSTree): Input Elements_t node
    Returns:
      VSrideArray : element connectivity
    Example:
      >>> ng = PT.new_NGonElements(eso=[0,3,7,10], ec=[1,5,4, 2,3,6,5, 4,9,8])
      >>> MT.Element.connectivity(ng)
      vsarray([
        [1, 5, 4],
        [2, 3, 6, 5],
        [4, 9, 8],
      ], dtype=int32)
    """
    eso = PT.get_child_from_name(elt_node, 'ElementStartOffset')
    ec  = PT.find_child_from_name(elt_node, 'ElementConnectivity')
    assert ec[1] is not None

    is_distri = search.get_Distribution(elt_node) is not None

    if eso is not None:
      assert eso is not None and eso[1] is not None
      eso_val = eso[1] - eso[1][0] if is_distri else eso[1]
      return vs.from_displs(eso_val, ec[1])
    else:
      assert PT.Element.Type(elt_node) not in ['NGON_n', 'NFACE_n', 'MIXED']
      counts = PT.Element.NVtx(elt_node)
      return vs.from_counts(ec[1].dtype.type(counts), ec[1])
      
class Subset:
  """ A subset is a node defining a subregion of the mesh through a PointList
  or a PointRange node (eg BC_t, some ZoneSubRegion_t, …). """

  @staticmethod
  def distribution(subset_node:CGNSTree) -> NDArray:
    """ Return the distribution array of a **distributed** subset

    Args:
      subset_node (CGNSTree): Input subset node, distributed
    Returns:
      NDArray : distribution array
    Example:
      >>> bc = PT.new_BC(point_list=[[23, 55, 42, 13, 56]])
      >>> MT.new_Distribution({'Index' : [5,10,20]}, parent=bc)
      >>> MT.Subset.distribution(bc)
      array([ 5, 10, 20], dtype=int32)
    """
    return PT.get_np_value(search.find_Distribution(subset_node, 'Index'))

  @staticmethod
  def globalnumbering(subset_node:CGNSTree) -> NDArray:
    """ Return the absolute numbering of a **partitioned** subset

    Args:
      subset_node (CGNSTree): Input subset node, partitioned
    Returns:
      NDArray : global numbering array
    Example:
      >>> zsr = PT.new_ZoneSubRegion(point_list=[[4,6,2,8]])
      >>> MT.new_GlobalNumbering({'Index' : [9,11,13,14]}, parent=zsr)
      >>> MT.Subset.globalnumbering(zsr)
      array([ 9, 11, 13, 14], dtype=int32)
    """
    return PT.get_np_value(search.find_GlobalNumbering(subset_node, 'Index'))


  @staticmethod
  def dn_elem(subset_node:CGNSTree) -> int:
    """ Return the local number of entities of a **distributed** subset

    Args:
      subset_node (CGNSTree): Input subset node, distributed
    Returns:
      int : local number of entities
    Example:
      >>> bc = PT.new_BC(point_list=[[23, 55, 42, 13, 56]])
      >>> MT.new_Distribution({'Index' : [5,10,20]}, parent=bc)
      >>> MT.Subset.dn_elem(bc)
      5
    """
    distri = Subset.distribution(subset_node)
    return int(distri[1] - distri[0])

  @staticmethod
  def pn_elem(subset_node:CGNSTree) -> int:
    """ Return the local number of entities of a **partitioned** subset

    Args:
      subset_node (CGNSTree): Input subset node, partitioned
    Returns:
      int : local number of entities
    Example:
      >>> zsr = PT.new_ZoneSubRegion(point_list=[[4,6,2,8]])
      >>> MT.new_GlobalNumbering({'Index' : [9,11,13,14]}, parent=zsr)
      >>> MT.Subset.pn_elem(zsr)
      4
    """
    # Return the local number of indices (only for partitioned subsets)
    # Use PT.Subset to deal missing gnum arrays
    return PT.Subset.n_elem(subset_node)

  @staticmethod
  def n_elem(subset_node:Union[List[CGNSTree], CGNSTree], comm:Optional[MPIComm]=None) -> int:
    """ Return the **total** number of entities of a subset.

    The input subset can be either distributed (a single node is expected) or partitioned
    (the whole list of related nodes is expected).

    Args:
      subset_node (CGNSTree or List[CGNSTree]): Input subset node(s)
      comm (MPIComm): MPI communicator, mandatory for partitioned zones
    Returns:
      int : total number of entities
    Examples:
      >>> bcs = [PT.new_BC(point_list=[[1,4,7,10]]),
      ...        PT.new_BC(point_list=[[2,4,6,8]])]
      >>> MT.new_GlobalNumbering({'Index' : [1,3,5,7]},  parent=bcs[0])
      >>> MT.new_GlobalNumbering({'Index' : [2,4,6,8]}, parent=bcs[1])
      >>> MT.Subset.n_elem(bcs, comm)
      8
    """
    # Special case: for partitioned PointRange (S meshes), gnum array is
    # not always created -> we can not process
    if not is_single_node(subset_node):
      # Partitioned case
      assert comm is not None
      nodes = typing.cast(List[CGNSTree], subset_node)
      if not par_utils.exists_everywhere(nodes, f'{GLBNUM_NAME}/Index', comm):
        raise RuntimeError("GlobalNumbering nodes are mandatory to retrieve initial n_elem")

    # Fallback to standard case
    return _n_entity(subset_node, comm, 'Index')

class Container:
  """ A container is node designed to store fields, such as FlowSolution_t, ZoneSubRegion_t, ...  """

  @staticmethod
  def distribution(cnt_node:CGNSTree, parent_node:Optional[CGNSTree]=None) -> NDArray:
    
    # Simplest case : container has its own distribution
    if PT.Container._is_subset(cnt_node):
      return Subset.distribution(cnt_node)

    assert parent_node is not None, f"parent_node is mandatory for related container node"

    # Container is a related ZSR or BCDS
    if PT.get_label(cnt_node) in ['ZoneSubRegion_t', 'BCDataSet_t']:
      subset = PT.Container.SubsetNode(cnt_node, parent_node)
      return Subset.distribution(subset)

    # Container is a full FSLike (normally CellCenter or Vertex)
    loc = PT.Container.GridLocation(cnt_node)
    fn = {'Vertex' : Zone.vtx_distribution, 'CellCenter' : Zone.cell_distribution}[loc]
    return fn(parent_node)

  @staticmethod
  def globalnumbering(cnt_node:CGNSTree, parent_node:Optional[CGNSTree]=None) -> NDArray:
    
    # Simplest case : container has its own distribution
    if PT.Container._is_subset(cnt_node):
      return Subset.globalnumbering(cnt_node)

    assert parent_node is not None, f"parent_node is mandatory for related container node"

    # Container is a related ZSR or BCDS
    if PT.get_label(cnt_node) in ['ZoneSubRegion_t', 'BCDataSet_t']:
      subset = PT.Container.SubsetNode(cnt_node, parent_node)
      return Subset.globalnumbering(subset)

    # Container is a full FSLike (normally CellCenter or Vertex)
    loc = PT.Container.GridLocation(cnt_node)
    fn = {'Vertex' : Zone.vtx_globalnumbering, 'CellCenter' : Zone.cell_globalnumbering}[loc]
    return fn(parent_node)
