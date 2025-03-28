import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia.typing import CGNSTree, MPIComm, Iterator, Optional, Any
import numpy as np

from .dist import ngon_tools as dist_ngon_tools
from .part import ngon_tools as part_ngon_tools


is_poly_3d_zone = lambda z: PT.Zone.CellDimension(z) == 3 and PT.Zone.has_ngon_elements(z)
is_poly_2d_zone = lambda z: PT.Zone.CellDimension(z) == 2 and \
                            PT.Zone.Type(z) == 'Unstructured' and \
                            all(PT.Element.CGNSName(e) in ['BAR_2', 'NGON_n'] for e in PT.get_children_from_label(z, 'Elements_t'))

def iter_matching_zones(t: CGNSTree, cond: Any) -> Iterator[CGNSTree]:
  for z in PT.iter_all_Zone_t(t):
    if cond(z):
      yield z

def get_pe_local(node: CGNSTree) -> np.ndarray:
  """
  Shift the ParentElement array of a NGON or Edge node to have local (starting at 1)
  indices.
  If PE array was already local, no copy is done
  """
  assert PT.Element.CGNSName(node) in ['BAR_2', 'NGON_n']
  pe_n = PT.get_child_from_name(node, "ParentElements")
  if pe_n is None:
    raise RuntimeError(f"ParentElements node not found on node {node[0]}")
  pe_val = pe_n[1]
  if pe_val.size == 0:
    return pe_val
  else:
    first_parent = pe_val[1].max() #Get any parent and use it to check if offset is necessary
    if first_parent > PT.Element.Range(node)[1]:
      return pe_val - PT.Element.Range(node)[1] * (pe_val > 0)
    else:
      return pe_val

def pe_to_nface(t: CGNSTree, 
                comm: Optional[MPIComm] = None, 
                removePE: bool = False) -> None:
  """Create a NFace node from a NGon node with ParentElements.

  Input tree is modified inplace.

  Args:
    t           (CGNSTree): Distributed or Partitioned tree starting at Zone_t level or higher.
    comm       (MPIComm) : MPI communicator, mandatory only for distributed zones
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #pe_to_nface@start
        :end-before: #pe_to_nface@end
        :dedent: 2
  """
  predicate = lambda z: is_poly_3d_zone(z) and not PT.Zone.has_nface_elements(z)
  for zone in iter_matching_zones(t, predicate):
    if MT.getDistribution(zone) is not None:
      assert comm is not None
      dist_ngon_tools.pe_to_nface(zone, comm, removePE)
    else:
      part_ngon_tools.pe_to_nface(zone, removePE)


def nface_to_pe(t: CGNSTree, 
                comm: Optional[MPIComm] = None, 
                removeNFace: bool = False) -> None:
  """Create a ParentElements node in the NGon node from a NFace node.

  Input tree is modified inplace.

  Args:
    t           (CGNSTree): Distributed or Partitioned tree starting at Zone_t level or higher.
    comm        (MPIComm) : MPI communicator, mandatory only for distributed zones
    removeNFace (bool, optional): If True, remove the NFace node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #nface_to_pe@start
        :end-before: #nface_to_pe@end
        :dedent: 2
  """
  predicate = lambda z: is_poly_3d_zone(z) and PT.get_child_from_predicates(z, 'Elements_t/ParentElements') is None
  for zone in iter_matching_zones(t, predicate):
    if PT.maia.getDistribution(zone) is not None:
      assert comm is not None
      dist_ngon_tools.nface_to_pe(zone, comm, removeNFace)
    else:
      part_ngon_tools.nface_to_pe(zone, removeNFace)


def edge_pe_to_ngon(t: CGNSTree, 
                    comm: MPIComm, 
                    removePE: bool = False) -> None:
  """Create a NGon node from a Edge node with ParentElements.

  Input tree is modified inplace.

  Args:
    t           (CGNSTree): Distributed or Partitioned tree starting at Zone_t level or higher.
    comm       (MPIComm) : MPI communicator, mandatory only for distributed zones
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #edge_pe_to_ngon@start
        :end-before: #edge_pe_to_ngon@end
        :dedent: 2
  """
  predicate = lambda z: is_poly_2d_zone(z) and not PT.Zone.has_ngon_elements(z)
  for zone in iter_matching_zones(t, predicate):
    if PT.maia.getDistribution(zone) is not None:
      assert comm is not None
      dist_ngon_tools.edge_pe_to_ngon(zone, comm, removePE)
    else:
      part_ngon_tools.edge_pe_to_ngon(zone, removePE)

def ngon_to_edge_pe(t: CGNSTree, 
                    comm: MPIComm, 
                    remove_NGon: Optional[bool] = False) -> None:
  """Create a ParentElements node in the EdgeElements node from a NGon node.

  Note that EdgeElement is supposed to exist and define all (including internal)
  edges. This function retrieves the link between these edges and the NGon node.

  Input tree is modified inplace.

  Args:
    t           (CGNSTree): Distributed or Partitioned tree starting at Zone_t level or higher.
    comm        (MPIComm) : MPI communicator, mandatory only for distributed zones
    removeNFace (bool, optional): If True, remove the NGon node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #ngon_to_edge_pe@start
        :end-before: #ngon_to_edge_pe@end
        :dedent: 2
  """
  predicate = lambda z: is_poly_2d_zone(z) and PT.get_child_from_predicates(z, 'Elements_t/ParentElements') is None
  for zone in iter_matching_zones(t, predicate):
    if PT.maia.getDistribution(zone) is not None:
      assert comm is not None
      dist_ngon_tools.ngon_to_edge_pe(zone, comm, remove_NGon)
    else:
      part_ngon_tools.ngon_to_edge_pe(zone, remove_NGon)