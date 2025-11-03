import numpy as np

from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT

from .dist import ngon_tools as dist_ngon_tools
from .part import ngon_tools as part_ngon_tools

from maia.pytree.pred import NodePredicate
HAS_PE = NodePredicate(lambda z : PT.get_child_from_predicates(z, 'Elements_t/ParentElements') is not None)
HAS_NGON = NodePredicate(PT.Zone.has_ngon_elements)
HAS_NFACE = NodePredicate(PT.Zone.has_nface_elements)

def iter_matching_zones(t: CGNSTree, cond: Callable[[CGNSTree], bool]) -> Iterator[CGNSTree]:
  for z in PT.iter_all_Zone_t(t):
    if cond(z):
      yield z

def get_pe_local(node: CGNSTree) -> NDArray:
  """
  Shift the ParentElement array of a NGON or Edge node to have local (starting at 1)
  indices.
  If PE array was already local, no copy is done
  """
  assert PT.Element.Type(node) in ['BAR_2', 'NGON_n']
  pe_n = PT.get_child_from_name(node, "ParentElements")
  if pe_n is None:
    raise RuntimeError(f"ParentElements node not found on node {node[0]}")
  pe_val = PT.get_np_value(pe_n)
  if pe_val.size == 0:
    return pe_val
  else:
    first_parent = pe_val[0].max() #Get any parent and use it to check if offset is necessary
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
    t          (CGNSTree): Distributed, Partitioned or Full tree starting at Zone_t level or higher.
    comm       (MPIComm) : MPI communicator, mandatory only for distributed zones
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #pe_to_nface@start
        :end-before: #pe_to_nface@end
        :dedent: 2
  """
  for zone in iter_matching_zones(t, PT.pred.IS_POLY3D_ZONE & ~HAS_NFACE):
    if MT.get_Distribution(zone) is not None:
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
    t           (CGNSTree): Distributed, Partitioned or Full tree starting at Zone_t level or higher.
    comm        (MPIComm) : MPI communicator, mandatory only for distributed zones
    removeNFace (bool, optional): If True, remove the NFace node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #nface_to_pe@start
        :end-before: #nface_to_pe@end
        :dedent: 2
  """
  for zone in iter_matching_zones(t, PT.pred.IS_POLY3D_ZONE & ~HAS_PE):
    if MT.get_Distribution(zone) is not None:
      assert comm is not None
      dist_ngon_tools.nface_to_pe(zone, comm, removeNFace)
    else:
      part_ngon_tools.nface_to_pe(zone, removeNFace)


def edge_pe_to_ngon(t: CGNSTree,
                    comm: Optional[MPIComm], 
                    removePE: bool = False) -> None:
  """Create a NGon node from a Edge node with ParentElements.

  Input tree is modified inplace.

  Args:
    t          (CGNSTree): Distributed, Partitioned or Full tree starting at Zone_t level or higher.
    comm       (MPIComm) : MPI communicator, mandatory only for distributed zones
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #edge_pe_to_ngon@start
        :end-before: #edge_pe_to_ngon@end
        :dedent: 2
  """
  for zone in iter_matching_zones(t, PT.pred.IS_POLY2D_ZONE & ~HAS_NGON):
    if MT.get_Distribution(zone) is not None:
      assert comm is not None
      dist_ngon_tools.edge_pe_to_ngon(zone, comm, removePE)
    else:
      part_ngon_tools.edge_pe_to_ngon(zone, removePE)

def ngon_to_edge_pe(t: CGNSTree,
                    comm: Optional[MPIComm], 
                    remove_NGon: bool = False) -> None:
  """Create a ParentElements node in the EdgeElements node from a NGon node.

  Note that EdgeElement is supposed to exist and define all (including internal)
  edges. This function retrieves the link between these edges and the NGon node.

  Input tree is modified inplace.

  Args:
    t           (CGNSTree): Distributed, Partitioned or Full tree starting at Zone_t level or higher.
    comm        (MPIComm) : MPI communicator, mandatory only for distributed zones
    removeNFace (bool, optional): If True, remove the NGon node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #ngon_to_edge_pe@start
        :end-before: #ngon_to_edge_pe@end
        :dedent: 2
  """
  for zone in iter_matching_zones(t, PT.pred.IS_POLY2D_ZONE & ~HAS_PE):
    if MT.get_Distribution(zone) is not None:
      assert comm is not None
      dist_ngon_tools.ngon_to_edge_pe(zone, comm, remove_NGon)
    else:
      part_ngon_tools.ngon_to_edge_pe(zone, remove_NGon)