import numpy as np

import maia.pytree        as PT
import maia.pytree.sids   as sids
from maia.algo.apply_function_to_nodes import zones_iterator

def get_pe_local(node):
  """
  Shift the ParentElement array of a NGON or Edge node to have local (starting at 1)
  indices.
  If PE array was already local, no copy is done
  """
  assert sids.Element.CGNSName(node) in ['BAR_2', 'NGON_n']
  pe_n = PT.get_child_from_name(node, "ParentElements")
  if pe_n is None:
    raise RuntimeError(f"ParentElements node not found on node {node[0]}")
  pe_val = pe_n[1]
  if pe_val.size == 0:
    return pe_val
  else:
    first_parent = np.max(pe_val[1]) #Get any parent and use it to check if offset is necessary
    if first_parent > sids.Element.Range(node)[1]:
      return pe_val - sids.Element.Range(node)[1] * (pe_val > 0)
    else:
      return pe_val

def get_ngon_pe_local(ngon_node):
  raise NotImplementedError("This function has been removed in favor of indexing.get_pe_local")

def pe_to_nface(t, comm=None, removePE=False):
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
  for zone in zones_iterator(t):
    if PT.maia.getDistribution(zone) is not None:
      assert comm is not None
      from .dist.ngon_tools import pe_to_nface
      pe_to_nface(zone, comm, removePE)
    else:
      from .part.ngon_tools import pe_to_nface
      pe_to_nface(zone, removePE)


def nface_to_pe(t, comm=None, removeNFace=False):
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
  for zone in zones_iterator(t):
    if PT.maia.getDistribution(zone) is not None:
      assert comm is not None
      from .dist.ngon_tools import nface_to_pe
      nface_to_pe(zone, comm, removeNFace)
    else:
      from .part.ngon_tools import nface_to_pe
      nface_to_pe(zone, removeNFace)


def edge_pe_to_ngon(t, comm=None, removePE=False):
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
  for zone in zones_iterator(t):
    if PT.maia.getDistribution(zone) is not None:
      assert comm is not None
      from .dist.ngon_tools import edge_pe_to_ngon
      edge_pe_to_ngon(zone, comm, removePE)
    else:
      from .part.ngon_tools import edge_pe_to_ngon
      edge_pe_to_ngon(zone, removePE)

def ngon_to_edge_pe(t, comm, remove_NGon=False):
  """Create a ParentElements node in the EdgeElements node from a NGon node.

  Note that EdgeElement is supposed to exist and define all (including internal)
  edges. This function retrieves the link between these edges and the NGon node.

  Input tree is modified inplace.

  Args:
    t           (CGNSTree(s)): Distributed or Partitioned tree (or sequences of)
      starting at Zone_t level or higher.
    comm        (MPIComm) : MPI communicator, mandatory only for distributed zones
    removeNFace (bool, optional): If True, remove the NGon node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #ngon_to_edge_pe@start
        :end-before: #ngon_to_edge_pe@end
        :dedent: 2
  """
  for zone in zones_iterator(t):
    if PT.maia.getDistribution(zone) is not None:
      assert comm is not None
      from .dist.ngon_tools import ngon_to_edge_pe
      ngon_to_edge_pe(zone, comm, remove_NGon)
    else:
      from .part.ngon_tools import ngon_to_edge_pe
      ngon_to_edge_pe(zone, remove_NGon)