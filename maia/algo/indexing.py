import numpy as np

import maia.pytree        as PT
import maia.pytree.sids   as sids

def get_ngon_pe_local(ngon_node):
  """
  Shift the ParentElement array of a NGonNode to have local (starting at 1) cell
  indices.
  If PE array was already local, no copy is done
  """
  assert sids.Element.CGNSName(ngon_node) == 'NGON_n'
  pe_n = PT.request_child_from_name(ngon_node, "ParentElements")
  if pe_n is None:
    raise RuntimeError(f"ParentElements node not found on ngon node {ngon_node[0]}")
  pe_val = pe_n[1]
  if pe_val.size == 0:
    return pe_val
  else:
    first_cell = np.max(pe_val[1]) #Get any cell and use it to check if offset is necessary
    if first_cell > sids.Element.Range(ngon_node)[1]:
      return pe_val - sids.Element.Range(ngon_node)[1] * (pe_val > 0)
    else:
      return pe_val

def pe_to_nface(zone, comm=None, removePE=False):
  """Create a NFace node from a NGon node with ParentElements.

  Input tree is modified inplace.

  Args:
    zone       (CGNSTree): Distributed or Partitioned zone
    comm       (MPIComm) : MPI communicator, mandatory only for distributed zones
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #pe_to_nface@start
        :end-before: #pe_to_nface@end
        :dedent: 2
  """
  if PT.maia.getDistribution(zone) is not None:
    assert comm is not None
    from .dist.ngon_tools import pe_to_nface
    pe_to_nface(zone, comm, removePE)
  else:
    from .part.ngon_tools import pe_to_nface
    pe_to_nface(zone, removePE)

def nface_to_pe(zone, comm=None, removeNFace=False):
  """Create a ParentElements node in the NGon node from a NFace node.

  Input tree is modified inplace.

  Args:
    zone        (CGNSTree): Distributed or Partitioned zone
    comm        (MPIComm) : MPI communicator, mandatory only for distributed zones
    removeNFace (bool, optional): If True, remove the NFace node.
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #nface_to_pe@start
        :end-before: #nface_to_pe@end
        :dedent: 2
  """
  if PT.maia.getDistribution(zone) is not None:
    assert comm is not None
    from .dist.ngon_tools import nface_to_pe
    nface_to_pe(zone, comm, removeNFace)
  else:
    from .part.ngon_tools import nface_to_pe
    nface_to_pe(zone, removeNFace)
