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
