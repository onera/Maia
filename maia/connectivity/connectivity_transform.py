import numpy as np

from maia.sids import sids
from maia.sids import pytree as PT
from cmaia.connectivity.connectivity_transform import *

def get_ngon_pe_local(ngon_node):
  """
  Shift the ParentElement array of an NGonNode to have local (starting at 1) cell
  indices.
  If PE array was already local, no copy is done
  """
  assert sids.ElementCGNSName(ngon_node) == 'NGON_n'
  pe_n = PT.request_child_from_name(ngon_node, "ParentElements")
  if pe_n is None:
    raise RuntimeError(f"ParentElements node not found on ngon node {ngon_node[0]}")
  pe_val = pe_n[1]
  if pe_val.size == 0:
    return pe_val
  else:
    first_cell = np.max(pe_val[1]) #Get any cell and use it to check if offset is necessary
    if first_cell > sids.ElementRange(ngon_node)[1]:
      return pe_val - sids.ElementRange(ngon_node)[1] * (pe_val > 0)
    else:
      return pe_val

