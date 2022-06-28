import numpy as np

import Pypdm.Pypdm as PDM
import Converter.Internal as I

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo import indexing
from maia.utils import par_utils

import cmaia.part_algo as cpart_algo

def pe_to_nface(zone, remove_PE=False):
  """Create a NFace node from a NGon node with ParentElements.

  Input tree is modified inplace.

  Args:
    zone       (CGNSTree): Partitioned zone
    remove_PE  (bool, optional): If True, remove the ParentElements node.
      Defaults to False.
  """
  ngon_node = PT.Zone.NGonNode(zone)
  er = I.getNodeFromName1(ngon_node, 'ElementRange')[1]
  pe = I.getNodeFromName1(ngon_node, 'ParentElements')[1]
  max_cell = np.max(pe)
  min_cell = np.min(pe[np.nonzero(pe)])
  
  local_pe = indexing.get_ngon_pe_local(ngon_node)

  nface_eso, nface_ec = cpart_algo.local_pe_to_local_cellface(local_pe) #Compute NFace connectivity

  #Put NFace/EC in global numbering (to refer to ngon global ids)
  first_ngon = er[0]
  if first_ngon != 1:
    nface_ec_sign = np.sign(nface_ec)
    nface_ec_sign*(np.abs(nface_ec) + first_ngon - 1)

  #Create NFace node
  nface = I.newElements('NFaceElements', 'NFACE', parent=zone)
  I.newPointRange("ElementRange", np.array([min_cell, max_cell], dtype=np.int32), parent=nface)
  I.newDataArray("ElementStartOffset",   nface_eso, parent=nface)
  I.newDataArray("ElementConnectivity",  nface_ec,  parent=nface)

  if remove_PE:
    I._rmNodesByName(ngon_node, "ParentElements")


