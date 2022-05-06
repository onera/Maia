import numpy as np

import maia.pytree as PT
from cmaia.part_algo import enforce_pe_left_parent

def enforce_boundary_pe_left(zone_node):
  """
  Force the boundary ngon to have a non zero left parent cell.
  In such case, connetivities (FaceVtx & NFace) are reversed to preserve face
  orientation
  """
  ngon  = PT.Zone.NGonNode(zone_node)
  try:
    nface = PT.Zone.NFaceNode(zone_node)
    enforce_pe_left_parent(PT.get_child_from_name(ngon, 'ElementStartOffset')[1],
                           PT.get_child_from_name(ngon, 'ElementConnectivity')[1],
                           PT.get_child_from_name(ngon, 'ParentElements')[1],
                           PT.get_child_from_name(nface, 'ElementStartOffset')[1],
                           PT.get_child_from_name(nface, 'ElementConnectivity')[1])
  except RuntimeError: #No NFace
    enforce_pe_left_parent(PT.get_child_from_name(ngon, 'ElementStartOffset')[1],
                           PT.get_child_from_name(ngon, 'ElementConnectivity')[1],
                           PT.get_child_from_name(ngon, 'ParentElements')[1])

