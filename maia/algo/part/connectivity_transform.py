import numpy as np

import maia.pytree as PT
from cmaia.part_algo import enforce_pe_left_parent

def enforce_boundary_pe_left(zone_node):
  """
  Force the boundary ngon to have a non zero left parent cell.
  In such case, connetivities (FaceVtx & NFace) are reversed to preserve face
  orientation
  """
  if not PT.Zone.has_ngon_elements(zone_node): #Early return if zone is Elts defined
    return

  ngon  = PT.Zone.NGonNode(zone_node)
  z_dim = 2
  if PT.Zone.has_nface_elements(zone_node) or PT.get_child_from_name(ngon, "ParentElements") is not None:
    z_dim = 3
  
  if z_dim == 3:
    try:
      nface = PT.Zone.NFaceNode(zone_node)
      enforce_pe_left_parent(PT.get_child_from_name(ngon, 'ElementStartOffset')[1],
                             PT.get_child_from_name(ngon, 'ElementConnectivity')[1],
                             PT.get_child_from_name(ngon, 'ParentElements')[1],
                             PT.get_child_from_name(nface, 'ElementStartOffset')[1],
                             PT.get_child_from_name(nface, 'ElementConnectivity')[1])
    except RuntimeError: #3D, but no NFace
      enforce_pe_left_parent(PT.get_child_from_name(ngon, 'ElementStartOffset')[1],
                             PT.get_child_from_name(ngon, 'ElementConnectivity')[1],
                             PT.get_child_from_name(ngon, 'ParentElements')[1])

  elif z_dim == 2:
    bar_elts = [e for e in PT.iter_children_from_label(zone_node, 'Elements_t') if PT.Element.CGNSName(e) == 'BAR_2']
    if len(bar_elts) > 1:
      raise RuntimeError("Multiple BAR elements are not managed")
    elif len(bar_elts) == 1:
      nedge = bar_elts[0]
      nedge_eso = 2*np.arange(PT.Element.Size(nedge)+1, dtype=np.int32)
      enforce_pe_left_parent(nedge_eso,
                             PT.get_child_from_name(nedge, 'ElementConnectivity')[1],
                             PT.get_child_from_name(nedge, 'ParentElements')[1])


