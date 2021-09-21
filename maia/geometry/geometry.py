from cmaia.geometry.geometry import *

import numpy as np
import Converter.Internal       as I
import maia.sids.Internal_ext   as IE
import maia.sids.sids           as sids

@IE.check_is_label("Zone_t")
def compute_cell_center(zone):
  """
  Compute the cell centers of a NGon unstructured zone or a structured zone
  and return it as a flat (interlaced) np array. Centers are computed using
  a basic average over the vertices of the cells
  """
  cx, cy, cz = sids.coordinates(zone)

  if sids.Zone.Type(zone) == "Unstructured":
    n_cell     = sids.Zone.n_cell(zone)
    ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if sids.ElementCGNSName(e) == 'NGON_n']
    if len(ngons) != 1:
      raise NotImplementedError(f"Cell center computation is only available for NGON connectivity")
    face_vtx, face_vtx_idx, ngon_pe = sids.ngon_connectivity(zone)
    center_cell = compute_center_cell_u(n_cell, cx, cy, cz, face_vtx, face_vtx_idx, ngon_pe)
  else:
    center_cell = compute_center_cell_s(*sids.Zone.CellSize(zone), cx, cy, cz)

  return center_cell
