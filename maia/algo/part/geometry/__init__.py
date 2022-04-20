import cmaia.part_algo as cpart_algo

import Converter.Internal as I
import maia.pytree        as PT

@PT.check_is_label("Zone_t")
def compute_cell_center(zone):
  """
  Compute the cell centers of a NGon unstructured zone or a structured zone
  and return it as a flat (interlaced) np array. Centers are computed using
  a basic average over the vertices of the cells
  """
  cx, cy, cz = PT.Zone.coordinates(zone)

  if PT.Zone.Type(zone) == "Unstructured":
    n_cell     = PT.Zone.n_cell(zone)
    ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if PT.Element.CGNSName(e) == 'NGON_n']
    if len(ngons) != 1:
      raise NotImplementedError(f"Cell center computation is only available for NGON connectivity")
    face_vtx_idx, face_vtx, ngon_pe = PT.Zone.ngon_connectivity(zone)
    center_cell = cpart_algo.compute_center_cell_u(n_cell, cx, cy, cz, face_vtx, face_vtx_idx, ngon_pe)
  else:
    center_cell = cpart_algo.compute_center_cell_s(*PT.Zone.CellSize(zone), cx, cy, cz)

  return center_cell
