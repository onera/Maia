import cmaia.part_algo as cpart_algo

import Converter.Internal as I
import maia.pytree        as PT

@PT.check_is_label("Zone_t")
def compute_cell_center(zone):
  """Compute the cell centers of a partitioned zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the cells.

  Args:
    zone (CGNSZone): Partitionned Structured or U-NGon CGNS Zone
  Returns:
    center_cell (array): Flat (interlaced) numpy array of cell centers

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_cell_center@start
        :end-before: #compute_cell_center@end
        :dedent: 2
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
