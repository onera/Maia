import numpy as np

import maia.pytree as PT
from   maia.algo.part import connectivity_utils as CU
from   maia.utils     import np_utils

import cmaia.part_algo as cpart_algo

def _mean_coords_from_connectivity(vtx_id_idx, vtx_id, cx, cy, cz):

  vtx_id_n = np.diff(vtx_id_idx)

  mean_x = np.add.reduceat(cx[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n
  mean_y = np.add.reduceat(cy[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n
  mean_z = np.add.reduceat(cz[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n

  return np_utils.interweave_arrays([mean_x, mean_y, mean_z])

@PT.check_is_label("Zone_t")
def compute_cell_center(zone):
  """Compute the cell centers of a partitioned zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the cells.

  Args:
    zone (CGNSTree): Partitionned CGNS Zone
  Returns:
    cell_center (array): Flat (interlaced) numpy array of cell centers

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_cell_center@start
        :end-before: #compute_cell_center@end
        :dedent: 2
  """
  cx, cy, cz = PT.Zone.coordinates(zone)

  if PT.Zone.Type(zone) == "Unstructured":
    n_cell     = PT.Zone.n_cell(zone)
    if PT.Zone.has_ngon_elements(zone):
      face_vtx_idx, face_vtx, ngon_pe = PT.Zone.ngon_connectivity(zone)
      center_cell = cpart_algo.compute_center_cell_u(n_cell, cx, cy, cz, face_vtx, face_vtx_idx, ngon_pe)
    else:
      cell_vtx_idx, cell_vtx = CU.cell_vtx_connectivity(zone)
      center_cell = _mean_coords_from_connectivity(cell_vtx_idx, cell_vtx, cx, cy, cz)
  else:
    center_cell = cpart_algo.compute_center_cell_s(*PT.Zone.CellSize(zone), cx, cy, cz)

  return center_cell

@PT.check_is_label("Zone_t")
def compute_face_center(zone):
  """Compute the face centers of a partitioned zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node, and a unstructured-NGon connectivity.

  Centers are computed using a basic average over the vertices of the faces.

  Args:
    zone (CGNSTree): Partitionned 2D or 3D U-NGon CGNS Zone
  Returns:
    face_center (array): Flat (interlaced) numpy array of face centers

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_face_center@start
        :end-before: #compute_face_center@end
        :dedent: 2
  """
  cx, cy, cz = PT.Zone.coordinates(zone)

  if PT.Zone.Type(zone) == "Unstructured":
    ngon_node = PT.Zone.NGonNode(zone)
    face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
    face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    return _mean_coords_from_connectivity(face_vtx_idx, face_vtx, cx, cy, cz)
  else:
    raise NotImplementedError("Only U/NGon zones are managed")
