import numpy as np

import maia.pytree as PT
from   maia.utils  import np_utils

import cmaia.part_algo as cpart_algo

@PT.check_is_label("Zone_t")
def compute_cell_center(zone):
  """Compute the cell centers of a partitioned zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the cells.

  Args:
    zone (CGNSTree): Partitionned Structured or U-NGon CGNS Zone
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
    ngons  = [e for e in PT.iter_children_from_label(zone, 'Elements_t') if PT.Element.CGNSName(e) == 'NGON_n']
    if len(ngons) != 1:
      raise NotImplementedError(f"Cell center computation is only available for NGON connectivity")
    face_vtx_idx, face_vtx, ngon_pe = PT.Zone.ngon_connectivity(zone)
    center_cell = cpart_algo.compute_center_cell_u(n_cell, cx, cy, cz, face_vtx, face_vtx_idx, ngon_pe)
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
    face_vtx_n   = np.diff(face_vtx_idx)
    n_face = face_vtx_idx.size - 1

    face_center_x = np.add.reduceat(cx[face_vtx-1], face_vtx_idx[:-1]) / face_vtx_n
    face_center_y = np.add.reduceat(cy[face_vtx-1], face_vtx_idx[:-1]) / face_vtx_n
    face_center_z = np.add.reduceat(cz[face_vtx-1], face_vtx_idx[:-1]) / face_vtx_n
    
    face_center = np_utils.interweave_arrays([face_center_x, face_center_y, face_center_z])
    return face_center
  else:
    raise NotImplementedError("Only U/NGon zones are managed")
