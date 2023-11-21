import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils    import np_utils
from maia.transfer import protocols as EP

import cmaia.part_algo as cpart_algo

def _mean_coords_from_connectivity(vtx_id_idx, cx_expd, cy_expd, cz_expd):
  """ Coordinates should be repeted to match the size of vtx_id_idx """

  vtx_id_n = np.diff(vtx_id_idx)

  mean_x = np.add.reduceat(cx_expd, vtx_id_idx[:-1]) / vtx_id_n
  mean_y = np.add.reduceat(cy_expd, vtx_id_idx[:-1]) / vtx_id_n
  mean_z = np.add.reduceat(cz_expd, vtx_id_idx[:-1]) / vtx_id_n

  return np_utils.interweave_arrays([mean_x, mean_y, mean_z])

@PT.check_is_label("Zone_t")
def compute_face_normal(zone, comm):
  """Compute the face normal of a distributed zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.

  The normal is outward oriented and its norms equals the area of the faces.

  Args:
    zone (CGNSTree): Distributed 3D or 2D U-NGon CGNS Zone
  Returns:
    face_normal (array): Flat (interlaced) numpy array of face normal

  """
  cx, cy, cz  = PT.Zone.coordinates(zone)
  dist_coords = {'CoordinateX' : cx, 'CoordinateY': cy, 'CoordinateZ': cz}
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      _face_vtx_idx = np.empty(face_vtx_idx.size, np.int32)
      np.subtract(face_vtx_idx, face_vtx_idx[0], out=_face_vtx_idx)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
      part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
      coords = [part_data[f'Coordinate{key}'][0] for key in ['X', 'Y', 'Z']]

      return cpart_algo.compute_face_normal_u(_face_vtx_idx, *coords)
  raise NotImplementedError("Only NGON zones are managed")

@PT.check_is_label("Zone_t")
def compute_face_center(zone, comm):
  """Compute the face center of a distributed zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.

  Centers are computed using a basic average over the vertices of the faces.

  Args:
    zone (CGNSTree): Distributed 3D or 2D U-NGon CGNS Zone
  Returns:
    face_normal (array): Flat (interlaced) numpy array of face centers

  """
  cx, cy, cz  = PT.Zone.coordinates(zone)
  dist_coords = {'CoordinateX' : cx, 'CoordinateY': cy, 'CoordinateZ': cz}
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      _face_vtx_idx = np.empty(face_vtx_idx.size, np.int32)
      np.subtract(face_vtx_idx, face_vtx_idx[0], out=_face_vtx_idx)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
      part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
      coords = [part_data[f'Coordinate{key}'][0] for key in ['X', 'Y', 'Z']]

      return _mean_coords_from_connectivity(_face_vtx_idx, *coords)
  raise NotImplementedError("Only NGON zones are managed")
