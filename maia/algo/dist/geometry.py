import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP

import cmaia.part_algo as cpart_algo

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
