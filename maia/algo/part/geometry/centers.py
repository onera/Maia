import numpy as np

import maia.pytree as PT
from   maia.algo.part import connectivity_utils as CU
from   maia.utils     import np_utils

import cmaia.part_algo as cpart_algo

def _to_xyz(r, theta, z):
  return r*np.cos(theta), r*np.sin(theta), z
def _to_rthetaz(x, y, z):
  return np.sqrt(x**2+y**2), np.arctan2(y, x), z

def _mean_coords_from_connectivity(vtx_id_idx, vtx_id, cx, cy, cz):

  vtx_id_n = np.diff(vtx_id_idx)

  mean_x = np.add.reduceat(cx[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n
  mean_y = np.add.reduceat(cy[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n
  mean_z = np.add.reduceat(cz[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n

  return np_utils.interweave_arrays([mean_x, mean_y, mean_z])

def _mean_coords_from_connectivity_cyl(vtx_id_idx, vtx_id, cr, ctheta, cz):

  vtx_id_n = np.diff(vtx_id_idx)

  cx,cy,cz = _to_xyz(cr, ctheta, cz)

  mean_x = np.add.reduceat(cx[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n
  mean_y = np.add.reduceat(cy[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n
  mean_z = np.add.reduceat(cz[vtx_id-1], vtx_id_idx[:-1]) / vtx_id_n
  
  return np_utils.interweave_arrays(_to_rthetaz(mean_x, mean_y, mean_z))

def compute_cell_center(zone):
  """Compute the cell centers of a partitioned zone.

  Input zone must have cartesian or cylindrical coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the cells.

  Args:
    zone (CGNSTree): Partitionned CGNS Zone
  Returns:
    array: Flat (interlaced) numpy array of cell centers

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_cell_center@start
        :end-before: #compute_cell_center@end
        :dedent: 2
  """
  coords = PT.Zone.coordinates(zone)

  if PT.Zone.Type(zone) == "Unstructured":
    n_cell     = PT.Zone.n_cell(zone)
    if PT.Zone.has_ngon_elements(zone):
      face_vtx_idx, face_vtx, ngon_pe = PT.Zone.ngon_connectivity(zone)
      if isinstance(coords, PT.CylindricalCoordinates):
        center_cell = cpart_algo.compute_center_cell_u_cyl(n_cell, *coords, face_vtx, face_vtx_idx, ngon_pe)
      elif isinstance(coords, PT.CartesianCoordinates):
        center_cell = cpart_algo.compute_center_cell_u(n_cell, *coords, face_vtx, face_vtx_idx, ngon_pe)
    else:
      cell_vtx_idx, cell_vtx = CU.cell_vtx_connectivity(zone)
      if isinstance(coords, PT.CylindricalCoordinates):
        center_cell = _mean_coords_from_connectivity_cyl(cell_vtx_idx, cell_vtx, *coords)
      elif isinstance(coords, PT.CartesianCoordinates):
        center_cell = _mean_coords_from_connectivity(cell_vtx_idx, cell_vtx, *coords)
  else:
    if isinstance(coords, PT.CylindricalCoordinates):
      center_cell = cpart_algo.compute_center_cell_s_cyl(*PT.Zone.CellSize(zone), *coords)
    elif isinstance(coords, PT.CartesianCoordinates):
      center_cell = cpart_algo.compute_center_cell_s(*PT.Zone.CellSize(zone), *coords)

  return center_cell

def compute_face_center(zone):
  """Compute the face centers of a partitioned zone.

  Input zone must have cartesian or cylindrical coordinates recorded under a unique
  GridCoordinates node.

  Centers are computed using a basic average over the vertices of the faces.

  Note:
    If zone is described with standard elements, centers will be computed for elements
    explicitly defined in cgns tree.

  Args:
    zone (CGNSTree): Partitionned 2D or 3D U CGNS Zone
  Returns:
    array: Flat (interlaced) numpy array of face centers

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_face_center@start
        :end-before: #compute_face_center@end
        :dedent: 2
  """
  coords = PT.Zone.coordinates(zone)

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      face_vtx_idx, face_vtx = CU.cell_vtx_connectivity(zone, dim=2)
    if isinstance(coords, PT.CartesianCoordinates):
      return _mean_coords_from_connectivity(face_vtx_idx, face_vtx, *coords)
    elif isinstance(coords, PT.CylindricalCoordinates):
      return _mean_coords_from_connectivity_cyl(face_vtx_idx, face_vtx, *coords)
  else:
    zone_dim = PT.get_value(zone).shape[0]
    assert zone_dim >= 2, "1d zones are not managed"
    vtx_size = [1,1,1]
    vtx_size[:zone_dim] = PT.Zone.VertexSize(zone)
    # Create cz if zone_dim == 2 & cz is None
    remove_z = False
    _cx = np.atleast_3d(coords[0]) # Auto expand arrays if zone_dim == 2
    _cy = np.atleast_3d(coords[1])
    if zone_dim == 2 and coords[2] is None:
      _cz = np.zeros(vtx_size, dtype=float, order='F')
      remove_z = True
    else:
      _cz = np.atleast_3d(coords[2])
    if isinstance(coords, PT.CartesianCoordinates):
      centers = cpart_algo.compute_center_face_s(*vtx_size, _cx, _cy, _cz)
    elif isinstance(coords, PT.CylindricalCoordinates):
      centers = cpart_algo.compute_center_face_s_cyl(*vtx_size, _cx, _cy, _cz)
    if remove_z:
        centers = np.delete(centers, 3*np.arange(centers.size // 3)+2)

    return centers

def compute_edge_center(zone):
  """Compute the edge centers of a partitioned zone.

  Input zone must have cartesian or cylindrical coordinates recorded under a unique
  GridCoordinates node, and a unstructured standard elements connectivity.

  Note:
    If zone is described with standard elements, centers will be computed for elements
    explicitly defined in cgns tree.

  Args:
    zone (CGNSTree): Partitionned 2D or 3D U-elts CGNS Zone
  Returns:
    array: Flat (interlaced) numpy array of edge centers

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_edge_center@start
        :end-before: #compute_edge_center@end
        :dedent: 2
  """
  coords = PT.Zone.coordinates(zone)

  if PT.Zone.Type(zone) == "Unstructured":
    edge_vtx_idx, edge_vtx = CU.cell_vtx_connectivity(zone, dim=1)
    if isinstance(coords, PT.CartesianCoordinates):
      return _mean_coords_from_connectivity(edge_vtx_idx, edge_vtx, *coords)
    elif isinstance(coords, PT.CylindricalCoordinates):
      return _mean_coords_from_connectivity_cyl(edge_vtx_idx, edge_vtx, *coords)
  else:
    raise NotImplementedError("Only U-elts zones are managed")
