import numpy as np

import maia.pytree      as PT

from maia.algo.part import connectivity_utils as CU
from maia.utils     import np_utils
from maia.utils     import logging as mlog

from .utils         import place_in_container

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

def compute_cell_center(zone, cell_indices=None):
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
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  if cell_indices is not None:
    assert isinstance(cell_indices, np.ndarray) and cell_indices.ndim == 2 and cell_indices.shape[0] == 1
    if cell_indices.size == 0:
      return np.empty(0, dtype=np.float64)

  if PT.Zone.Type(zone) == "Unstructured":
    cell_vtx_idx, cell_vtx = CU.cell_vtx_connectivity(zone, dim=3, elts_subset=cell_indices)

    if isinstance(coords, PT.CylindricalCoordinates):
      center_cell = _mean_coords_from_connectivity_cyl(cell_vtx_idx, cell_vtx, *coords)
    elif isinstance(coords, PT.CartesianCoordinates):
      center_cell = _mean_coords_from_connectivity(cell_vtx_idx, cell_vtx, *coords)
        
  else:
    if isinstance(coords, PT.CylindricalCoordinates):
      center_cell = cpart_algo.compute_center_cell_s_cyl(*PT.Zone.CellSize(zone), *coords)
    elif isinstance(coords, PT.CartesianCoordinates):
      center_cell = cpart_algo.compute_center_cell_s(*PT.Zone.CellSize(zone), *coords)
    if cell_indices is not None:
      #raise NotImplementedError # Input should be a pointlist of size 3, and not a global num, which is not defined in S CGNS
      center_cell = np_utils.interweave_arrays([center_cell[i::3][cell_indices[0]-1] for i in range(3)])

  return center_cell

def compute_face_center(zone,face_indices=None):
  """Compute the face centers of a partitioned zone.

  Input zone must have cartesian or cylindrical coordinates recorded under a unique
  GridCoordinates node.

  Centers are computed using a basic average over the vertices of the faces.

  Note:
    If zone is described with standard elements, centers will be computed for elements
    explicitly defined in cgns tree.

  Args:
    zone (CGNSTree): Partitionned 2D or 3D U CGNS Zone
    face_indices ((n_face,) array): Optional face index filtering array 
  Returns:
    array: Flat (interlaced) numpy array of face centers

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_face_center@start
        :end-before: #compute_face_center@end
        :dedent: 2
  """
  coords = PT.Zone.coordinates(zone)
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"

  if face_indices is not None:
    assert isinstance(face_indices, np.ndarray) and face_indices.ndim == 2 and face_indices.shape[0] == 1
    if face_indices.size == 0:
      return np.empty(0, dtype=np.float64)

  if PT.Zone.Type(zone) == "Unstructured":
    face_vtx_idx, face_vtx = CU.cell_vtx_connectivity(zone, dim=2, elts_subset=face_indices)
    _coords = coords if coords[2] is not None else [coords[0], coords[1], np.zeros_like(coords[0])]
    if isinstance(coords, PT.CartesianCoordinates):
      return _mean_coords_from_connectivity(face_vtx_idx, face_vtx, *_coords)
    elif isinstance(coords, PT.CylindricalCoordinates):
      return _mean_coords_from_connectivity_cyl(face_vtx_idx, face_vtx, *_coords)
  else:
    vtx_size = [1,1,1]
    vtx_size[:zone_dim] = PT.Zone.VertexSize(zone)
    # Create cz if zone_dim == 2 & cz is None
    _cx = np.atleast_3d(coords[0]) # Auto expand arrays if zone_dim == 2
    _cy = np.atleast_3d(coords[1])
    if zone_dim == 2 and coords[2] is None:
      _cz = np.zeros(vtx_size, dtype=float, order='F')
    else:
      _cz = np.atleast_3d(coords[2])
    if isinstance(coords, PT.CartesianCoordinates):
      centers = cpart_algo.compute_center_face_s(*vtx_size, _cx, _cy, _cz)
    elif isinstance(coords, PT.CylindricalCoordinates):
      centers = cpart_algo.compute_center_face_s_cyl(*vtx_size, _cx, _cy, _cz)
    if face_indices is not None: # filtering afterward
      #raise NotImplementedError # Input should be a pointlist of size 3, and not a global num, which is not defined in S CGNS
      centers = np_utils.interweave_arrays([centers[i::3][face_indices[0]-1] for i in range(3)])
    return centers

def compute_edge_center(zone,edge_indices=None):
  """Compute the edge centers of a partitioned zone.

  Input zone must have cartesian or cylindrical coordinates recorded under a unique
  GridCoordinates node, and a unstructured standard elements connectivity.

  Note:
    If zone is described with standard elements, centers will be computed for elements
    explicitly defined in cgns tree.

  Args:
    zone (CGNSTree): Partitionned 2D or 3D U-elts CGNS Zone
    edge_indices ((n_edge,) array): Optional edge index filtering array 
  Returns:
    array: Flat (interlaced) numpy array of edge centers

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #compute_edge_center@start
        :end-before: #compute_edge_center@end
        :dedent: 2
  """
  coords = PT.Zone.coordinates(zone)

  if edge_indices is not None:
    assert isinstance(edge_indices, np.ndarray) and edge_indices.ndim == 2 and edge_indices.shape[0] == 1
    if edge_indices.size == 0:
      return np.empty(0, dtype=np.float64)

  _coords = []
  for c in coords:
    _coords.append(c if c is not None else np.zeros_like(coords[0]))

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone) and PT.Zone.CellDimension(zone) == 3:
      raise NotImplementedError("Only U-elts zones are managed")
    edge_vtx_idx, edge_vtx = CU.cell_vtx_connectivity(zone, dim=1, elts_subset=edge_indices)
    if isinstance(coords, PT.CartesianCoordinates):
      return _mean_coords_from_connectivity(edge_vtx_idx, edge_vtx, *_coords)
    elif isinstance(coords, PT.CylindricalCoordinates):
      return _mean_coords_from_connectivity_cyl(edge_vtx_idx, edge_vtx, *_coords)
  else:
    raise NotImplementedError("Only U-elts zones are managed")


def _compute_elements_center(zone, dim, element_indices=None):
  """Dispatch centers computing according to zone dimension and 
  requested dimension.
  If element_indices is not None, center is computed only for the
  specified elements (in absolute numbering)
  Return a raw interlaced array or None"""
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'CellCenter':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return compute_cell_center(zone, element_indices)
  elif dim == 2 and zone_dim >= 2:
    return compute_face_center(zone, element_indices)
  elif dim == 1 and zone_dim >= 1:
    return compute_edge_center(zone, element_indices)

def compute_elements_center(zone, dim):
  """ Implementation of maia.algo.compute_elements_center for a given partitioned zone.
  See the calling function for full documentation """
  
  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  interlaced_centers = _compute_elements_center(zone, rq_dim)
  if interlaced_centers is None:
    msg = f"Zone '{PT.get_name(zone)}' skipped during centers computing because "\
          f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
    mlog.warning(msg)
  elif interlaced_centers.size > 0:
    # Underlying function always return a concatenated array of size 3*n_entity
    # We must filter it if phy_dim is lower
    coords = PT.Zone.coordinates(zone)
    center_names = [s.replace('Coordinate', 'Center') for s in coords._fields]
    phy_dim  = len([c for c in coords if c is not None]) # 1, 2 or 3
    centers = {name : interlaced_centers[i::3] \
               for i,name in enumerate(center_names) if i < phy_dim}

    place_in_container(zone, rq_dim, centers)