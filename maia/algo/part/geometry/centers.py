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

def _reduce_mean(vtx_id, vtx_id_idx, *arrays, skip_odd_coords=False):
  if not len(vtx_id_idx): return np.array([]),np.array([]),np.array([])
  vtx_id_n = np.diff(vtx_id_idx)
  coords_mean = []
  for array in arrays:
    array_vtx_id = array[vtx_id-1]
    # cf issue #147
    if np.amax(vtx_id_idx[:-1],initial=0) == len(array_vtx_id):
      array_vtx_id = np.append(array_vtx_id,0)
      coord_sum = np.add.reduceat(array_vtx_id, vtx_id_idx)[:-1]
    elif vtx_id_idx[-1] == len(array_vtx_id):
      coord_sum = np.add.reduceat(array_vtx_id, vtx_id_idx[:-1])
    else:
      coord_sum = np.add.reduceat(array_vtx_id, vtx_id_idx)[:-1]
    if skip_odd_coords:
      coords_mean.append(coord_sum[::2] / vtx_id_n[::2])
    else:
      coords_mean.append(coord_sum / vtx_id_n)
  return coords_mean

def _mean_coords_from_connectivity(vtx_id_idx, vtx_id, cx, cy, cz, skip_odd_coords=False):
  coords_mean = _reduce_mean(vtx_id, vtx_id_idx, cx, cy, cz, skip_odd_coords=skip_odd_coords)
  return np_utils.interweave_arrays(coords_mean)

def _mean_coords_from_connectivity_cyl(vtx_id_idx, vtx_id, cr, ctheta, cz, skip_odd_coords=False):

  # vtx_id_n = np.diff(vtx_id_idx)

  cx,cy,cz = _to_xyz(cr, ctheta, cz)
  coords_mean = _reduce_mean(vtx_id, vtx_id_idx, cx, cy, cz, skip_odd_coords=skip_odd_coords)
  return np_utils.interweave_arrays(_to_rthetaz(*coords_mean))

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
    # cell_indices must be broadcastable to (1,face_nb) (to follow the cgns standard)
    cell_indices = np.atleast_2d(cell_indices).astype(np.int32)
    assert cell_indices.ndim == 2
    assert cell_indices.shape[0] == 1
    if not len(cell_indices[0]): return np.array([],dtype=np.float64)

  if PT.Zone.Type(zone) == "Unstructured":
    n_cell     = PT.Zone.n_cell(zone)
    if PT.Zone.has_ngon_elements(zone):
      face_vtx_idx, face_vtx, ngon_pe = PT.Zone.ngon_connectivity(zone)
      if cell_indices is not None:
        n_face = len(ngon_pe)
        n_cell = len(cell_indices[0])
        cell_permutation = np.argsort(cell_indices[0])
        sorted_cell_indices = cell_indices[0][cell_permutation]+n_face
        isin_msk = np.isin(ngon_pe,sorted_cell_indices)
        ngon_pe[~isin_msk] = 0

        replace_sorted = np.arange(n_cell)+n_face+1
        # having the same 'order' guarantees pe_view is a view and not a copy
        pe_view = ngon_pe.ravel(order="F" if ngon_pe.flags.f_contiguous else "C")
        msk = pe_view !=0
        find_ind = np.searchsorted(sorted_cell_indices,pe_view[msk])
        pe_view[msk] = replace_sorted[find_ind]

      if isinstance(coords, PT.CylindricalCoordinates):
        center_cell = cpart_algo.compute_center_cell_u_cyl(n_cell, *coords, face_vtx, face_vtx_idx, ngon_pe)
      elif isinstance(coords, PT.CartesianCoordinates):
        center_cell = cpart_algo.compute_center_cell_u(n_cell, *coords, face_vtx, face_vtx_idx, ngon_pe)
        if cell_indices is not None:
          # reordering cells in the order required by the user
          for i in range(3): 
            # '.copy()' is needed because otherwise numpy overrides the array improperly
            center_cell[3*cell_permutation+i] = center_cell[i::3].copy()
    else:
      cell_vtx_idx, cell_vtx = CU.cell_vtx_connectivity(zone)
      if cell_indices is None:
        if isinstance(coords, PT.CylindricalCoordinates):
          center_cell = _mean_coords_from_connectivity_cyl(cell_vtx_idx, cell_vtx, *coords)
        elif isinstance(coords, PT.CartesianCoordinates):
          center_cell = _mean_coords_from_connectivity(cell_vtx_idx, cell_vtx, *coords)
      else:
        if len(cell_indices[0]):
          cell_vtx_idx = np.concatenate([cell_vtx_idx[ind:ind+2] for ind in cell_indices[0]-1])
        else:
          cell_vtx_idx = np.array([],dtype=np.int32)
        if isinstance(coords, PT.CylindricalCoordinates):
          center_cell = _mean_coords_from_connectivity_cyl(cell_vtx_idx, cell_vtx, *coords, skip_odd_coords=True)
        elif isinstance(coords, PT.CartesianCoordinates):
          center_cell = _mean_coords_from_connectivity(cell_vtx_idx, cell_vtx, *coords, skip_odd_coords=True)
  else:
    if isinstance(coords, PT.CylindricalCoordinates):
      center_cell = cpart_algo.compute_center_cell_s_cyl(*PT.Zone.CellSize(zone), *coords)
    elif isinstance(coords, PT.CartesianCoordinates):
      center_cell = cpart_algo.compute_center_cell_s(*PT.Zone.CellSize(zone), *coords)
    if cell_indices is not None: # filtering afterward
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
    # face_indices must be broadcastable to (1,face_nb) (to follow the cgns standard)
    face_indices = np.atleast_2d(face_indices).astype(np.int32)
    assert face_indices.ndim == 2
    assert face_indices.shape[0] == 1
    if not len(face_indices[0]): return np.array([],dtype=np.float64)

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      face_vtx_idx, face_vtx = CU.cell_vtx_connectivity(zone, dim=2)
    _coords = coords if coords[2] is not None else [coords[0], coords[1], np.zeros_like(coords[0])]
    if face_indices is None:
      if isinstance(coords, PT.CartesianCoordinates):
        return _mean_coords_from_connectivity(face_vtx_idx, face_vtx, *_coords)
      elif isinstance(coords, PT.CylindricalCoordinates):
        return _mean_coords_from_connectivity_cyl(face_vtx_idx, face_vtx, *_coords)
    else:
      face_vtx_idx = np.concatenate([face_vtx_idx[ind:ind+2] for ind in face_indices[0]-1])
      if isinstance(coords, PT.CartesianCoordinates):
        return _mean_coords_from_connectivity(face_vtx_idx, face_vtx, *_coords, skip_odd_coords=True)
      elif isinstance(coords, PT.CylindricalCoordinates):
        return _mean_coords_from_connectivity_cyl(face_vtx_idx, face_vtx, *_coords, skip_odd_coords=True)
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
    # edge_indices must be broadcastable to (1,edge_nb) (to follow the cgns standard)
    edge_indices = np.atleast_2d(edge_indices).astype(np.int32)
    assert edge_indices.ndim == 2
    assert edge_indices.shape[0] == 1
    if not len(edge_indices[0]): return np.array([],dtype=np.float64)

  _coords = []
  for c in coords:
    _coords.append(c if c is not None else np.zeros_like(coords[0]))

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone) and PT.Zone.CellDimension(zone) == 3:
      raise NotImplementedError("Only U-elts zones are managed")
    edge_vtx_idx, edge_vtx = CU.cell_vtx_connectivity(zone, dim=1)
    if edge_indices is None:
      if isinstance(coords, PT.CartesianCoordinates):
        return _mean_coords_from_connectivity(edge_vtx_idx, edge_vtx, *_coords)
      elif isinstance(coords, PT.CylindricalCoordinates):
        return _mean_coords_from_connectivity_cyl(edge_vtx_idx, edge_vtx, *_coords)
    else:
      edge_vtx_idx = np.concatenate([edge_vtx_idx[ind:ind+2] for ind in edge_indices[0]-1])
      if isinstance(coords, PT.CartesianCoordinates):
        return _mean_coords_from_connectivity(edge_vtx_idx, edge_vtx, *_coords, skip_odd_coords=True)
      elif isinstance(coords, PT.CylindricalCoordinates):
        return _mean_coords_from_connectivity_cyl(edge_vtx_idx, edge_vtx, *_coords, skip_odd_coords=True)
  else:
    raise NotImplementedError("Only U-elts zones are managed")


def _compute_elements_center(zone, dim, element_indices=None):
  """Dispatch centers computing according to zone dimension and 
  requested dimension.
  Return a raw interlaced array or None"""
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'CellCenter':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return compute_cell_center(zone,cell_indices=element_indices)
  elif dim == 2 and zone_dim >= 2:
    return compute_face_center(zone,face_indices=element_indices)
  elif dim == 1 and zone_dim >= 1:
    return compute_edge_center(zone,edge_indices=element_indices)

def compute_elements_center(zone, dim, element_indices=None):
  """ Implementation of maia.algo.compute_elements_center for a given partitioned zone.
  See the calling function for full documentation """
  
  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  interlaced_centers = _compute_elements_center(zone, rq_dim, element_indices)
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