import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from   maia.algo.part import connectivity_utils as CU
from   maia.utils     import np_utils
from   maia.utils     import logging as mlog

from maia.algo.geometry_utils import DIM_TO_LOC, get_or_create_container

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
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

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
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      face_vtx_idx, face_vtx = CU.cell_vtx_connectivity(zone, dim=2)
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

  _coords = []
  for c in coords:
    _coords.append(c if c is not None else np.zeros_like(coords[0]))

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone) and PT.Zone.CellDimension(zone) == 3:
      raise NotImplementedError("Only U-elts zones are managed")
    edge_vtx_idx, edge_vtx = CU.cell_vtx_connectivity(zone, dim=1)
    if isinstance(coords, PT.CartesianCoordinates):
      return _mean_coords_from_connectivity(edge_vtx_idx, edge_vtx, *_coords)
    elif isinstance(coords, PT.CylindricalCoordinates):
      return _mean_coords_from_connectivity_cyl(edge_vtx_idx, edge_vtx, *_coords)
  else:
    raise NotImplementedError("Only U-elts zones are managed")


def _compute_zone_centers(zone, dim):
  """Dispatch centers computing according to zone dimension and 
  requested dimension.
  Return a raw interlaced array or None"""
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'Cell':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return compute_cell_center(zone)
  elif dim == 2 and zone_dim >= 2:
    return compute_face_center(zone)
  elif dim == 1 and zone_dim >= 1:
    return compute_edge_center(zone)

def compute_zone_centers(zone, dim):
  """ Implementation of maia.algo.compute_centers for a given partitioned zone.
  See the above function for full documentation """
  
  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  interlaced_centers = _compute_zone_centers(zone, rq_dim)
  if interlaced_centers is None:
    msg = f"Zone '{PT.get_name(zone)}' skipped in compute_centers because "\
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

    output_loc = DIM_TO_LOC[cell_dim][rq_dim]
    if PT.Zone.Type(zone) == 'Structured':
      # Reshape is needed for S / part zones
      if output_loc == 'FaceCenter':
        # Zone is 3D, and we computed FaceCenter --> We have to split it into I/J/KFaceCenter
        facesize = PT.Zone.FaceSize(zone)
        dirfacesizefunc = [PT.Zone.IFaceSize, PT.Zone.JFaceSize, PT.Zone.KFaceSize]
        start = 0
        for i,dir in enumerate(['I', 'J', 'K']):
          end = start + facesize[i]
          newsize = dirfacesizefunc[i](zone)
          dircenter = {key: val[start:end].reshape(newsize, order='F') \
                       for key, val in centers.items()}
          container = get_or_create_container(zone, f'Geometry_{rq_dim}d_{dir}', f'{dir}{output_loc}', dircenter)
          start = end

      if output_loc == 'CellCenter':
        centers = {key: val.reshape(PT.Zone.CellSize(zone), order='F') \
                   for key, val in centers.items()}

        container = get_or_create_container(zone, f'Geometry_{rq_dim}d', output_loc, centers)

    else: # Unstructured
      container = get_or_create_container(zone, f'Geometry_{rq_dim}d', output_loc, centers)
      if output_loc in ['EdgeCenter', 'FaceCenter']: # PointList is supposed to be mandatory. Maybe we could make it optional in maia ?
        if PT.Zone.has_ngon_elements(zone):
          if output_loc == 'FaceCenter':
            ng = PT.Zone.NGonNode(zone)
          elif output_loc == 'EdgeCenter':
            assert PT.Zone.CellDimension(zone) == 2
            ng = MT.Zone.EdgeNode(zone)
          er = PT.Element.Range(ng)
          pl = np.arange(er[0], er[1]+1, dtype=np.int32).reshape((1,-1), order='F')
          gnum =  PT.maia.getGlobalNumbering(ng, 'Element')[1]
        else: # Must collect faces or edge in same order than the one used to compute face centers
          subdim = 2 if output_loc == 'FaceCenter' else 1
          ordered_faces = PT.Zone.get_ordered_elements_per_dim(zone)[subdim]
          sizes =  [PT.Element.Size(e) for e in ordered_faces]
          pl = np.empty((1, sum(sizes)), order='F', dtype=np.int32)
          start = 0
          for i,e in enumerate(ordered_faces):
            er = PT.Element.Range(e)
            pl[0,start:start+sizes[i]] = np.arange(er[0], er[1]+1, dtype=np.int32)
            start += sizes[i]
          # For gnum, we computed on all face or edge so Element/GlobalNumbering/Sections should be fine
          _, gnum = np_utils.concatenate_np_arrays([PT.maia.getGlobalNumbering(e, 'Sections')[1] for e in ordered_faces])

        PT.new_IndexArray('PointList', pl, container)
        PT.maia.newGlobalNumbering({'Index' : gnum}, container)

