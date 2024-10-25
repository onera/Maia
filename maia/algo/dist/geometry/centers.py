import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import connectivity_utils as CU
from maia.utils     import np_utils
from maia.utils     import logging as mlog
from maia.transfer  import protocols as EP

from ..s_to_u       import zonedims_to_ngon

from .utils import place_in_container

def _to_xyz(r, theta, z):
  return r*np.cos(theta), r*np.sin(theta), z
def _to_rthetaz(x, y, z):
  return np.sqrt(x**2+y**2), np.arctan2(y, x), z

def _reduce_mean(vtx_id_idx, *arrays, skip_odd_coords=False):
  # cf issue #147
  vtx_id_n = np.diff(vtx_id_idx)
  coords_mean = []
  for array in arrays:
    if vtx_id_idx[-1] == len(array):
      coord_sum = np.add.reduceat(array, vtx_id_idx[:-1])
    else:
      coord_sum = np.add.reduceat(array, vtx_id_idx)[:-1]
    if skip_odd_coords:
      coords_mean.append(coord_sum[::2] / vtx_id_n[::2])
    else:
      coords_mean.append(coord_sum / vtx_id_n)
  return coords_mean
  # vtx_id_n = np.diff(vtx_id_idx)
  # return [np.add.reduceat(array, vtx_id_idx[:-1]) / vtx_id_n for array in arrays]


def _mean_coords_from_connectivity(vtx_id_idx, cx_expd, cy_expd, cz_expd, skip_odd_coords=False):
  """ Coordinates should be repeted to match the size of vtx_id_idx """
  coords_mean = _reduce_mean(vtx_id_idx, cx_expd, cy_expd, cz_expd, skip_odd_coords=skip_odd_coords)
  return np_utils.interweave_arrays(coords_mean)

def _mean_coords_from_connectivity_cyl(vtx_id_idx, cr_expd, ctheta_expd, cz_expd, skip_odd_coords=False):
  """ Coordinates should be repeted to match the size of vtx_id_idx """
  cx,cy,cz = _to_xyz(cr_expd, ctheta_expd, cz_expd)
  coords_mean = _reduce_mean(vtx_id_idx, cx, cy, cz, skip_odd_coords=skip_odd_coords)
  return np_utils.interweave_arrays(_to_rthetaz(*coords_mean))

def compute_edge_center(zone, comm):
  """Compute the edge centers of a distributed zone.

  Input zone must have cartesian coordinates or cylindrical coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the edges.
  """
  if PT.Zone.Type(zone) == "Unstructured":
    global_distri = PT.Zone.CellDimension == 1
    edge_vtx_idx, edge_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 1, global_distri)
  else:
    raise NotImplementedError("Only U zones are managed")

  coords = PT.Zone.coordinates(zone)

  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)) if coords[i] is not None)
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [edge_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]

  while len(local_coords) < 3 : #We are in phydim < 3 case, add Y and/or Z array
    local_coords.append(np.zeros_like(local_coords[0]))

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(edge_vtx_idx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(edge_vtx_idx, *local_coords)

def compute_face_center(zone, comm, face_indices=None):
  """Compute the face center of a distributed zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.

  Centers are computed using a basic average over the vertices of the faces.

  Args:
    zone (CGNSTree): Distributed 3D or 2D U-NGon CGNS Zone
  Returns:
    face_normal (array): Flat (interlaced) numpy array of face centers

  """
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"

  from super_miles.utils import logger
  if PT.Zone.Type(zone) == "Structured":
    vtx_size = np.ones(3, zone[1].dtype) # This trick allows to call zonedims_to_ngon even on 2D meshes
    vtx_size[:zone_dim] = PT.Zone.VertexSize(zone)
    ngon_node = zonedims_to_ngon(vtx_size, comm)
    _face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
    face_vtx_idx = np.empty(_face_vtx_idx.size, np.int32)
    np.subtract(_face_vtx_idx, _face_vtx_idx[0], out=face_vtx_idx)
    face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
  else:
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      _face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx_idx = np.empty(_face_vtx_idx.size, np.int32)
      np.subtract(_face_vtx_idx, _face_vtx_idx[0], out=face_vtx_idx)
      if face_indices is not None:
        all_face_distri = MT.getDistribution(ngon_node, 'Element')[1]
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      global_distri = PT.Zone.CellDimension(zone) == 2
      face_vtx_idx, face_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 2, global_distri)


  if face_indices is not None:
    dface_stride = np.diff(face_vtx_idx).astype(np.int32, copy=False)
    face_indices = np.atleast_2d(face_indices)
    assert face_indices.ndim == 2
    assert face_indices.shape[0] == 1
    # block to part avec la dist des faces avec ln_to_gn == face_indices
    # --> on va chercher seulement les face_vtx qui nous interessent
    from super_miles.utils import logger
    logger.error(comm.rank,dface_stride,face_indices)
    ext_face_vtx_stride, ext_face_vtx = EP.block_to_part_strided(dface_stride,
                          face_vtx, all_face_distri, [face_indices[0]-1], comm)
    face_vtx = ext_face_vtx[0]
    face_vtx_idx = np.cumsum(np.concatenate([[0],ext_face_vtx_stride[0]]))
    logger.warn(comm.rank,face_vtx,face_vtx_idx)
  
  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)) if coords[i] is not None)
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]

  if len(local_coords) == 2 : #We are in phydim==2, Add Z array
    local_coords.append(np.zeros_like(local_coords[0]))

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(face_vtx_idx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(face_vtx_idx, *local_coords)

def compute_cell_center(zone, comm):
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  if PT.Zone.Type(zone) == "Structured":
    cell_vtx_idx, cell_vtx = CU.cell_vtx_connectivity_S(zone, PT.Zone.CellDimension(zone))
  else:
    if PT.Zone.has_ngon_elements(zone):
      cell_vtx_idx, cell_vtx = CU.cell_vtx_connectivity_ngon(zone, comm)
    else:
      cell_vtx_idx, cell_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 3, True)


  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)))
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [cell_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(cell_vtx_idx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(cell_vtx_idx, *local_coords)


def _compute_elements_center(zone, dim, comm):
  """Dispatch centers computing according to zone dimension and 
  requested dimension
  Return a raw interlaced array or None"""
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'CellCenter':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return compute_cell_center(zone, comm)
  elif dim == 2 and zone_dim >= 2:
    return compute_face_center(zone, comm)
  elif dim == 1 and zone_dim >= 1:
    return compute_edge_center(zone, comm)

def compute_elements_center(zone, dim, comm):
  """ Implementation of maia.algo.compute_elements_center for a given distributed zone.
  See the above function for full documentation """

  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  interlaced_centers = _compute_elements_center(zone, rq_dim, comm)
  if interlaced_centers is None:
    msg = f"Zone '{PT.get_name(zone)}' skipped during centers computing because "\
          f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
    mlog.warning(msg)
  else:
    # Underlying function always return a concatenated array of size 3*n_entity
    # We must filter it if phy_dim is lower
    coords = PT.Zone.coordinates(zone)
    center_names = [s.replace('Coordinate', 'Center') for s in coords._fields]
    phy_dim  = len([c for c in coords if c is not None]) # 1, 2 or 3
    centers = {name : interlaced_centers[i::3] \
               for i,name in enumerate(center_names) if i < phy_dim}

    place_in_container(zone, rq_dim, centers, comm)