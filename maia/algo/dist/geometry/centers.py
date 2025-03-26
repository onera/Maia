import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import connectivity_utils as CU
from maia.utils     import np_utils
from maia.utils     import logging as mlog
from maia.transfer  import protocols as EP

from ..s_to_u       import zonedims_to_ngon

from .utils import place_in_container
from maia.utils import vstride as vs

def _to_xyz(r, theta, z):
  return r*np.cos(theta), r*np.sin(theta), z
def _to_rthetaz(x, y, z):
  return np.sqrt(x**2+y**2), np.arctan2(y, x), z

def _reduce_mean(vtx_id:vs.VStrideArray, *arrays):
  return [np.add.reduceat(array, vtx_id.displs[:-1]) / vtx_id.counts for array in arrays]


def _mean_coords_from_connectivity(vtx_id:vs.VStrideArray, cx_expd, cy_expd, cz_expd):
  """ Coordinates should be repeted to match the size of vtx_id_idx """
  coords_mean = _reduce_mean(vtx_id, cx_expd, cy_expd, cz_expd)
  return np_utils.interweave_arrays(coords_mean)

def _mean_coords_from_connectivity_cyl(vtx_id:vs.VStrideArray, cr_expd, ctheta_expd, cz_expd):
  """ Coordinates should be repeted to match the size of vtx_id_idx """
  cx,cy,cz = _to_xyz(cr_expd, ctheta_expd, cz_expd)
  coords_mean = _reduce_mean(vtx_id, cx, cy, cz)
  return np_utils.interweave_arrays(_to_rthetaz(*coords_mean))

def compute_edge_center(zone, comm, edge_indices=None):
  """Compute the edge centers of a distributed zone.

  Input zone must have cartesian coordinates or cylindrical coordinates recorded under a unique
  GridCoordinates node.
  Centers are computed using a basic average over the vertices of the edges.

  edge_indices is a pointlist like array of edges ids or None. If provided,
    centers are computed only for the specified edges. 
  """
  #if edge_indices is not None:
    #assert isinstance(edge_indices, np.ndarray) and edge_indices.ndim == 2 and edge_indices.shape[0] == 1

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone) and PT.Zone.CellDimension(zone) == 3:
      raise NotImplementedError("Only U-elts zones are managed")
    global_distri = PT.Zone.CellDimension == 1
    _edge_indices = edge_indices[0] if edge_indices is not None else None
    edge_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 1, global_distri, _edge_indices)
  else:
    raise NotImplementedError("Only U zones are managed")

  coords = PT.Zone.coordinates(zone)

  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)) if coords[i] is not None)
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, edge_vtx.values-1, comm)
  local_coords = [part_data[key] for key in part_data.keys()]

  while len(local_coords) < 3 : #We are in phydim < 3 case, add Y and/or Z array
    local_coords.append(np.zeros_like(local_coords[0]))

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(edge_vtx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(edge_vtx, *local_coords)

def compute_face_center(zone, comm, face_indices=None, face_indices_loc=None):
  """Compute the face center of a distributed zone.

  Input zone must have cartesian coordinates recorded under a unique
  GridCoordinates node.

  Centers are computed using a basic average over the vertices of the faces.

  Args:
    zone (CGNSTree): Distributed 3D or 2D U-NGon CGNS Zone
    face_indices (ndarray) : pointlist like array of faces ids or None. If provided,
      centers are computed only for the specified faces. If the mesh is structured 2D,
      face_indices_loc is requested as well and indicates the location of the pointlist.
  Returns:
    face_normal (array): Flat (interlaced) numpy array of face centers

  """
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"

  if face_indices is not None:
    #assert isinstance(face_indices, np.ndarray) and face_indices.ndim == 2
    if PT.Zone.Type(zone) == 'Structured' and zone_dim == 3:
      assert face_indices_loc in ['IFaceCenter', 'JFaceCenter', 'KFaceCenter'], \
        "Indices location must be specified when filtering faces center on 3D structured meshes"

  if PT.Zone.Type(zone) == "Structured" and zone_dim == 2:
    face_vtx = CU.cell_vtx_connectivity_S(zone, zone_dim, face_indices)
  elif PT.Zone.Type(zone) == "Unstructured" and not PT.Zone.has_ngon_elements(zone): # unstructured elements
    global_distri = PT.Zone.CellDimension(zone) == 2
    _face_indices = face_indices[0] if face_indices is not None else None
    face_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 2, global_distri, _face_indices)

  # Other cases (Structured 3D or NGON) does not manage idx filtering : we do it manually
  else:
    if PT.Zone.Type(zone) == "Structured" and zone_dim == 3:
      ngon_node = zonedims_to_ngon(PT.Zone.VertexSize(zone), comm)
      if face_indices is not None:
        from maia.utils.numbering import s_numbering_funcs
        _face_indices = s_numbering_funcs.ijk_to_index_from_loc(*face_indices, face_indices_loc, PT.Zone.VertexSize(zone)) - 1
    elif PT.Zone.Type(zone) == "Unstructured" and PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      if face_indices is not None:
        _face_indices = face_indices[0] - PT.Element.Range(ngon_node)[0]

    face_vtx = MT.Element.connectivity(ngon_node)
    if face_indices is not None:
      face_distri = PT.maia.getDistribution(ngon_node, 'Element')[1]
      face_vtx_n, face_vtx_v = EP.block_to_part_strided(face_vtx.counts, face_vtx.values, face_distri, _face_indices, comm)
      face_vtx = vs.from_counts(face_vtx_n, face_vtx_v)
  
  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)) if coords[i] is not None)
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, face_vtx.values-1, comm)
  local_coords = [part_data[key] for key in part_data.keys()]

  if len(local_coords) == 2 : #We are in phydim==2, Add Z array
    local_coords.append(np.zeros_like(local_coords[0]))

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(face_vtx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(face_vtx, *local_coords)

def compute_cell_center(zone, comm, cell_indices=None):
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  #if cell_indices is not None:
    #assert isinstance(cell_indices, np.ndarray) and cell_indices.ndim == 2

  if PT.Zone.Type(zone) == "Structured":
    cell_vtx = CU.cell_vtx_connectivity_S(zone, PT.Zone.CellDimension(zone), cell_indices)
  else:
    _cell_indices = cell_indices[0] if cell_indices is not None else None
    if PT.Zone.has_ngon_elements(zone):
      cell_vtx = CU.cell_vtx_connectivity_ngon(zone, comm, _cell_indices)
    else:
      cell_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 3, True, _cell_indices)

  coords = PT.Zone.coordinates(zone)
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)))
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, cell_vtx.values-1, comm)
  local_coords = [part_data[key] for key in part_data.keys()]

  if isinstance(coords, PT.CartesianCoordinates):
    return _mean_coords_from_connectivity(cell_vtx, *local_coords)
  elif isinstance(coords, PT.CylindricalCoordinates):
    return _mean_coords_from_connectivity_cyl(cell_vtx, *local_coords)


def _compute_elements_center(zone, dim, comm, element_indices=None, element_loc=None):
  """Dispatch centers computing according to zone dimension and 
  requested dimension
  If element_indices is not None, center is computed only for the
  specified elements (in absolute numbering)
  Return a raw interlaced array or None"""
  zone_dim = PT.Zone.CellDimension(zone)
  if dim == 'CellCenter':
    dim = zone_dim
  if dim == 3 and zone_dim >= 3:
    return compute_cell_center(zone, comm, element_indices)
  elif dim == 2 and zone_dim >= 2:
    return compute_face_center(zone, comm, element_indices, element_loc)
  elif dim == 1 and zone_dim >= 1:
    return compute_edge_center(zone, comm, element_indices)

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
