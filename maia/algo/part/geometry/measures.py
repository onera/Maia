import numpy as np

import maia.pytree      as PT

from   maia.utils     import np_utils
from   maia.utils     import logging as mlog

from   maia.algo.part import connectivity_utils as CU

from .utils import get_local_coordinates, place_in_container

from maia.algo.geometry_utils import ELT_FACE_VTX, compute_center_and_flux

import cmaia.part_algo as cpart_algo

def compute_edge_measure(zone):
  """ Compute the length of all edges of a 1D, 2D or 3D zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"

  if PT.Zone.Type(zone) == "Unstructured":
    edge_vtx_idx, edge_vtx = CU.cell_vtx_connectivity(zone, dim=1)

    # Compute length : |L| = ||x2 - x1||
    first_vtx  = edge_vtx[0::2] - 1
    second_vtx = edge_vtx[1::2] - 1
    length = (coords[0][second_vtx] - coords[0][first_vtx])**2
    if (cy := coords[1]) is not None:
      length += (cy[second_vtx] - cy[first_vtx])**2
    if (cz := coords[2]) is not None:
      length += (cz[second_vtx] - cz[first_vtx])**2
    return np.sqrt(length)
  else:
    raise NotImplementedError("Structured zones are not managed")

def compute_face_measure(zone):
  """ Compute the area of all faces of a 2D or 3D zone and return a raw array"""

  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"


  if PT.Zone.Type(zone) == "Unstructured":

    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      face_vtx_idx, face_vtx = CU.cell_vtx_connectivity(zone, dim=2)

    local_coords = get_local_coordinates(zone, face_vtx)
    face_vtx_n = np.diff(face_vtx_idx)
    _, normalflux = compute_center_and_flux(local_coords, face_vtx_idx, face_vtx_n)
    measure = np.linalg.norm(normalflux, axis=1)

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
    measure = cpart_algo.compute_area_face_s(*vtx_size, _cx, _cy, _cz)

  return measure

def _compute_elt_volume(zone, elt_node, coords, out):
  assert out.size == PT.Element.Size(elt_node)
  elt_kind = PT.Element.CGNSName(elt_node)

  ec = PT.get_child_from_name(elt_node, 'ElementConnectivity')[1]

  # Use direct formula for TETRA since they are always planar. Otherwise, fallback to
  # circulation formulae for polyedron (maybe more costly, but this is more robust if
  # there is non planar faces)
  if elt_kind == 'TETRA_4':
    vtxa = ec[0::4] - 1
    vtxb = ec[1::4] - 1
    vtxc = ec[2::4] - 1
    vtxd = ec[3::4] - 1
    a = [coords[i][vtxa] - coords[i][vtxd] for i in range(3)]
    b = [coords[i][vtxb] - coords[i][vtxd] for i in range(3)]
    c = [coords[i][vtxc] - coords[i][vtxd] for i in range(3)]
    out[:] = np.fabs(a[0]*b[1]*c[2] + b[0]*c[1]*a[2] + c[0]*a[1]*b[2] 
                   - c[0]*b[1]*a[2] - b[0]*a[1]*c[2] - a[0]*c[1]*b[2]) / 6.

  else:
    n_elt = PT.Element.Size(elt_node)
    base_n, base_seq = ELT_FACE_VTX[elt_kind]

    face_vtx_n = np.tile(base_n, n_elt)
    face_vtx_idx = np_utils.sizes_to_indices(face_vtx_n)

    # Where to read in element connectivity to reconstitute all faces (with reps)
    read_idx = np.tile(base_seq, n_elt) + np.repeat(PT.Element.NVtx(elt_node)*np.arange(n_elt), base_seq.size)
    face_vtx = ec[read_idx]

    # Final assembly : for each cell, sum the quantities computed on each cell. We don't need to recover cell_face
    # since this is identity by construction
    local_coords = get_local_coordinates(zone, face_vtx)
    center, normalflux = compute_center_and_flux(local_coords, face_vtx_idx, face_vtx_n)
    face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux

    cell_face_idx = base_n.size * np.arange(PT.Element.Size(elt_node))
    np.add.reduceat(face_contrib, cell_face_idx, out=out)
    out *= (1/3.)

def compute_cell_measure(zone):
  """ Compute the volume of all cells of a 3D zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):

      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
      face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      face_vtx_n   = np.diff(face_vtx_idx)

      nface_node = PT.Zone.NFaceNode(zone)
      cell_face_idx = PT.get_child_from_name(nface_node, 'ElementStartOffset')[1]
      cell_face     = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]

      local_coords = get_local_coordinates(zone, face_vtx)

      center, normalflux = compute_center_and_flux(local_coords, face_vtx_idx, face_vtx_n)
      face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux

      # Assembly : for each cell, sum the quantities computed on each face
      measure = (1/3.) * np.add.reduceat(np.sign(cell_face) * face_contrib[np.abs(cell_face)-1], cell_face_idx[:-1])

    else:
      measure = np.empty(PT.Zone.n_cell(zone))
      start = 0
      for elt in PT.Zone.get_ordered_elements_per_dim(zone)[3]:
        end = start + PT.Element.Size(elt)
        _compute_elt_volume(zone, elt, coords, measure[start:end])
        start = end
  else:
    measure = cpart_algo.compute_volume_cell_s(*PT.Zone.CellSize(zone), *coords)

  return measure


def _compute_zone_measures(zone, dim):
  """Dispatch measures computing according to zone dimension and 
  requested dimension. Return a raw array"""
  if dim == 'CellCenter':
    dim = PT.Zone.CellDimension(zone)
  return {3: compute_cell_measure,
          2: compute_face_measure,
          1: compute_edge_measure}[dim](zone)

def compute_zone_measures(zone, dim):
  """ Implementation of maia.algo.compute_measures for a given partitioned zone.
  See the calling function for full documentation """

  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  if cell_dim < rq_dim:
    msg = f"Zone '{PT.get_name(zone)}' skipped during measures computing because "\
          f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
    mlog.warning(msg)
  else:
    measure = _compute_zone_measures(zone, rq_dim)
    if measure.size > 0:
      place_in_container(zone, rq_dim, {'Measure' : measure})