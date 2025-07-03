import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from   maia.utils     import np_utils, s_numbering
from   maia.utils     import vstride as vs
from   maia.utils     import logging as mlog

from   maia.algo.part import connectivity_utils as CU

from .utils import get_local_coordinates, place_in_container

from maia.algo.geometry_utils import ELT_FACE_VTX, compute_center_and_flux

import cmaia.part_algo as cpart_algo

def compute_edge_measure(zone, edge_indices=None):
  """ Compute the length of all edges of a 1D, 2D or 3D zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"

  if edge_indices is not None:
    assert isinstance(edge_indices, np.ndarray) and edge_indices.ndim == 2 and edge_indices.shape[0] == 1

  if PT.Zone.Type(zone) == "Unstructured":
    edge_vtx = CU.cell_vtx_connectivity(zone, 1, edge_indices)

    # Compute length : |L| = ||x2 - x1||
    first_vtx  = edge_vtx.values[0::2] - 1
    second_vtx = edge_vtx.values[1::2] - 1
    length = (coords[0][second_vtx] - coords[0][first_vtx])**2
    if (cy := coords[1]) is not None:
      length += (cy[second_vtx] - cy[first_vtx])**2
    if (cz := coords[2]) is not None:
      length += (cz[second_vtx] - cz[first_vtx])**2
    return np.sqrt(length)
  else:
    raise NotImplementedError("Structured zones are not managed")

def compute_face_measure(zone, face_indices=None, face_indices_loc=None):
  """ Compute the area of all faces of a 2D or 3D zone and return a raw array"""

  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"

  if face_indices is not None:
    assert isinstance(face_indices, np.ndarray) and face_indices.ndim == 2
    if PT.Zone.Type(zone) == 'Structured' and zone_dim == 3:
      assert face_indices_loc in ['IFaceCenter', 'JFaceCenter', 'KFaceCenter'], \
        "Indices location must be specified when filtering faces measure on 3D structured meshes"
    if face_indices.size == 0:
      return np.empty(0, dtype=np.float64)

  # For S/2D zones, if face_indices is provided, it is faster to rebuild face_vtx filtered cnt,
  # as for unstructured cases. Il faces_indices is None (ie we compute all faces) pybind
  # function is more efficient
  if PT.Zone.Type(zone) == "Unstructured" or (zone_dim==2 and face_indices is not None):

    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx = MT.Element.connectivity(ngon_node)
      if face_indices is not None:
        face_vtx = vs.take(face_vtx, face_indices[0]-PT.Element.Range(ngon_node)[0])
    else:
      face_vtx = CU.cell_vtx_connectivity(zone, 2, face_indices)

    local_coords = get_local_coordinates(zone, face_vtx.values)
    _, normalflux = compute_center_and_flux(local_coords, face_vtx.displs, face_vtx.counts)
    measure = np.linalg.norm(normalflux, axis=1)

  else:
    # This is for 3D S zones or 2D zones w/o filtering
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
    if face_indices is not None:
      _face_indices = s_numbering.ijk_to_index_from_loc(*face_indices, face_indices_loc, PT.Zone.VertexSize(zone))
      measure = measure[_face_indices-1]

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

def compute_cell_measure(zone, cell_indices=None):
  """ Compute the volume of all cells of a 3D zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  if cell_indices is not None:
    assert isinstance(cell_indices, np.ndarray) and cell_indices.ndim == 2
    if cell_indices.size == 0:
      return np.empty(0, dtype=np.float64)

  if PT.Zone.Type(zone) == "Unstructured":
    if PT.Zone.has_ngon_elements(zone):

      nface_node = PT.Zone.NFaceNode(zone)
      cell_face = MT.Element.connectivity(nface_node)

      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx = MT.Element.connectivity(ngon_node)

      if cell_indices is not None:
        # 1. Filter cell_face
        cell_face = vs.take(cell_face, cell_indices[0]-PT.Element.Range(nface_node)[0])
        # 2. Filter face_vtx : we need to detect appearing faces
        face_selector = np.zeros(len(face_vtx), bool)
        face_selector[np.abs(cell_face.values)-PT.Element.Range(ngon_node)[0]] = True
        face_vtx = vs.take(face_vtx, np.where(face_selector)[0])
        # 3. Since we filtered faces, we need an indirection to access face_contrib
        # (which will be computed only on 'active' faces)
        old_to_new = np.cumsum(face_selector)
        face_accessor = old_to_new[np.abs(cell_face.values)-1]-1 # At this point keeping sign is useless
      else:
        face_accessor = np.abs(cell_face.values)-1
      local_coords = get_local_coordinates(zone, face_vtx.values)

      center, normalflux = compute_center_and_flux(local_coords, face_vtx.displs, face_vtx.counts)
      face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux

      # Assembly : for each cell, sum the quantities computed on each face
      measure = (1/3.) * np.add.reduceat(np.sign(cell_face.values) * face_contrib[face_accessor], cell_face.displs[:-1])

    else:
      volumic_sections = PT.Zone.get_ordered_elements_per_dim(zone)[3]
      if cell_indices is not None:
        # If cell_indices is provided, we still need to work section by section
        # The idea is to build "fake sections" where only the referenced elts appears
        assert PT.Zone.elt_ordering_by_dim(zone) != 0, "Elements sections must be sorted by dim"
        offset = PT.Element.Range(volumic_sections[0])[0]
        section_mask = np.zeros(PT.Zone.n_cell(zone), bool)
        section_mask[cell_indices[0]-offset] = True

        start = 0
        fake_elts = []
        for elt in volumic_sections:
          end = start + PT.Element.Size(elt)
          cur_section_mask = section_mask[start:end]
          elt_vtx = MT.Element.connectivity(elt)
          _elt = PT.new_Elements(f"Fake_{PT.get_name(elt)}", 
                                 PT.Element.CGNSName(elt), 
                                 erange=[1, cur_section_mask.sum()], # Size matters but range doesnt
                                 econn=vs.take(elt_vtx, np.where(cur_section_mask)[0]).values)
          fake_elts.append(_elt)
          start = end

        volumic_sections = fake_elts

      measure = np.empty(sum(PT.Element.Size(elt) for elt in volumic_sections))
      start = 0
      for elt in volumic_sections:
        end = start + PT.Element.Size(elt)
        _compute_elt_volume(zone, elt, coords, measure[start:end])
        start = end

      if cell_indices is not None:
        # We need to reorder measure (which is in section order) to access elts in cell_indices order
        # Since we filtered elements we have the additional old_to_new indirection
        old_to_new = np.cumsum(section_mask)
        measure = measure[old_to_new[cell_indices[0]-offset]-1]
        
  else: # Structured meshes
    if cell_indices is not None:
      cell_vtx = CU.cell_vtx_connectivity(zone, 3, cell_indices)
      _elt = PT.new_Elements("Fake_Hexa", "HEXA_8", erange=[1, len(cell_vtx)], econn=cell_vtx.values)
      _compute_elt_volume(zone, _elt, coords, measure:=np.empty(len(cell_vtx)))
    else:
      measure = cpart_algo.compute_volume_cell_s(*PT.Zone.CellSize(zone), *coords)

  return measure


def _compute_elements_measure(zone, dim, element_indices=None, element_loc=None):
  """Dispatch measures computing according to zone dimension and 
  requested dimension.
  If element_indices is not None, measure is computed only for the
  specified elements (in absolute numbering)
  Return a raw array"""
  if dim == 'CellCenter':
    dim = PT.Zone.CellDimension(zone)
  if dim == 3:
    return compute_cell_measure(zone, element_indices)
  elif dim == 2:
    return compute_face_measure(zone, element_indices, element_loc)
  elif dim == 1:
    return compute_edge_measure(zone, element_indices)

def compute_elements_measure(zone, dim):
  """ Implementation of maia.algo.compute_elements_measure for a given partitioned zone.
  See the calling function for full documentation """

  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  if cell_dim < rq_dim:
    msg = f"Zone '{PT.get_name(zone)}' skipped during measures computing because "\
          f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
    mlog.warning(msg)
  else:
    measure = _compute_elements_measure(zone, rq_dim)
    if measure.size > 0:
      place_in_container(zone, rq_dim, {'Measure' : measure})