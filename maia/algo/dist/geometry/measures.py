import numpy as np
import maia.pytree      as PT
import maia.pytree.maia as MT

import maia

from maia.algo.dist import connectivity_utils as CU
from maia.transfer  import protocols as EP

from maia.utils import np_utils
from maia.utils import logging as mlog

from ..s_to_u import zonedims_to_ngon, convert_s_to_ngon
from  .utils  import place_in_container

def compute_edge_measure(zone, comm):
  """ Compute the lenght of all edges of a 1D, 2D or 3D zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"

  if PT.Zone.Type(zone) == "Unstructured":
    global_distri = PT.Zone.CellDimension == 1
    edge_vtx_idx, edge_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 1, global_distri)

    dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)) if coords[i] is not None)
    vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

    part_data = EP.block_to_part(dist_coords, vtx_distri, [edge_vtx], comm)
    local_coords = [part_data[key][0] for key in part_data.keys()]

    # Compute lenght : |L| = ||x2 - x1||
    lenght = np.zeros(edge_vtx_idx.size-1)
    for dircoord in local_coords:
      lenght += (dircoord[1::2] - dircoord[0::2])**2
    return np.sqrt(lenght)
  else:
    raise NotImplementedError("Structured zones are not managed")

def compute_face_measure(zone, comm):
  """ Compute the area of all faces of a 2D or 3D distributed zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"

  # First, get face_vtx connectivity
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
      face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    else:
      global_distri = PT.Zone.CellDimension(zone) == 2
      face_vtx_idx, face_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 2, global_distri)

  # Get local coordinates
  dist_coords = dict((coords._fields[i], coords[i]) for i in range(len(coords)) if coords[i] is not None)
  vtx_distri = MT.getDistribution(zone, 'Vertex')[1]

  part_data = EP.block_to_part(dist_coords, vtx_distri, [face_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]
  local_coords_next = [np_utils.roll_once_by_stride(face_vtx_idx, coords) for coords in local_coords]

  if len(local_coords) == 2 : #We are in phydim==2, Add Z array
    local_coords.append(np.zeros_like(local_coords[0]))
    local_coords_next.append(np.zeros_like(local_coords[0]))

  # Compute area using cross product + triangulation from face meancenter
  _local_coords = np_utils.interweave_arrays(local_coords)
  _local_coords_next = np_utils.interweave_arrays(local_coords_next)
  _local_coords.shape        = (-1,3)
  _local_coords_next.shape   = (-1,3)

  face_vtx_n      = np.diff(face_vtx_idx)
  face_meancenter = np.add.reduceat(_local_coords, face_vtx_idx[:-1]) / face_vtx_n.reshape((-1,1))

  # |K| = ½ || sum_i CV_i ⨯ CV_{i+1}|| (C := face center)
  reps = np_utils.repeated_arange(face_vtx_n) # To access face center
  face_center_reps = face_meancenter[reps]
  crossprod = np.cross(_local_coords - face_center_reps, _local_coords_next - face_center_reps)
  # Sum per face
  normalflux = 0.5*np.add.reduceat(crossprod, face_vtx_idx[:-1])
  measure = np.linalg.norm(normalflux, axis=1)
  return measure

def _compute_face_circulation(vtx_distri, dist_coords, face_vtx_idx, face_vtx_n, face_vtx, comm):
  """
  Compute, for each face, the term xF.nF|F| where xF is the face mean center, nF the unit outward normal
  and |F| the area of the face.
  """
  # Get local coords corresponding to face_vtx
  part_data = EP.block_to_part(dist_coords._asdict(), vtx_distri, [face_vtx], comm)
  local_coords = [part_data[key][0] for key in part_data.keys()]
  local_coords_next = [np_utils.roll_once_by_stride(face_vtx_idx, coords) for coords in local_coords]

  _local_coords = np.stack(local_coords, axis=1)
  _local_coords_next = np.stack(local_coords_next, axis=1)
  center = np.add.reduceat(_local_coords, face_vtx_idx[:-1]) / face_vtx_n.reshape((-1,1))

  # Compute mean normal flux on each face : ½ || sum_i CV_i ⨯ CV_{i+1}|| (C := face center)
  reps = np_utils.repeated_arange(face_vtx_n) # To access face center
  face_center_reps = center[reps]
  crossprod = np.cross(_local_coords - face_center_reps, _local_coords_next - face_center_reps)
  normalflux = 0.5*np.add.reduceat(crossprod, face_vtx_idx[:-1])

  face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux
  return face_contrib

def _decompose_sections_to_face_vtx(zone):
  """
  Create a ngon like connectivity from 3D elements of a zone, but without face unification
  (face appears duplicated and cell_face connectivity is implicit)
  """
  all_face_vtx = []
  all_face_vtx_n = []
  all_cell_face_n = []
  for elt in PT.Zone.get_ordered_elements_per_dim(zone)[3]:
    ec = PT.get_child_from_name(elt, 'ElementConnectivity')[1]
    elt_distri = MT.getDistribution(elt, 'Element')[1]
    elt_kind = PT.Element.CGNSName(elt)
    n_elt = elt_distri[1] - elt_distri[0]

    if elt_kind == 'TETRA_4':
      base_n   = np.array([3,3,3,3], np.int32)
      base_seq = np.array([1,3,2, 1,2,4, 2,3,4, 3,1,4]) - 1
    if elt_kind == 'PYRA_5':
      base_n   = np.array([4,3,3,3,3], np.int32)
      base_seq = np.array([1,4,3,2, 1,2,5, 2,3,5, 3,4,5, 4,1,5]) - 1
    elif elt_kind == 'PENTA_6':
      base_n   = np.array([4,4,4,3,3], np.int32)
      base_seq = np.array([1,2,5,4, 2,3,6,5 ,3,1,4,6, 1,3,2, 4,5,6]) - 1
    elif elt_kind == 'HEXA_8':
      base_n   = np.array([4,4,4,4,4,4], np.int32)
      base_seq = np.array([1,4,3,2, 1,2,6,5 ,2,3,7,6, 3,4,8,7, 1,5,8,4, 5,6,7,8]) - 1

    face_vtx_n = np.tile(base_n, n_elt)
    # Where to read in element connectivity to reconstitute all faces (with reps)
    read_idx = np.tile(base_seq, n_elt) + np.repeat(PT.Element.NVtx(elt)*np.arange(n_elt), base_seq.size)
    face_vtx = ec[read_idx]

    all_face_vtx_n.append(face_vtx_n)
    all_face_vtx.append(face_vtx)
    all_cell_face_n.append(base_n.size*np.ones(n_elt, np.int32))

  face_vtx_n = np.concatenate(all_face_vtx_n)
  face_vtx   = np.concatenate(all_face_vtx)
  face_vtx_idx = np_utils.sizes_to_indices(face_vtx_n)
  cell_face_idx = np_utils.sizes_to_indices(np.concatenate(all_cell_face_n))
  return face_vtx_idx, face_vtx_n, face_vtx, cell_face_idx

def compute_cell_measure(zone, comm):
  """ Compute the volume of all cells of a 3D distributed zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  # Trick : if input zone is structured, convert it to unstructured so we can use same formulae
  if PT.Zone.Type(zone) == "Structured":
    _tree = PT.new_CGNSTree()
    _base = PT.new_CGNSBase(parent=_tree)
    _zone = PT.new_Zone(type='Structured', size=zone[1], parent=_base)
    PT.add_child(_zone, PT.get_child_from_label(zone, 'GridCoordinates_t'))
    PT.add_child(_zone, PT.get_child_from_name (zone, ':CGNS#Distribution'))
    convert_s_to_ngon(_tree, comm)
    zone = _zone

  assert PT.Zone.Type(zone) == 'Unstructured'
  if PT.Zone.has_ngon_elements(zone):

    ngon_node = PT.Zone.NGonNode(zone)
    face_vtx     = PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1]
    _face_vtx_idx = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
    face_vtx_idx = np.empty(_face_vtx_idx.size, np.int32)
    np.subtract(_face_vtx_idx, _face_vtx_idx[0], out=face_vtx_idx)
    face_vtx_n   = np.diff(face_vtx_idx)
    face_distri = MT.getDistribution(ngon_node, 'Element')[1]

    if not PT.Zone.has_nface_elements(zone):
      maia.algo.pe_to_nface(zone, comm)
    nface_node = PT.Zone.NFaceNode(zone)
    cell_face     = PT.get_child_from_name(nface_node, 'ElementConnectivity')[1]
    _cell_face_idx = PT.get_child_from_name(nface_node, 'ElementStartOffset')[1]
    cell_face_idx = np.empty(_cell_face_idx.size, np.int32)
    np.subtract(_cell_face_idx, _cell_face_idx[0], out=cell_face_idx)

    face_contrib = _compute_face_circulation(MT.getDistribution(zone, 'Vertex')[1], coords, face_vtx_idx, face_vtx_n, face_vtx, comm)
    # Assembly : for each cell, sum the quantities computed on each face
    face_contrib_loc = EP.block_to_part(face_contrib, face_distri, [np.abs(cell_face)], comm)[0]
    measure = (1/3.) * np.add.reduceat(np.sign(cell_face) * face_contrib_loc, cell_face_idx[:-1])

  else:
    # Compute center in current layout (section by section), then we will exchange to match 
    # cell distribution (we could probably do the opposite as well)
    face_vtx_idx, face_vtx_n, face_vtx, cell_face_idx = _decompose_sections_to_face_vtx(zone)
    face_contrib = _compute_face_circulation(MT.getDistribution(zone, 'Vertex')[1], coords, face_vtx_idx, face_vtx_n, face_vtx, comm)
    measure_elt = (1/3.) * np.add.reduceat(face_contrib, cell_face_idx[:-1])

    # Finally, move measure to allCell distribution (same method than _entity_vtx_connectivity_elt)
    distri_cell = MT.get_distribution(zone, 'Cell')[1]
    start = 0
    read_idx = 0
    measure_cell = []
    for elt in PT.Zone.get_ordered_elements_per_dim(zone)[3]:
      distri = MT.get_distribution(elt, 'Element')[1]
      dn_elt = distri[1] - distri[0]
      end = start + PT.Element.Size(elt)
      distri_out = distri.copy()
      # Here we restrict the total cell distribution to ElementRange (ignoring low order elts), 
      # then we shift it to make it start a 0
      distri_out[0] = max(min(distri_cell[0], end), start) - start
      distri_out[1] = max(min(distri_cell[1], end), start) - start
      this_elt_measure = measure_elt[read_idx : read_idx+dn_elt]
      measure_cell.append(EP.block_to_block(this_elt_measure, distri, distri_out, comm))
      read_idx += dn_elt
      start = end

    measure = np.concatenate(measure_cell)

  return measure


def _compute_zone_measures(zone, dim, comm):
  """Dispatch measures computing according to zone dimension and 
  requested dimension. Return a raw array"""
  if dim == 'CellCenter':
    dim = PT.Zone.CellDimension(zone)
  return {3: compute_cell_measure,
          2: compute_face_measure,
          1: compute_edge_measure}[dim](zone, comm)

def compute_zone_measures(zone, dim, comm):
  """ Implementation of maia.algo.compute_measures for a given distributed zone.
  See the calling function for full documentation """

  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  if cell_dim < rq_dim:
    msg = f"Zone '{PT.get_name(zone)}' skipped during measures computing because "\
          f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
    mlog.warning(msg)
  else:
    measure = _compute_zone_measures(zone, rq_dim, comm)
    place_in_container(zone, rq_dim, {'Measure' : measure}, comm)