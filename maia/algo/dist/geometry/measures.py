import numpy as np
import maia.pytree      as PT
import maia.pytree.maia as MT

import maia

from maia.algo.dist import connectivity_utils as CU
from maia.transfer  import protocols as EP

from maia.utils import np_utils
from maia.utils import logging as mlog
from maia.utils import vstride as vs

from ..s_to_u import zonedims_to_ngon, convert_s_to_ngon
from  .utils  import get_local_coordinates, place_in_container

from maia.algo.geometry_utils import ELT_FACE_VTX, compute_center_and_flux


def compute_edge_measure(zone, comm):
  """ Compute the length of all edges of a 1D, 2D or 3D zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"

  if PT.Zone.Type(zone) == "Unstructured":
    global_distri = PT.Zone.CellDimension == 1
    edge_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 1, global_distri)
  else:
    raise NotImplementedError("Structured zones are not managed")

  local_coords = get_local_coordinates(zone, edge_vtx.values, comm)

  # Compute length : |L| = ||x2 - x1||
  length = np.zeros(len(edge_vtx))
  for dircoord in local_coords:
    if dircoord is not None:
      length += (dircoord[1::2] - dircoord[0::2])**2
  return np.sqrt(length)

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
    face_vtx = MT.Element.connectivity(ngon_node)
  else:
    if PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      face_vtx = MT.Element.connectivity(ngon_node)
    else:
      global_distri = PT.Zone.CellDimension(zone) == 2
      face_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 2, global_distri)

  # Get local coordinates
  local_coords = get_local_coordinates(zone, face_vtx.values, comm)

  _, normalflux = compute_center_and_flux(local_coords, face_vtx.displs, face_vtx.counts)
  measure = np.linalg.norm(normalflux, axis=1)
  return measure

def _decompose_sections_to_face_vtx(zone):
  """
  Create a ngon like connectivity from 3D elements of a zone, but without face unification
  (face appears duplicated and cell_face connectivity is implicit)
  """
  
  all_cell_face_n = []
  all_face_vtx = []
  for elt in PT.Zone.get_ordered_elements_per_dim(zone)[3]:
    ec = PT.get_child_from_name(elt, 'ElementConnectivity')[1]
    elt_distri = MT.distribution_value(elt, 'Element')
    elt_kind = PT.Element.Type(elt)
    n_elt = elt_distri[1] - elt_distri[0]

    base_n, base_seq = ELT_FACE_VTX[elt_kind]

    face_vtx_n = np.tile(base_n, n_elt)
    # Where to read in element connectivity to reconstitute all faces (with reps)
    read_idx = np.tile(base_seq, n_elt) + np.repeat(PT.Element.NVtx(elt)*np.arange(n_elt), base_seq.size)
    face_vtx = ec[read_idx]

    all_face_vtx.append(vs.from_counts(face_vtx_n, face_vtx))
    all_cell_face_n.append(base_n.size*np.ones(n_elt, np.int32))

  face_vtx = vs.concatenate(all_face_vtx, vs.OUTER_AXIS)
  cell_face_idx = np_utils.sizes_to_indices(np.concatenate(all_cell_face_n))
  return face_vtx, cell_face_idx

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
    face_vtx = MT.Element.connectivity(ngon_node)
    face_distri = MT.distribution_value(ngon_node, 'Element')

    if not PT.Zone.has_nface_elements(zone):
      maia.algo.pe_to_nface(zone, comm)
    nface_node = PT.Zone.NFaceNode(zone)
    cell_face = MT.Element.connectivity(nface_node)

    local_coords = get_local_coordinates(zone, face_vtx.values, comm)
    center, normalflux = compute_center_and_flux(local_coords, face_vtx.displs, face_vtx.counts)
    face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux

    # Assembly : for each cell, sum the quantities computed on each face
    face_contrib_loc = EP.block_to_part(face_contrib, face_distri, np.abs(cell_face.values)-1, comm)
    face_contrib_loc = vs.from_displs(cell_face.displs, face_contrib_loc)
    measure = (1/3.) * (vs.sign(cell_face) * face_contrib_loc).reduce(vs.ReduceOp.SUM)

  else:
    # Compute center in current layout (section by section), then we will exchange to match 
    # cell distribution (we could probably do the opposite as well)
    face_vtx, cell_face_idx = _decompose_sections_to_face_vtx(zone)
    local_coords = get_local_coordinates(zone, face_vtx.values, comm)
    center, normalflux = compute_center_and_flux(local_coords, face_vtx.displs, face_vtx.counts)
    face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux
    face_contrib = vs.from_displs(cell_face_idx, face_contrib)
    measure_elt = (1/3.) * face_contrib.reduce(vs.ReduceOp.SUM)


    # Finally, move measure to allCell distribution (same method than _entity_vtx_connectivity_elt)
    distri_cell = MT.distribution_value(zone, 'Cell')
    start = 0
    read_idx = 0
    measure_cell = []
    for elt in PT.Zone.get_ordered_elements_per_dim(zone)[3]:
      distri = MT.distribution_value(elt, 'Element')
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


def _compute_elements_measure(zone, dim, comm):
  """Dispatch measures computing according to zone dimension and 
  requested dimension. Return a raw array"""
  if dim == 'CellCenter':
    dim = PT.Zone.CellDimension(zone)
  return {3: compute_cell_measure,
          2: compute_face_measure,
          1: compute_edge_measure}[dim](zone, comm)

def compute_elements_measure(zone, dim, comm):
  """ Implementation of maia.algo.compute_elements_measure for a given distributed zone.
  See the calling function for full documentation """

  cell_dim = PT.Zone.CellDimension(zone)
  rq_dim = cell_dim if dim == 'CellCenter' else dim
  if cell_dim < rq_dim:
    msg = f"Zone '{PT.get_name(zone)}' skipped during measures computing because "\
          f"its dimension is too low (cell_dim={cell_dim} < {rq_dim})"
    mlog.warning(msg)
  else:
    measure = _compute_elements_measure(zone, rq_dim, comm)
    place_in_container(zone, rq_dim, {'Measure' : measure}, comm)
