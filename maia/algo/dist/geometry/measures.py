import numpy as np
import maia.pytree      as PT
import maia.pytree.maia as MT

import Pypdm.Pypdm as PDM

import maia

from maia.algo.dist import connectivity_utils as CU
from maia.transfer  import protocols as EP

from maia.utils import np_utils, s_numbering
from maia.utils import logging as mlog
from maia.utils import vstride as vs

from ..s_to_u import zonedims_to_ngon, convert_s_to_ngon
from  .utils  import get_local_coordinates, place_in_container

from maia.algo.geometry_utils import ELT_FACE_VTX, compute_center_and_flux


def compute_edge_measure(zone, comm, edge_indices=None):
  """ Compute the length of all edges of a 1D, 2D or 3D zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"

  if PT.Zone.Type(zone) == "Unstructured":
    global_distri = PT.Zone.CellDimension == 1
    _edge_indices = edge_indices[0] if edge_indices is not None else None
    edge_vtx = CU.entity_vtx_connectivity_elt(zone, comm, 1, global_distri, _edge_indices)
  else:
    raise NotImplementedError("Structured zones are not managed")

  local_coords = get_local_coordinates(zone, edge_vtx.values, comm)

  # Compute length : |L| = ||x2 - x1||
  length = np.zeros(len(edge_vtx))
  for dircoord in local_coords:
    if dircoord is not None:
      length += (dircoord[1::2] - dircoord[0::2])**2
  return np.sqrt(length)

def compute_face_measure(zone, comm, face_indices=None, face_indices_loc=None):
  """ Compute the area of all faces of a 2D or 3D distributed zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  zone_dim = PT.Zone.CellDimension(zone)
  assert zone_dim >= 2, "CellDimension of zone must be >= 2 to compute face centers"

  if face_indices is not None:
    assert isinstance(face_indices, np.ndarray) and face_indices.ndim == 2
    if PT.Zone.Type(zone) == 'Structured' and zone_dim == 3:
      assert face_indices_loc in ['IFaceCenter', 'JFaceCenter', 'KFaceCenter'], \
        "Indices location must be specified when filtering faces on 3D structured meshes"

  # First, get face_vtx connectivity
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
        _face_indices = s_numbering.ijk_to_index_from_loc(*face_indices, face_indices_loc, PT.Zone.VertexSize(zone)) - 1
    elif PT.Zone.Type(zone) == "Unstructured" and PT.Zone.has_ngon_elements(zone):
      ngon_node = PT.Zone.NGonNode(zone)
      if face_indices is not None:
        _face_indices = face_indices[0] - PT.Element.Range(ngon_node)[0]

    face_vtx = MT.Element.connectivity(ngon_node)
    if face_indices is not None:
      face_distri = MT.distribution_value(ngon_node, 'Element')
      face_vtx = EP.block_to_part(face_vtx, face_distri, _face_indices, comm)


  # Get local coordinates
  local_coords = get_local_coordinates(zone, face_vtx.values, comm)

  _, normalflux = compute_center_and_flux(local_coords, face_vtx.displs, face_vtx.counts)
  measure = np.linalg.norm(normalflux, axis=1)
  return measure

def _decompose_section_to_face_vtx(elt, elt_mask_loc=None):
  """
  Create a ngon like connectivity from a 3D elements section, but without face unification
  (face appears duplicated and cell_face connectivity is implicit)
  If elt_mask_loc is provided, it must be a bool array (distributed as elt distribution)
  """
      
  ec = PT.get_np_value(PT.find_child_from_name(elt, 'ElementConnectivity'))
  elt_distri = MT.distribution_value(elt, 'Element')
  elt_kind = PT.Element.Type(elt)
  n_elt = elt_distri[1] - elt_distri[0]

  if elt_mask_loc is not None:
    elt_loc_range = np.arange(n_elt)[elt_mask_loc]
    n_elt = elt_loc_range.size # Update nb of element -> only selected
  else:
    elt_loc_range = np.arange(n_elt)

  base_n, base_seq = ELT_FACE_VTX[elt_kind]

  face_vtx_n = np.tile(base_n, n_elt)
  # Where to read in element connectivity to reconstitute all faces (with reps)
  read_idx = np.tile(base_seq, n_elt) + np.repeat(PT.Element.NVtx(elt)*elt_loc_range, base_seq.size)
  face_vtx_val = ec[read_idx]

  face_vtx = vs.from_counts(face_vtx_n, face_vtx_val)

  return face_vtx

def compute_cell_measure(zone, comm, cell_indices=None):
  """ Compute the volume of all cells of a 3D distributed zone and return a raw array"""
  coords = PT.Zone.coordinates(zone)
  assert isinstance(coords, PT.CartesianCoordinates), "Only cartesian coordinates are supported"
  assert PT.Zone.CellDimension(zone) == 3, "CellDimension of zone must be == 3 to compute cell centers"

  cell_distri = MT.distribution_value(zone, 'Cell')
  dn_cell = cell_distri[1] - cell_distri[0]

  if cell_indices is not None:
    assert isinstance(cell_indices, np.ndarray) and cell_indices.ndim == 2

  # Trick : if input zone is structured, convert it to unstructured so we can use same formulae
  if PT.Zone.Type(zone) == "Structured":
    if cell_indices is not None: # Convert indices as well (if any)
      cell_indices = s_numbering.ijk_to_index(*cell_indices, PT.Zone.CellSize(zone)).reshape((1,-1))
      cell_indices += PT.Zone.n_face(zone) # Offset with nface, since on NGon faces are first
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

    if cell_indices is not None:
      # Filter cell_face to keep only appearing cells
      cell_GI = EP.GlobalIndexer(cell_distri, cell_indices[0]-PT.Element.Range(nface_node)[0], comm)
      cell_selector = cell_GI.access_counts > 0
      cell_face = vs.take(cell_face, np.flatnonzero(cell_selector))
      # Filter face_vtx to keep only filtered faces
      face_GI = EP.GlobalIndexer(face_distri, np.abs(cell_face.values)-PT.Element.Range(ngon_node)[0], comm)
      face_selector = face_GI.access_counts > 0
      face_vtx = vs.take(face_vtx, np.flatnonzero(face_selector))
      # Compute face flux on selected faces (+selected vtx)
      local_coords = get_local_coordinates(zone, face_vtx.values, comm)
      center, normalflux = compute_center_and_flux(local_coords, face_vtx.displs, face_vtx.counts)
      face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux
      # Send back face flux to cells in which faces appears to compute measure
      # (this is a Take_v since removed faces have no value for flux)
      _, face_contrib_loc = face_GI.Take_v((face_selector.astype(np.int32), face_contrib))
      face_contrib_loc = vs.from_displs(cell_face.displs, face_contrib_loc)
      measure = (1/3.) * (vs.sign(cell_face) * face_contrib_loc).reduce(vs.ReduceOp.SUM)
      # Send back cell measure to requesting rank throught cell_indices (again Take_v)
      _, measure = cell_GI.Take_v((cell_selector.astype(np.int32), measure))

    else:

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
    all_elt_mask = None
    volumic_sections = PT.Zone.get_ordered_elements_per_dim(zone)[3]
    if cell_indices is not None:

      part2 = []
      all_elt_mask = []
      for elt in volumic_sections:
        elt_distri = MT.distribution_value(elt, 'Element')
        elt_offset = PT.Element.Range(elt)[0]
        part2.append(np.arange(elt_distri[0], elt_distri[1]) + elt_offset)
        all_elt_mask.append(np.zeros(elt_distri[1]-elt_distri[0], bool))
        
      ptp = EP.PartToPart([cell_indices[0]], part2, comm)
      # We use the part to part to easily find the ids of elements appearing in cell_indices
      for elt_mask, ref_lnum2 in zip(all_elt_mask, ptp.get_referenced_lnum2()):
        elt_mask[ref_lnum2-1] = True

    all_measure_elt = []
    for i,elt in enumerate(volumic_sections):
      # Work section by section
      elt_mask = all_elt_mask[i] if all_elt_mask is not None else None
      face_vtx = _decompose_section_to_face_vtx(elt, elt_mask)
      local_coords = get_local_coordinates(zone, face_vtx.values, comm)
      center, normalflux = compute_center_and_flux(local_coords, face_vtx.displs, face_vtx.counts)
      face_contrib = np.sum(center*normalflux, axis=1) # Scalar product face_center * normal_flux
      face_contrib = vs.from_counts(ELT_FACE_VTX[PT.Element.Type(elt)][0].size, face_contrib)
      measure_elt = (1/3.) * face_contrib.reduce(vs.ReduceOp.SUM)
      all_measure_elt.append(measure_elt)

    if cell_indices is not None:
      # In partial case, we use part to part to directly fetch relevant data
      req = ptp.reverse_iexch(PDM._PDM_MPI_COMM_KIND_P2P, 
                              PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART2, 
                              all_measure_elt, 
                              [t.astype(np.int32) for t in all_elt_mask])
      _, out = ptp.reverse_wait(req)
      measure = out[0] # Only one part

    else:
      # In full case, we move measure to allCell distribution (same method than _entity_vtx_connectivity_elt)
      distri_cell = MT.distribution_value(zone, 'Cell')
      start = 0
      measure_cell = []
      for elt_measure, elt in zip(all_measure_elt, volumic_sections):
        distri = MT.distribution_value(elt, 'Element')
        end = start + PT.Element.Size(elt)
        distri_out = distri.copy()
        # Here we restrict the total cell distribution to ElementRange (ignoring low order elts), 
        # then we shift it to make it start a 0
        distri_out[0] = max(min(distri_cell[0], end), start) - start
        distri_out[1] = max(min(distri_cell[1], end), start) - start
        measure_cell.append(EP.block_to_block(elt_measure, distri, distri_out, comm))
        start = end

      measure = np.concatenate(measure_cell)

  return measure


def _compute_elements_measure(zone, dim, comm, element_indices=None, element_loc=None):
  """Dispatch measures computing according to zone dimension and 
  requested dimension (1,2,3 or 'CellCenter').

  If element_indices is None, measure is computed for all elements
  of relevant dimension of the grid (distributed)
  Otherwise, a PointList-like array is expected: measure will be computed
  only for the specified indices. Indices must be provided in absolute 'cgns numbering',
  (ie. refering to ElementRange_t ids, independantly of element dimension).
  In addition, element_loc is mandatory when filtering faces (resp edges) on 
  3D/S (resp. 2D/S) meshes, to specify if faces (resp. edges) are in I,J, or K
  direction (using IFaceCenter, JFaceCenter, ... JEdgeCenter value).
  
  Return a raw array"""
  if dim == 'CellCenter':
    dim = PT.Zone.CellDimension(zone)
  if dim == 3:
    return compute_cell_measure(zone, comm, element_indices)
  elif dim == 2:
    return compute_face_measure(zone, comm, element_indices, element_loc)
  elif dim == 1:
    return compute_edge_measure(zone, comm, element_indices)

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
