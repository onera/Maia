import numpy   as np

import maia.pytree        as PT

from maia.utils     import np_utils, s_numbering, pr_utils
from maia.transfer  import utils as te_utils
from maia.algo.dist import s_to_u as S2U

from .point_cloud_utils import create_sub_numbering

def _pr_to_face_pl(n_vtx_zone, pr, input_loc):
  """
  Transform a (partitioned) PointRange pr of any location input_loc into a PointList
  supported by the faces. n_vtx_zone is the number of vertices of the zone to which the
  pr belongs. Output face are numbered using s_numb conventions (i faces, then j faces, then
  k faces in increasing i,j,k for each group)
  """

  bnd_axis = PT.Subset.normal_axis(PT.new_BC(point_range=pr, loc=input_loc))

  # It is safer to reuse slabs to manage all cases (eg input location or reversed pr)
  bc_size = pr_utils.transform_bnd_pr_size(pr, input_loc, "FaceCenter")

  slab = np.empty((3,2), order='F', dtype=np.int32)
  slab[:,0] = pr[:,0]
  slab[:,1] = bc_size + pr[:,0] - 1
  slab[bnd_axis,:] += pr_utils.normal_index_shift(pr, n_vtx_zone, bnd_axis, input_loc, "FaceCenter")

  return S2U.compute_pointList_from_pointRanges([slab], n_vtx_zone, 'FaceCenter', bnd_axis)

def _extract_sub_connectivity(array_idx, array, sub_elts):
  """
  From an idx+array mother->child connectivity (eg face->vtx or cell->face) and a list of
  mother element ids (starting at 1), create a sub connectivity involving only these mothers.
  Return the new idx+array, where child element are renumbered from 1 to nb (unique) childs, without
  hole. In addition, return the childs_ids array containing the new_to_old indirection for child ids.
  """
  starts = array_idx[sub_elts - 1]
  ends   = array_idx[sub_elts - 1 + 1]

  sub_array_idx = np_utils.sizes_to_indices(ends - starts)
  #This is the sub connectivity (only for sub_elts), but in old numbering
  sub_face_vtx = array[np_utils.multi_arange(starts, ends)]

  #Get the udpated connectivity with local numbering
  child_ids, sub_array = np.unique(sub_face_vtx, return_inverse=True)
  sub_array = sub_array.astype(array.dtype) + 1

  return sub_array_idx, sub_array, child_ids


def extract_faces_mesh(zone, face_ids):
  """
  Extract a sub mesh from a U or S zone and a (flat) list of face ids to extract :
  create the sub ngon connectivity and extract the coordinates of vertices 
  belonging to the sub mesh.
  For S zone, faces to extract must be converted from i,j,k to index before processing
  """
  # NGon Extraction
  if PT.Zone.Type(zone) == 'Unstructured':
    if PT.Zone.has_ngon_elements(zone):
      face_vtx_idx, face_vtx, _ = PT.Zone.ngon_connectivity(zone)
    else: # Zone has std elements
      sections_2d = PT.Zone.get_ordered_elements_per_dim(zone)[2]
      elem_size_list = [PT.Element.Size(elt) for elt in sections_2d]
      face_n_vtx_list = [PT.Element.NVtx(elt) for elt in sections_2d]
      _, face_vtx = np_utils.concatenate_np_arrays([PT.get_node_from_name(elt, 'ElementConnectivity')[1] for elt in sections_2d])
      face_vtx_idx = np_utils.sizes_to_indices(np.repeat(face_n_vtx_list, elem_size_list), dtype=np.int32)
  elif PT.Zone.Type(zone) == 'Structured':
    # For S zone, create a NGon connectivity
    n_vtx_zone = PT.Zone.VertexSize(zone)
    nf_i, nf_j, nf_k = PT.Zone.FaceSize(zone)
    n_face_tot = nf_i + nf_j + nf_k

    bounds = np.array([0, nf_i, nf_i + nf_j, nf_i + nf_j + nf_k], np.int32)

    face_vtx_idx = 4*np.arange(0, n_face_tot+1, dtype=np.int32)
    face_vtx, _ = s_numbering.ngon_dconnectivity_from_gnum(bounds+1, n_vtx_zone-1, dtype=np.int32)

  ex_face_vtx_idx, ex_face_vtx, vtx_ids = _extract_sub_connectivity(face_vtx_idx, face_vtx, face_ids)
  
  # Vertex extraction
  cx, cy, cz = PT.Zone.coordinates(zone)
  if PT.Zone.Type(zone) == 'Unstructured':
    ex_cx = cx[vtx_ids-1]
    ex_cy = cy[vtx_ids-1]
    ex_cz = cz[vtx_ids-1]
  elif PT.Zone.Type(zone) == 'Structured':
    i_idx, j_idx, k_idx = s_numbering.index_to_ijk(vtx_ids, n_vtx_zone)
    ex_cx = cx[i_idx-1, j_idx-1, k_idx-1].flatten()
    ex_cy = cy[i_idx-1, j_idx-1, k_idx-1].flatten()
    ex_cz = cz[i_idx-1, j_idx-1, k_idx-1].flatten()

  return ex_cx, ex_cy, ex_cz, ex_face_vtx_idx, ex_face_vtx, vtx_ids


def extract_surf_from_bc(part_zones, bc_predicate, comm):
  """
  From a list of partitioned zones (coming from the same initial domain), get the list
  of faces belonging to any bc satisfiyng bc_predicate and extract the surfacic mesh.
  In addition, compute a new global numbering (over the procs and the part_zones) of the extracted
  faces and vertex (starting a 1 without gap)

  Return lists (of size n_part) of sub face_vtx connectivity, sub vtx coordinates and global numberings
  """

  bc_face_vtx_l     = []
  bc_face_vtx_idx_l = []
  bc_coords_l       = []
  parent_face_lngn_l = []
  parent_vtx_lngn_l  = []
  for zone in part_zones:
    is_relevant_bc = lambda n: PT.get_label(n) == 'BC_t' and bc_predicate(n)
    if PT.Zone.Type(zone) == 'Unstructured':
      bc_nodes = PT.get_children_from_predicates(zone, ['ZoneBC_t', lambda n: is_relevant_bc(n) and PT.Subset.GridLocation(n) == 'FaceCenter'])
      bc_face_ids = [PT.get_child_from_name(bc_node, 'PointList')[1][0] for bc_node in bc_nodes]
    else:
      n_vtx_z = PT.Zone.VertexSize(zone)
      bc_nodes = PT.get_children_from_predicates(zone, ['ZoneBC_t', is_relevant_bc])
      bc_face_ids = [_pr_to_face_pl(n_vtx_z, PT.get_child_from_name(bc_node, 'PointRange')[1], PT.Subset.GridLocation(bc_node))[0] \
          for bc_node in bc_nodes]

    _, bc_face_ids = np_utils.concatenate_np_arrays(bc_face_ids, np.int32)
    # If zone is by elements, we must shift the bc_face_ids to make it start a 1
    if PT.Zone.Type(zone) == 'Unstructured' and not PT.Zone.has_ngon_elements(zone):
      ordering = PT.Zone.elt_ordering_by_dim(zone)
      if ordering == 1: #Increasing elements : substract starting point of 2D
        bc_face_ids -= (PT.Zone.get_elt_range_per_dim(zone)[2][0] - 1)
      elif ordering == -1: #Decreasing elements : substract number of 3D
        bc_face_ids -= PT.Zone.get_elt_range_per_dim(zone)[3][1]
      else:
        raise RuntimeError("Unable to extract unordered faces")

    cx, cy, cz, bc_face_vtx_idx, bc_face_vtx, bc_vtx_ids = extract_faces_mesh(zone, bc_face_ids)

    ex_coords = np_utils.interweave_arrays([cx, cy, cz])
    bc_coords_l.append(ex_coords)
    bc_face_vtx_l.append(bc_face_vtx)
    bc_face_vtx_idx_l.append(bc_face_vtx_idx)

    vtx_ln_to_gn_zone = PT.maia.getGlobalNumbering(zone, 'Vertex')[1]

    if PT.Zone.Type(zone) == 'Unstructured' and not PT.Zone.has_ngon_elements(zone):
      face_ln_to_gn_zone = np.concatenate([PT.maia.getGlobalNumbering(elt, "Sections")[1] \
              for elt in PT.Zone.get_ordered_elements_per_dim(zone)[2]])
    else:
      _, _, face_ln_to_gn_zone, _ = te_utils.get_entities_numbering(zone) # !! Only S or NGON

    parent_face_lngn_l.append(face_ln_to_gn_zone[bc_face_ids-1])
    parent_vtx_lngn_l .append(vtx_ln_to_gn_zone[bc_vtx_ids-1]  )

  # Compute extracted gnum from parents
  bc_face_lngn_l = create_sub_numbering(parent_face_lngn_l, comm)
  bc_vtx_lngn_l  = create_sub_numbering(parent_vtx_lngn_l, comm)

  return bc_face_vtx_l, bc_face_vtx_idx_l, bc_face_lngn_l, bc_coords_l, bc_vtx_lngn_l

