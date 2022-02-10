import numpy   as np

import Converter.Internal as I

import Pypdm.Pypdm as PDM

import maia.sids.Internal_ext  as IE

from maia                     import npy_pdm_gnum_dtype as pdm_dtype
from maia.sids                import sids
from maia.utils               import py_utils
from maia.tree_exchange       import utils as te_utils
from maia.transform.dist_tree import convert_s_to_u as S2U
from maia.transform.dist_tree import s_numbering_funcs      as s_numb

# ------------------------------------------------------------------------
def compute_gnum_from_parent_gnum(parent_gnum_l, comm):
  n_part = len(parent_gnum_l)
  gnum = PDM.GlobalNumbering(3, n_part, 0, 0., comm)
  for i_part, parent_gnum in enumerate(parent_gnum_l):
    gnum.gnum_set_from_parent(i_part, parent_gnum.shape[0], parent_gnum)
  gnum.gnum_compute()
  return [gnum.gnum_get(i_part)['gnum'] for i_part in range(n_part)]
# ------------------------------------------------------------------------

def _pr_to_face_pl(n_vtx_zone, pr, input_loc):
  """
  Transform a (partitioned) PointRange pr of any location input_loc into a PointList
  supported by the faces. n_vtx_zone is the number of vertices of the zone to which the
  pr belongs. Output face are numbered using s_numb conventions (i faces, then j faces, then
  k faces in increasing i,j,k for each group)
  """

  bnd_axis = S2U.guess_bnd_normal_index(pr, input_loc)

  # It is safer to reuse slabs to manage all cases (eg input location or reversed pr)
  bc_size = S2U.transform_bnd_pr_size(pr, input_loc, "FaceCenter")

  slab = np.empty((3,2), order='F', dtype=np.int64)
  slab[:,0] = pr[:,0]
  slab[:,1] = bc_size + pr[:,0] - 1
  slab[bnd_axis,:] += S2U.normal_index_shift(pr, n_vtx_zone, bnd_axis, input_loc, "FaceCenter")

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

  sub_array_idx = py_utils.sizes_to_indices(ends - starts)
  #This is the sub connectivity (only for sub_elts), but in old numbering
  sub_face_vtx = array[py_utils.multi_arange(starts, ends)]

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
  if sids.Zone.Type(zone) == 'Unstructured':
    face_vtx, face_vtx_idx, _ = sids.ngon_connectivity(zone)
  elif sids.Zone.Type(zone) == 'Structured':
    # For S zone, create a NGon connectivity
    n_vtx_zone = sids.Zone.VertexSize(zone)
    nf_i, nf_j, nf_k = S2U.n_face_per_dir(n_vtx_zone, n_vtx_zone-1)
    n_face_tot = nf_i + nf_j + nf_k
    face_distri = [0, n_face_tot]

    bounds = np.array([0, nf_i, nf_i + nf_j, nf_i + nf_j + nf_k], np.int32)

    face_vtx_idx = 4*np.arange(0, n_face_tot+1, dtype=pdm_dtype)
    face_vtx, _ = s_numb.ngon_dconnectivity_from_gnum(bounds+1, n_vtx_zone-1, dtype=pdm_dtype)

  ex_face_vtx_idx, ex_face_vtx, vtx_ids = _extract_sub_connectivity(face_vtx_idx, face_vtx, face_ids)
  
  # Vertex extraction
  cx, cy, cz = sids.coordinates(zone)
  if sids.Zone.Type(zone) == 'Unstructured':
    ex_cx = cx[vtx_ids-1]
    ex_cy = cy[vtx_ids-1]
    ex_cz = cz[vtx_ids-1]
  elif sids.Zone.Type(zone) == 'Structured':
    i_idx, j_idx, k_idx = s_numb.index_to_ijk(vtx_ids, n_vtx_zone)
    ex_cx = cx[i_idx-1, j_idx-1, k_idx-1].flatten()
    ex_cy = cy[i_idx-1, j_idx-1, k_idx-1].flatten()
    ex_cz = cz[i_idx-1, j_idx-1, k_idx-1].flatten()

  return ex_cx, ex_cy, ex_cz, ex_face_vtx_idx, ex_face_vtx, vtx_ids


def extract_surf_from_bc(part_zones, families, comm):
  """
  From a list of partitioned zones (coming from the same initial domain), get the list
  of faces belonging to a family whose name is in families list and extract the surfacic
  mesh.
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

    bc_nodes = sids.Zone.getBCsFromFamily(zone, families)
    if sids.Zone.Type(zone) == 'Unstructured':
      bc_face_ids = [I.getNodeFromName1(bc_node, 'PointList')[1][0] for bc_node in bc_nodes]
    else:
      n_vtx_z = sids.Zone.VertexSize(zone)
      bc_face_ids = [_pr_to_face_pl(n_vtx_z, I.getNodeFromName1(bc_node, 'PointRange')[1], sids.GridLocation(bc_node))[0] \
          for bc_node in bc_nodes]

    _, bc_face_ids = py_utils.concatenate_np_arrays(bc_face_ids, pdm_dtype)
    cx, cy, cz, bc_face_vtx_idx, bc_face_vtx, bc_vtx_ids = extract_faces_mesh(zone, bc_face_ids)

    ex_coords = py_utils.interweave_arrays([cx, cy, cz])
    bc_coords_l.append(ex_coords)
    bc_face_vtx_l.append(bc_face_vtx)
    bc_face_vtx_idx_l.append(bc_face_vtx_idx)

    vtx_ln_to_gn_zone, face_ln_to_gn_zone, _ = te_utils.get_entities_numbering(zone)

    parent_face_lngn_l.append(face_ln_to_gn_zone[bc_face_ids-1])
    parent_vtx_lngn_l .append(vtx_ln_to_gn_zone[bc_vtx_ids-1]  )

  # Compute extracted gnum from parents
  bc_face_lngn_l = compute_gnum_from_parent_gnum(parent_face_lngn_l, comm)
  bc_vtx_lngn_l  = compute_gnum_from_parent_gnum(parent_vtx_lngn_l, comm)

  return bc_face_vtx_l, bc_face_vtx_idx_l, bc_face_lngn_l, bc_coords_l, bc_vtx_lngn_l

