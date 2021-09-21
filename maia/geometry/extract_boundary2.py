import numpy   as np

import Converter.Internal as I

import Pypdm.Pypdm as PDM

import maia.sids.Internal_ext  as IE
from maia                    import npy_pdm_gnum_dtype as pdm_dtype
from maia.sids               import sids               as SIDS
from maia.utils              import py_utils
from maia.transform.dist_tree import convert_s_to_u as S2U
from maia.transform.dist_tree import s_numbering_funcs      as s_numb

# ------------------------------------------------------------------------
def compute_gnum_from_parent_gnum(bcs_ln_to_gn, comm):
  gnum = PDM.GlobalNumbering(3, 1, 0, 0., comm)
  bcs_ln_to_gn_pdm = bcs_ln_to_gn.astype(pdm_dtype)
  gnum.gnum_set_from_parent(0, bcs_ln_to_gn.shape[0], bcs_ln_to_gn_pdm)
  gnum.gnum_compute()
  return gnum.gnum_get(0)['gnum']

def compute_gnum_from_parent_gnum2(parent_gnum_l, comm):
  n_part = len(parent_gnum_l)
  gnum = PDM.GlobalNumbering(3, n_part, 0, 0., comm)
  for i_part, parent_gnum in enumerate(parent_gnum_l):
    gnum.gnum_set_from_parent(i_part, parent_gnum.shape[0], parent_gnum)
  gnum.gnum_compute()
  return [gnum.gnum_get(i_part)['gnum'] for i_part in range(n_part)]
# ------------------------------------------------------------------------

def _pr_to_face_pl(n_vtx_zone, pr, input_loc):

  bnd_axis = S2U.guess_bnd_normal_index(pr, input_loc)

  # It is safer to reuse slabs to manage all cases (eg input location or reversed pr)
  bc_size = S2U.transform_bnd_pr_size(pr, input_loc, "FaceCenter")

  slab = np.empty((3,2), order='F', dtype=np.int)
  slab[:,0] = pr[:,0]
  slab[:,1] = bc_size + pr[:,0] - 1
  slab[bnd_axis,:] += S2U.normal_index_shift(pr, n_vtx_zone, bnd_axis, input_loc, "FaceCenter")

  return S2U.compute_pointList_from_pointRanges([slab], n_vtx_zone, 'FaceCenter', bnd_axis)

def _extract_face_connectivity(face_vtx_idx, face_vtx, face_ids):
  """
  """
  starts = face_vtx_idx[face_ids - 1]
  ends   = face_vtx_idx[face_ids - 1 + 1]

  ex_face_vtx_idx = py_utils.sizes_to_indices(ends - starts)
  #This is the face->vtx of face_ids, but in old numbering
  sub_face_vtx = face_vtx[py_utils.multi_arange(starts, ends)]

  #Get the udpated face->vtx, with local numbergin new numbering
  vtx_ids, ex_face_vtx = np.unique(sub_face_vtx, return_inverse=True)
  ex_face_vtx = ex_face_vtx.astype(face_vtx.dtype) + 1

  return ex_face_vtx_idx, ex_face_vtx, vtx_ids


def extract_faces_mesh(zone, face_ids):
  """
  """
  # NGon Extraction
  if SIDS.Zone.Type(zone) == 'Unstructured':
    face_vtx, face_vtx_idx, _ = SIDS.ngon_connectivity(zone)
  elif SIDS.Zone.Type(zone) == 'Structured':
    # For S zone, create a NGon connectivity
    n_vtx_zone = SIDS.Zone.VertexSize(zone)
    nf_i, nf_j, nf_k = S2U.n_face_per_dir(n_vtx_zone, n_vtx_zone-1)
    n_face_tot = nf_i + nf_j + nf_k
    face_distri = [0, n_face_tot]

    bounds = np.array([0, nf_i, nf_i + nf_j, nf_i + nf_j + nf_k], np.int32)

    face_vtx_idx = 4*np.arange(0, n_face_tot+1, dtype=pdm_dtype)
    face_vtx, _ = s_numb.ngon_dconnectivity_from_gnum(bounds+1, n_vtx_zone-1, dtype=pdm_dtype)

  ex_face_vtx_idx, ex_face_vtx, vtx_ids = _extract_face_connectivity(face_vtx_idx, face_vtx, face_ids)
  
  # Vertex extraction
  cx, cy, cz = SIDS.coordinates(zone)
  if SIDS.Zone.Type(zone) == 'Unstructured':
    ex_cx = cx[vtx_ids-1]
    ex_cy = cy[vtx_ids-1]
    ex_cz = cz[vtx_ids-1]
  elif SIDS.Zone.Type(zone) == 'Structured':
    i_idx, j_idx, k_idx = s_numb.index_to_ijk(vtx_ids, n_vtx_zone)
    ex_cx = cx[i_idx-1, j_idx-1, k_idx-1].flatten()
    ex_cy = cy[i_idx-1, j_idx-1, k_idx-1].flatten()
    ex_cz = cz[i_idx-1, j_idx-1, k_idx-1].flatten()

  return ex_cx, ex_cy, ex_cz, ex_face_vtx_idx, ex_face_vtx, vtx_ids


def extract_surf_from_bc_new(part_zones, families, comm):

  bc_face_vtx_l     = []
  bc_face_vtx_idx_l = []
  bc_coords_l       = []
  parent_face_lngn_l = []
  parent_vtx_lngn_l  = []
  for zone in part_zones:

    bc_nodes = SIDS.Zone.getBCsFromFamily(zone, families)
    if SIDS.Zone.Type(zone) == 'Unstructured':
      bc_face_ids = [I.getNodeFromName1(bc_node, 'PointList')[1][0] for bc_node in bc_nodes]
    else:
      n_vtx_z = SIDS.Zone.VertexSize(zone)
      bc_face_ids = [_pr_to_face_pl(n_vtx_z, I.getNodeFromName1(bc_node, 'PointRange')[1], SIDS.GridLocation(bc_node))[0] \
          for bc_node in bc_nodes]

    _, bc_face_ids = py_utils.concatenate_np_arrays(bc_face_ids)
    cx, cy, cz, bc_face_vtx_idx, bc_face_vtx, bc_vtx_ids = extract_faces_mesh(zone, bc_face_ids)

    ex_coords = py_utils.interweave_arrays([cx, cy, cz])
    bc_coords_l.append(ex_coords)
    bc_face_vtx_l.append(bc_face_vtx)
    bc_face_vtx_idx_l.append(bc_face_vtx_idx)

    vtx_ln_to_gn_zone   = I.getVal(IE.getGlobalNumbering(zone, 'Vertex')).astype(pdm_dtype)
    if SIDS.Zone.Type(zone) == "Structured":
      face_ln_to_gn_zone = I.getVal(IE.getGlobalNumbering(zone, 'Face')).astype(pdm_dtype)
    else:
      ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if SIDS.ElementCGNSName(e) == 'NGON_n']
      assert len(ngons) == 1, "For unstructured zones, only NGon connectivity is supported"
      face_ln_to_gn_zone = I.getVal(IE.getGlobalNumbering(ngons[0], 'Element')).astype(pdm_dtype)

    parent_face_lngn_l.append(face_ln_to_gn_zone[bc_face_ids-1])
    parent_vtx_lngn_l .append(vtx_ln_to_gn_zone[bc_vtx_ids-1]  )

  # Compute extracted gnum from parents
  bc_face_lngn_l = compute_gnum_from_parent_gnum2(parent_face_lngn_l, comm)
  bc_vtx_lngn_l  = compute_gnum_from_parent_gnum2(parent_vtx_lngn_l, comm)



    #return all_connect_l, all_bc_v_vtxidx, face_ln_to_gn, ex_coords, vtx_ln_to_gn
  return bc_face_vtx_l, bc_face_vtx_idx_l, bc_face_lngn_l, bc_coords_l, bc_vtx_lngn_l

def extract_surf_from_bc_old(part_tree, families, comm):

  import time
  assert len(I.getZones(part_tree)) == 1

  all_bc_f_lngn = []

  for zone in I.getZones(part_tree):
    debut = time.time()
    vtx_ln_to_gn_zone  = I.getVal(IE.getGlobalNumbering(zone, 'Vertex'))
    ngons  = [e for e in I.getNodesFromType1(zone, 'Elements_t') if SIDS.ElementCGNSName(e) == 'NGON_n']
    assert len(ngons) == 1, "For unstructured zones, only NGon connectivity is supported"
    face_ln_to_gn_zone = I.getVal(IE.getGlobalNumbering(ngons[0], 'Element'))

    face_vtx, face_vtx_idx, _ = SIDS.ngon_connectivity(zone)
    cx, cy, cz = SIDS.coordinates(zone)

    vtx_new_id = -np.ones(SIDS.Zone.n_vtx(zone), np.int32)
    is_treated = np.zeros(SIDS.Zone.n_vtx(zone), bool)

    pl_vtx_l = []
    all_connect_l = []
    all_connect_size = []
    ivtx = 0
    end = time.time()
    print("Init ", end - debut)

    marange_time = 0
    debut = time.time()
    for bc_node in SIDS.Zone.getBCsFromFamily(zone, families):
      point_list = I.getNodeFromName1(bc_node, 'PointList')[1]
      all_bc_f_lngn.append(point_list[0])

      starts = face_vtx_idx[point_list[0] - 1]
      ends   = face_vtx_idx[point_list[0] - 1 + 1]
      point_list_v_idx = ends - starts
      all_connect_size.append(point_list_v_idx)
      t1 = time.time()
      point_list_v = face_vtx[py_utils.multi_arange(starts, ends)]
      marange_time += time.time() - t1


      for vtx in point_list_v:
        if vtx_new_id[vtx-1] == -1:
          vtx_new_id[vtx-1] = ivtx + 1
          ivtx += 1
      all_connect_l.append(vtx_new_id[point_list_v-1])

    end = time.time()
    print("Loop 1 ", end - debut)
    print("Multi arange", marange_time)


    debut = time.time()
    #Extract coords
    filtered_vtx_new_id = vtx_new_id[(vtx_new_id > -1)]
    argsort = np.argsort(filtered_vtx_new_id)
    ex_cx = cx[(vtx_new_id > -1)][argsort]
    ex_cy = cy[(vtx_new_id > -1)][argsort]
    ex_cz = cz[(vtx_new_id > -1)][argsort]
    end = time.time()
    print("Extraction ", end - debut)

  debut = time.time()
  ex_coords = py_utils.interweave_arrays([ex_cx, ex_cy, ex_cz])
  all_bc_v_vtxidx = py_utils.sizes_to_indices(np.concatenate(all_connect_size))
  all_connect_l = np.concatenate(all_connect_l)
  end = time.time()
  print("Interweave + concat ", end - debut)


  debut = time.time()
  all_bc_f_lngn = py_utils.concatenate_np_arrays(all_bc_f_lngn, pdm_dtype)[1]
  all_bc_f_lngn = face_ln_to_gn_zone[all_bc_f_lngn-1]
  face_ln_to_gn = compute_gnum_from_parent_gnum(all_bc_f_lngn)

  all_bc_v_lngn = vtx_ln_to_gn_zone[np.where(vtx_new_id > - 1)[0]].astype(pdm_dtype)[argsort]
  vtx_ln_to_gn = compute_gnum_from_parent_gnum(all_bc_v_lngn)
  end = time.time()
  print("lngns ", end - debut)

  rank = comm.rank
  # np.savetxt(f'coords_{rank}.dat', ex_coords)
  # np.savetxt(f'face_lngn_{rank}.dat', face_ln_to_gn, fmt='%i')
  # np.savetxt(f'vtx_lngn_{rank}.dat', vtx_ln_to_gn, fmt='%i')
  # np.savetxt(f'face_vtx_bcs_{rank}.dat', all_connect_l, fmt='%i')
  # np.savetxt(f'face_vtx_idx_bcs_{rank}.dat', all_bc_v_vtxidx, fmt='%i')

  assert np.min(all_connect_l) > -1
  return all_connect_l, all_bc_v_vtxidx, face_ln_to_gn, ex_coords, vtx_ln_to_gn

