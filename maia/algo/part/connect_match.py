import numpy              as NPY
import Pypdm.Pypdm        as PDM

import maia.pytree        as PT

from cmaia.part_algo import compute_face_center_and_characteristic_length, adapt_match_information

def compute_n_point_cloud(zones, family_list):
  """
  """
  n_point_cloud = 0
  for zone in zones:
    if PT.Zone.Type(zone) == 'Structured':
      raise NotImplementedError("connect_match_from_family for structured zone not allowed yet")
    for bc in PT.Zone.getBCsFromFamily(zone, family_list):
      n_point_cloud = n_point_cloud + 1

  return n_point_cloud

def prepare_pdm_point_merge_structured(pdm_point_merge, i_point_cloud, match_type,
                                       i_zone, zone,
                                       bnd_to_join_path_list,
                                       l_send_entity_stri,
                                       l_send_entity_data,
                                       l_send_zone_id_data):
  """
  """
  raise NotImplementedError("connect_match_from_family for structured zone not allowed yet")


def prepare_pdm_point_merge_unstructured(pdm_point_merge, i_point_cloud, match_type,
                                         i_zone, zone, family_list,
                                         bnd_to_join_path_list,
                                         l_send_entity_stri,
                                         l_send_entity_data,
                                         l_send_zone_id_data):
  """
  match_type can be vertex or face
  """
  cx, cy, cz = PT.Zone.coordinates(zone)

  face_vtx_idx, face_vtx, _ = PT.Zone.ngon_connectivity(zone)

  for bc in PT.Zone.getBCsFromFamily(zone, family_list):

    pl = PT.get_child_from_name(bc, 'PointList')[1]

    l_send_entity_data .append(pl[0,:])
    l_send_entity_stri .append(NPY.ones(pl[0,:].shape, dtype='int32'))
    l_send_zone_id_data.append(NPY.full(pl[0,:].shape, i_zone, dtype='int32'))

    bnd_xyz, bnd_cl = compute_face_center_and_characteristic_length(pl, cx, cy, cz, face_vtx, face_vtx_idx)

    # print("Setup caracteristice lenght and coordinate for ", zone[0], " --> ", bc[0], bnd_cl.shape[0])
    pdm_point_merge.cloud_set(i_point_cloud, bnd_cl.shape[0], bnd_xyz, bnd_cl)

    # > Needed to hold memory
    PT.new_DataArray("bnd_xyz", value=bnd_xyz, parent=bc)
    PT.new_DataArray("bnd_cl" , value=bnd_cl , parent=bc)
    # print("bnd_xyz::", bnd_xyz)
    # print("bnd_cl ::", bnd_cl)

    bnd_to_join_path_list.append(bc[0])
    i_point_cloud = i_point_cloud + 1

  return i_point_cloud


def connect_match_from_family(part_tree, family_list, comm,
                              match_type = ['FaceCenter'],
                              rel_tol=1e-5):
  """
  Utily fonction to find in a configuration the List of Wall contains in Family or BC
  TODO : Structured / DG
  """
  zones = PT.get_all_Zone_t(part_tree)

  n_point_cloud = compute_n_point_cloud(zones, family_list)

  pdm_point_merge = PDM.PointsMerge(comm, n_point_cloud, rel_tol)

  i_point_cloud         = 0
  l_send_entity_data    = list()
  l_send_entity_stri    = list()
  l_send_zone_id_data   = list()
  zone_name_and_lid     = dict()
  bnd_to_join_path_list = [[]]*len(zones)
  for i_zone, zone in enumerate(zones):
    bnd_to_join_path_list_local = list()
    if PT.Zone.Type(zone) == 'Structured':
      i_point_cloud += prepare_pdm_point_merge_structured(pdm_point_merge, i_point_cloud, match_type,
                                                          i_zone, zone, family_list,
                                                          bnd_to_join_path_list_local,
                                                          l_send_entity_stri,
                                                          l_send_entity_data,
                                                          l_send_zone_id_data)
    else:
      i_point_cloud += prepare_pdm_point_merge_unstructured(pdm_point_merge, i_point_cloud, match_type,
                                                            i_zone, zone, family_list,
                                                            bnd_to_join_path_list_local,
                                                            l_send_entity_stri,
                                                            l_send_entity_data,
                                                            l_send_zone_id_data)
    bnd_to_join_path_list[i_zone] = bnd_to_join_path_list_local
    zone_name_and_lid[i_zone] = zone[0]

  pdm_point_merge.compute()

  l_neighbor_idx  = list()
  l_neighbor_desc = list()
  for i_cloud in range(n_point_cloud):
    res = pdm_point_merge.get_merge_candidates(i_cloud)

    l_neighbor_idx .append(res['candidates_idx' ])
    l_neighbor_desc.append(res['candidates_desc'])

    # print("res['candidates_idx ']::", res['candidates_idx' ])
    # print("res['candidates_desc']::", res['candidates_desc'])

  DNE = PDM.DistantNeighbor(comm,
                            n_point_cloud,
                            l_neighbor_idx,
                            l_neighbor_desc)

  l_recv_entity_data = list()
  l_recv_entity_stri = list()
  cst_stride         = 1
  DNE.DistantNeighbor_Exchange(l_send_entity_data,
                               l_recv_entity_data,
                               cst_stride,
                               l_send_entity_stri,
                               l_recv_entity_stri)

  # print("l_recv_entity_stri::", l_recv_entity_stri)
  # print("l_recv_entity_data::", l_recv_entity_data)

  l_recv_zone_id_data = list()
  l_recv_zone_id_stri = list()
  DNE.DistantNeighbor_Exchange(l_send_zone_id_data,
                               l_recv_zone_id_data,
                               cst_stride,
                               l_send_entity_stri,
                               l_recv_zone_id_stri)

  all_zone_name_and_lid = comm.gather(zone_name_and_lid   , root=0)
  all_zone_name_and_lid = comm.bcast(all_zone_name_and_lid, root=0)

  # if(comm.rank == 0):
  #   print("all_zone_name_and_lid::", all_zone_name_and_lid)

  # Setup at join
  i_point_cloud = 0
  for i_zone, zone in enumerate(zones):
    zgc_n = PT.update_child(zone, "ZoneGridConnectivity", "ZoneGridConnectivity_t")
    for bc in PT.Zone.getBCsFromFamily(zone, family_list):
      section_idx = adapt_match_information(l_neighbor_idx    [i_point_cloud],
                                            l_neighbor_desc   [i_point_cloud],
                                            l_recv_entity_stri[i_point_cloud],
                                            l_send_entity_data[i_point_cloud],
                                            l_recv_entity_data[i_point_cloud])

      for i in range(section_idx.shape[0]-1):
        n_entity_per_join = section_idx[i+1] - section_idx[i]
        # print(n_entity_per_join)
        pl     = NPY.empty((1, n_entity_per_join), order='F', dtype=NPY.int32)
        pl[0]  = NPY.copy(l_send_entity_data[i_point_cloud][section_idx[i]:section_idx[i+1]])

        pld    = NPY.empty((1, n_entity_per_join), order='F', dtype=NPY.int32)
        pld[0] = NPY.copy(l_recv_entity_data[i_point_cloud][section_idx[i]:section_idx[i+1]])

        connect_proc  = l_neighbor_desc   [i_point_cloud][0]
        connect_part  = l_neighbor_desc   [i_point_cloud][1]
        zone_opp_name = all_zone_name_and_lid[connect_proc][connect_part]

        jn_name = 'JNM.P{0}.N{1}.LT.P{2}.N{3}.{4}'.format(comm.Get_rank(), i_zone, connect_proc, connect_part, i_point_cloud)
        join_n = PT.new_GridConnectivity(name=jn_name, donor_name=zone_opp_name, type='Abutting1to1', loc='FaceCenter', parent=zgc_n)

        PT.new_PointList(name='PointList'     , value=pl , parent=join_n)
        PT.new_PointList(name='PointListDonor', value=pld, parent=join_n)

      i_point_cloud = i_point_cloud + 1

  # Remove all bcs
  for i_zone, zone in enumerate(zones):
    for zone_bc in PT.get_children_from_label(zone, 'ZoneBC_t'):
      for bc_name in bnd_to_join_path_list[i_zone]:
        PT.rm_children_from_name(zone_bc, bc_name)


