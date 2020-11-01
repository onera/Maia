import Converter.Internal as I
import numpy              as NPY
import Pypdm.Pypdm        as PDM

from .geometry import compute_face_center_and_characteristic_length

def bcs_if_in_family_list(zone, fams, family_list):
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      bctype = bc[1].tostring()
      if(bctype == b'FamilySpecified'):
        family_name_n = I.getNodeFromType1(bc, 'FamilyName_t')
        family_name   = family_name_n[1].tostring().decode()
        if(family_name in family_list):
          yield bc


def compute_n_point_cloud(zones, fams, family_list):
  """
  """
  n_point_cloud = 0
  for zone in zones:
    zone_type_n = I.getNodeFromType1(zone, 'ZoneType_t')
    zone_type   = zone_type_n[1].tostring()
    if(zone_type == b'Structured'):
      raise NotImplemented("connect_match_from_family for structured zone not allowed yet")
    for bc in bcs_if_in_family_list(zone, fams, family_list):
      n_point_cloud = n_point_cloud + 1

  return n_point_cloud

def prepare_pdm_point_merge_structured(pdm_point_merge, i_point_cloud, match_type,
                                       zone, fams,
                                       bnd_to_join_path_list,
                                       l_send_entity_stri,
                                       l_send_entity_data):
  """
  """
  raise NotImplemented("connect_match_from_family for structured zone not allowed yet")


def prepare_pdm_point_merge_unstructured(pdm_point_merge, i_point_cloud, match_type,
                                         zone, fams, family_list,
                                         bnd_to_join_path_list,
                                         l_send_entity_stri,
                                         l_send_entity_data):
  """
  match_type can be vertex or face
  """
  gridc_n = I.getNodeFromName1(zone   , 'GridCoordinates')
  cx      = I.getNodeFromName1(gridc_n, 'CoordinateX'    )[1]
  cy      = I.getNodeFromName1(gridc_n, 'CoordinateY'    )[1]
  cz      = I.getNodeFromName1(gridc_n, 'CoordinateZ'    )[1]

  for elmt in I.getNodesFromType1(zone, 'Elements_t'):
    if(elmt[1][0] == 22):
      found    = True
      face_vtx     = I.getNodeFromName1(elmt, 'ElementConnectivity')[1]
      face_vtx_idx = I.getNodeFromName1(elmt, 'ElementStartOffset' )[1]
      break
  if(not found):
    raise NotImplemented("Connect match need at least the NGonElements")

  for bc in bcs_if_in_family_list(zone, fams, family_list):
    print("Setup caracteristice lenght and coordinate for ", bc[0])

    pl = I.getNodeFromName1(bc, 'PointList')[1]

    l_send_entity_data.append(pl[0,:])
    l_send_entity_stri.append(NPY.ones(pl[0,:].shape, dtype='int32'))

    bnd_xyz, bnd_cl = compute_face_center_and_characteristic_length(pl, cx, cy, cz, face_vtx, face_vtx_idx)
    print(compute_face_center_and_characteristic_length.__doc__)
    print(bnd_xyz)
    print(bnd_cl)


    # pdm_point_merge.cloud_set(i_point_cloud, BCCharLen.shape[0], BCXYZ, BCCharLen)

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

  fams  = I.getNodesFromType2(part_tree, 'Family_t')
  zones = I.getNodesFromType2(part_tree, 'Zone_t')

  n_point_cloud = compute_n_point_cloud(zones, fams, family_list)

  pdm_point_merge = PDM.PointsMerge(comm, n_point_cloud, rel_tol)

  i_point_cloud         = 0
  l_send_entity_data    = list()
  l_send_entity_stri    = list()
  bnd_to_join_path_list = [[]]*len(zones)
  for i_zone, zone in enumerate(zones):
    zone_type_n = I.getNodeFromType1(zone, 'ZoneType_t')
    zone_type   = zone_type_n[1].tostring()
    if(zone_type == b'Structured'):
      i_point_cloud += prepare_pdm_point_merge_structured(pdm_point_merge, i_point_cloud, match_type,
                                                          zone, fams, family_list,
                                                          bnd_to_join_path_list[i_zone],
                                                          l_send_entity_stri,
                                                          l_send_entity_data)
    else:
      i_point_cloud += prepare_pdm_point_merge_unstructured(pdm_point_merge, i_point_cloud, match_type,
                                                            zone, fams, family_list,
                                                            bnd_to_join_path_list[i_zone],
                                                            l_send_entity_stri,
                                                            l_send_entity_data)

