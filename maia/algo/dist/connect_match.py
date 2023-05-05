from mpi4py import MPI

import maia.pytree      as PT
import maia.pytree.maia as MT
import Pypdm.Pypdm as PDM
import numpy as np
from maia.utils import np_utils, par_utils
from maia.factory.partitioning.split_U import cgns_to_pdm_dmesh


def dist_set_difference(ids, others, comm):
  """ Return the list of elements that belong only to ids and not to any other
  ids = numpy array
  others = list of numpy arrays
  """
  from maia.transfer import protocols as EP
  ln_to_gn = [ids] + others
  
  PTB = EP.PartToBlock(None, ln_to_gn, comm, keep_multiple=True)

  part_data   = [np.ones(ids.size, dtype=bool)] + [np.zeros(other.size, dtype=bool) for other in others]
  part_stride = [np.ones(pdata.size, dtype=np.int32) for pdata in part_data]

  dist_stride, dist_data = PTB.exchange_field(part_data, part_stride)
  dist_data = np.logical_and.reduceat(dist_data, np_utils.sizes_to_indices(dist_stride)[:-1])

  selected = PTB.getBlockGnumCopy()[dist_data]
  distri_in  = par_utils.gather_and_shift(selected.size, comm, dtype=np.int32)  
  distri_out = par_utils.uniform_distribution(distri_in[-1], comm)


  # Si on veut se caller sur les points d'entrée
  # tt = EP.block_to_part(dist_data, PTB.getDistributionCopy(), [ids], comm)
  # distri = PTB.getDistributionCopy()
  # BTP = EP.BlockToPart(distri, [ids], comm)
  # d_stride = np.zeros(distri[comm.Get_rank()+1] - distri[comm.Get_rank()], np.int32)
  # d_stride[PTB.getBlockGnumCopy() - distri[comm.Get_rank()] - 1] = 1

  # ts, tt = BTP.exchange_field(dist_data, d_stride)
  # print(tt)

  # dist_data = EP.part_to_block(part_data, None, ln_to_gn, comm, reduce_func=reduce_prod)

  # Sur chaque rank, on a une liste d'id (qui étaient sur ids ou pas) et un flag valant 1
  # si faut les garder
  # Il reste à supprimer et rééquilibrer
  selected = EP.block_to_block(selected, distri_in, distri_out, comm)
  return selected


def _point_merge(clouds, comm, rel_tol=1e-5):
  """
  Wraps PDM.PointsMerge. A cloud is a tuple (coordinates, carac_lenght, parent_gnum)
  """
  n_point_cloud = len(clouds)
  pdm_point_merge = PDM.PointsMerge(comm, len(clouds), rel_tol)

  for icloud, cloud in enumerate(clouds):
    coords, carac_length, _ = cloud
    pdm_point_merge.cloud_set(icloud, coords.size//3, coords, carac_length)

  pdm_point_merge.compute()

  vertex_parent_num = [cloud[2] for cloud in clouds]

  return pdm_point_merge.make_interface(vertex_parent_num)

def _dmesh_extract_2d(dmesh, gnum, loc, comm):
  """ Wraps DMeshExtract """
  dmesh_extractor = PDM.DMeshExtract(2, comm)
  dmesh_extractor.register_dmesh(dmesh)
  dmesh_extractor.set_gnum_to_extract(PDM._PDM_MESH_ENTITY_FACE, gnum)

  dmesh_extractor.compute()

  dmesh_extracted = dmesh_extractor.get_dmesh()

  parent = dmesh_extractor.get_extract_parent_gnum(PDM._PDM_MESH_ENTITY_VERTEX)
  return dmesh_extracted, parent

def _get_cloud(dmesh, gnum, comm):
    dmesh_extracted, parent = _dmesh_extract_2d(dmesh, gnum, 'Face', comm)

    coords  = dmesh_extracted.dmesh_vtx_coord_get()
    face_vtx_idx, face_vtx = dmesh_extracted.dmesh_connectivity_get(PDM._PDM_CONNECTIVITY_TYPE_FACE_VTX)
    carac_length = PDM.compute_vtx_characteristic_length(comm,
                                                         face_vtx_idx.size-1, #dn_face
                                                         0,                   #dn_edge
                                                         coords.size//3,      #dn_vtx
                                                         face_vtx_idx,
                                                         face_vtx,
                                                         None,                #edge_vtx
                                                         coords)
    return (coords, carac_length, parent)
           

def _convert_match_result_to_faces(dist_tree, clouds_path, out_vtx, comm):

    clouds_to_zone = []
    filtered_zones = []

    for cloud_path in clouds_path:
      zone_path = PT.path_head(cloud_path, 2)
      try:
        zone_index = filtered_zones.index(zone_path)
      except ValueError:
        zone_index = len(filtered_zones)
        filtered_zones.append(zone_path)
      clouds_to_zone.append(zone_index)

    # Get usefull data for zones referenced by some interface
    zones_dn_vtx  = []
    zones_dn_face = []
    zones_face_vtx_idx = []
    zones_face_vtx     = []
    for zone_path in filtered_zones:
      zone = PT.get_node_from_path(dist_tree, zone_path)
      ngon_node = PT.Zone.NGonNode(zone)
      vtx_distri = MT.getDistribution(zone, 'Vertex')[1]
      face_distri = MT.getDistribution(ngon_node, 'Element')[1]
      zones_dn_vtx.append(vtx_distri[1] - vtx_distri[0])
      zones_dn_face.append(face_distri[1] - face_distri[0])
      eso = PT.get_child_from_name(ngon_node, 'ElementStartOffset')[1]
      zones_face_vtx_idx.append(eso - eso[0])
      zones_face_vtx    .append(PT.get_child_from_name(ngon_node, 'ElementConnectivity')[1])


    n_interface = len(out_vtx['np_cloud_pair']) // 2
    dom_id =  [clouds_to_zone[c] for c in out_vtx['np_cloud_pair']]
    interface_dn_vtx  = [cur.size for cur in out_vtx['lgnum_cur']]
    interface_ids_vtx = [np_utils.interweave_arrays([cur, opp]) for (cur, opp) in zip(out_vtx['lgnum_cur'], out_vtx['lgnum_opp'])]
    interface_dom_vtx = [dom_id[i:i+2] for i in range(0, len(dom_id), 2)]


    _out_face = PDM.interface_vertex_to_face(n_interface,
                                             len(filtered_zones),
                                             False,
                                             interface_dn_vtx,
                                             interface_ids_vtx,
                                             interface_dom_vtx,
                                             zones_dn_vtx,
                                             zones_dn_face,
                                             zones_face_vtx_idx,
                                             zones_face_vtx,
                                             comm)


    # Filter empty interfaces
    is_empty_l = np.array([_out_face[j]['interface_dn_face'] == 0 for j in range(n_interface)])
    is_empty = np.empty(n_interface, bool)
    comm.Allreduce(is_empty_l, is_empty, op=MPI.LAND)

    # Use same format for out_face
    out_face = {'np_cloud_pair' : out_vtx['np_cloud_pair'][~np.repeat(is_empty, 2)],
                'lgnum_cur'     : [np.absolute(_out_face[j]['np_interface_ids_face'][0::2]) for j in range(n_interface) if not is_empty[j]],
                'lgnum_opp'      :[np.absolute(_out_face[j]['np_interface_ids_face'][1::2]) for j in range(n_interface) if not is_empty[j]]}

    return out_face


def connect_match_from_family(dist_tree, families, comm, periodic=None, **kwargs):

    assert len(families) == 2

    # Steps are
    # 1.  Get input PL at faces (if they are vertex -> TODO
    # 2.  Extract 2d face mesh
    # (Apply periodicity)
    # 3.  Get matching result at vertex
    # 4.  Convert matching result at faces
    # 5.  Check resulting faces vs input faces
    # 6.  Create output : non localized + match


    clouds = []
    clouds_path = []

    for i_zone, zone_path in enumerate(PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t')):
      zone  = PT.get_node_from_path(dist_tree, zone_path)
      dmesh = cgns_to_pdm_dmesh.cgns_dist_zone_to_pdm_dmesh(zone, comm)
      for container in PT.get_children_from_predicate(zone, lambda n: PT.get_label(n) in ['ZoneBC_t', 'ZoneGridConnectivity_t']):
          for family in families:
              predicate = lambda n : PT.get_label(n) in ['BC_t', 'GridConnectivity_t', 'GridConnectivity1to1_t'] \
                                     and PT.predicate.belongs_to_family(n, family, True)

              for node in PT.get_children_from_predicate(container, predicate):
                                                             
                assert PT.Subset.GridLocation(node) == 'FaceCenter', "Only face center nodes are managed"
                pl = PT.get_child_from_name(node, 'PointList')[1][0]

                clouds.append(_get_cloud(dmesh, pl, comm))
                clouds_path.append(f"{zone_path}/{PT.get_name(container)}/{PT.get_name(node)}") 

    out_vtx = _point_merge(clouds, comm)
    out_face = _convert_match_result_to_faces(dist_tree, clouds_path, out_vtx, comm)

    PT.rm_nodes_from_name(dist_tree, ":CGNS#MultiPart")

    # Add created nodes in tree
    output_loc = kwargs.get("location", "FaceCenter")
    if output_loc == 'Vertex':
      cloud_pair = out_vtx['np_cloud_pair']
      gnum_cur   = out_vtx['lgnum_cur']
      gnum_opp   = out_vtx['lgnum_opp']
    elif output_loc == 'FaceCenter':
      cloud_pair = out_face['np_cloud_pair']
      gnum_cur   = out_face['lgnum_cur']
      gnum_opp   = out_face['lgnum_opp']

    n_spawn = {path:0 for path in clouds_path}
    for i_interface in range(cloud_pair.size // 2):

      first_node_path  = clouds_path[cloud_pair[2*i_interface]]
      second_node_path = clouds_path[cloud_pair[2*i_interface+1]]

      first_leaf_name = PT.path_tail(first_node_path)
      second_leaf_name = PT.path_tail(second_node_path)
      first_zone_path = PT.path_head(first_node_path, 2)
      second_zone_path = PT.path_head(second_node_path, 2)
      first_zone  = PT.get_node_from_path(dist_tree, first_zone_path)
      second_zone = PT.get_node_from_path(dist_tree, second_zone_path)

      first_node = PT.get_node_from_path(dist_tree, first_node_path)
      second_node = PT.get_node_from_path(dist_tree, second_node_path)

      first_jn_name = f"{first_leaf_name}_{n_spawn[first_node_path]}"
      second_jn_name = f"{second_leaf_name}_{n_spawn[second_node_path]}"

      zgc = PT.update_child(first_zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')

      first_jn = PT.new_GridConnectivity(first_jn_name,
                                         second_zone_path,
                                         'Abutting1to1',
                                         loc=output_loc,
                                         point_list = gnum_cur[i_interface].reshape((1,-1), order='F'),
                                         point_list_donor = gnum_opp[i_interface].reshape((1,-1), order='F'),
                                         parent=zgc)
      PT.new_child(first_jn, "GridConnectivityDonorName", "Descriptor_t", second_jn_name)

      zgc = PT.update_child(second_zone, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')
      secondjn = PT.new_GridConnectivity(second_jn_name,
                                         first_zone_path,
                                         'Abutting1to1',
                                         loc=output_loc,
                                         point_list = gnum_opp[i_interface].reshape((1,-1), order='F'),
                                         point_list_donor = gnum_cur[i_interface].reshape((1,-1), order='F'),
                                         parent=zgc)
      PT.new_child(secondjn, "GridConnectivityDonorName", "Descriptor_t", first_jn_name)

      MT.newDistribution({"Index" : par_utils.dn_to_distribution(gnum_cur[i_interface].size, comm)}, first_jn)
      MT.newDistribution({"Index" : par_utils.dn_to_distribution(gnum_cur[i_interface].size, comm)}, secondjn)

      to_copy = lambda n: PT.get_label(n) in ['FamilyName_t', 'AdditionalFamilyName_t']
      for node in PT.get_children_from_predicate(first_node, to_copy):
        PT.add_child(first_jn, node)
      for node in PT.get_children_from_predicate(second_node, to_copy):
        PT.add_child(secondjn, node)

      n_spawn[first_node_path] += 1
      n_spawn[second_node_path] += 1


    # Cleanup : remove input node or transform it to keep only unmatched faces
    for i_cloud, cloud_path in enumerate(clouds_path):
        spawn = np.where(cloud_pair == i_cloud)[0]
        itrf_id, side = np.divmod(spawn, 2) #  Convert into num iterface + pos (O or 1)
        input_face = PT.get_node_from_path(dist_tree, f"{cloud_path}/PointList")[1][0]
        output_faces = []
        for j,s in zip(itrf_id, side):
            output_faces.append(out_face[['lgnum_cur', 'lgnum_opp'][s]][j])
        # Search input_face that are not in output face
        unfound = dist_set_difference(input_face, output_faces, comm)
        if comm.allreduce(unfound.size, MPI.SUM) > 0:
          input_node = PT.get_node_from_path(dist_tree, cloud_path)
          PT.set_name(input_node, f"{PT.get_name(input_node)}_X")
          PT.update_child(input_node, 'GridLocation', value='FaceCenter')
          PT.update_child(input_node, 'PointList', value=unfound.reshape((-1,1), order='F'))
          MT.newDistribution({'Index':  par_utils.dn_to_distribution(unfound.size, comm)}, input_node)
        else:
          PT.rm_node_from_path(dist_tree, cloud_path)

