from mpi4py import MPI

import maia.pytree      as PT
import maia.pytree.maia as MT
import Pypdm.Pypdm as PDM
import numpy as np
from maia.utils import np_utils, par_utils, as_pdm_gnum
from maia.factory.partitioning.split_U import cgns_to_pdm_dmesh
from maia.transfer import protocols as EP

def vtx_ids_to_face_ids(vtx_ids, ngon, comm):
  """
  From an array of vertex ids, search in the distributed NGon node
  the id of faces constituted by these vertices.
  Only Face having all their vertices in vtx_ids are returned
  """
  i_rank = comm.Get_rank()
  from maia.transfer import protocols as EP
  from maia.utils import as_pdm_gnum
  dface_vtx     = PT.get_child_from_name(ngon, 'ElementConnectivity')[1]
  dface_vtx_idx = PT.get_child_from_name(ngon, 'ElementStartOffset')[1]
  ngon_distri   = MT.getDistribution(ngon, 'Element')[1]

  # Prepare next BTP : we need a block view of the vtx ids (no duplicate, increasing order)
  # on a distribution involving all the vtx, so we include dface_vtx as a partition
  # We can exchange anything, since what is important is just the *number* of recv
  # data (== number of apparitions of this vtx in vtx_ids)
  PTB = EP.PartToBlock(None, [dface_vtx, vtx_ids], comm, keep_multiple=True)
  distri = PTB.getDistributionCopy()
  count,_  = PTB.exchange_field([np.empty(0, bool), np.ones(vtx_ids.size, bool)],                      #Data
                                [np.zeros(dface_vtx.size, np.int32), np.ones(vtx_ids.size, np.int32)]) #Stride

  # Fill tag array with True if vtx appears one or more in vtx_ids
  d_vtx_tag  = np.zeros(distri[i_rank+1] - distri[i_rank], bool)
  d_vtx_tag[count > 0] = True

  # Now send this array using face_vtx as lngn (so each face will receive the flag
  # of each one of its vertices)
  dface_vtx_tag = EP.block_to_part(d_vtx_tag, distri, [dface_vtx], comm)
  # Then reduce : select face if all its vertices have flag set to 1
  dface_vtx_idx_loc = dface_vtx_idx - dface_vtx_idx[0]
  dface_vtx_tag = np.logical_and.reduceat(dface_vtx_tag[0], dface_vtx_idx_loc[:-1])
  face_ids = np.where(dface_vtx_tag)[0] + ngon_distri[0] + 1

  return np_utils.safe_int_cast(face_ids, vtx_ids.dtype)


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


def _point_merge(clouds, comm, rel_tol=1e-3):
  """
  Wraps PDM.PointsMerge. A cloud is a tuple (coordinates, carac_lenght, parent_gnum)
  """
  pdm_point_merge = PDM.PointsMerge(comm, len(clouds), rel_tol)

  for icloud, cloud in enumerate(clouds):
    coords       = cloud['coords']
    carac_length = cloud['carac_length']
    dn_cloud_pts = carac_length.size
    pdm_point_merge.cloud_set(icloud, dn_cloud_pts, coords, carac_length)

  pdm_point_merge.compute()

  return pdm_point_merge.make_interface()

def _get_cloud(dmesh, gnum, comm):
    dmesh_extractor = PDM.DMeshExtract(2, comm)
    dmesh_extractor.register_dmesh(dmesh)
    _gnum = as_pdm_gnum(gnum)
    dmesh_extractor.set_gnum_to_extract(PDM._PDM_MESH_ENTITY_FACE, _gnum)

    dmesh_extractor.compute()

    dmesh_extracted = dmesh_extractor.get_dmesh()
    parent_vtx  = dmesh_extractor.get_extract_parent_gnum(PDM._PDM_MESH_ENTITY_VERTEX)
    parent_face = dmesh_extractor.get_extract_parent_gnum(PDM._PDM_MESH_ENTITY_FACE)

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

    return {'coords'       : coords,
            'carac_length' : carac_length,
            'face_vtx_idx' : face_vtx_idx,
            'face_vtx'     : face_vtx,
            'parent_vtx'   : parent_vtx,
            'parent_face'  : parent_face}
           

def _convert_match_result_to_faces(out_vtx, clouds, comm):

    zones_dn_vtx       = [cloud['parent_vtx'].size  for cloud in clouds]
    zones_dn_face      = [cloud['parent_face'].size for cloud in clouds]
    zones_face_vtx_idx = [cloud['face_vtx_idx']     for cloud in clouds]
    zones_face_vtx     = [cloud['face_vtx']         for cloud in clouds]

    n_interface = len(out_vtx['np_cloud_pair']) // 2
    interface_dn_vtx  = [cur.size for cur in out_vtx['lgnum_cur']]
    interface_ids_vtx = [np_utils.interweave_arrays([cur, opp]) for (cur, opp) in zip(out_vtx['lgnum_cur'], out_vtx['lgnum_opp'])]
    interface_dom_vtx = [out_vtx['np_cloud_pair'][i:i+2] for i in range(0, len(out_vtx['np_cloud_pair']), 2)]


    _out_face = PDM.interface_vertex_to_face(n_interface,
                                             len(clouds),
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
    is_empty_l = np.array([_out_face[j]['interface_dn_face'] == 0 for j in range(n_interface)], dtype=bool)
    is_empty = np.empty(n_interface, bool)
    comm.Allreduce(is_empty_l, is_empty, op=MPI.LAND)

    # Use same format for out_face
    out_face = {'np_cloud_pair' : out_vtx['np_cloud_pair'][~np.repeat(is_empty, 2)],
                'lgnum_cur'     : [np.absolute(_out_face[j]['np_interface_ids_face'][0::2]) for j in range(n_interface) if not is_empty[j]],
                'lgnum_opp'      :[np.absolute(_out_face[j]['np_interface_ids_face'][1::2]) for j in range(n_interface) if not is_empty[j]]}

    return out_face


def get_vtx_cloud_from_subset(dist_tree, subset_path, comm, dmesh_cache={}):
  zone_path = PT.path_head(subset_path, 2)
  try:
    dmesh = dmesh_cache[zone_path]
  except KeyError:
    zone = PT.get_node_from_path(dist_tree, zone_path)
    dmesh = cgns_to_pdm_dmesh.cgns_dist_zone_to_pdm_dmesh(zone, comm)
    dmesh_cache[zone_path] = dmesh

  node = PT.get_node_from_path(dist_tree, subset_path)
  assert PT.Subset.GridLocation(node) == 'FaceCenter', "Only face center nodes are managed"
  pl = PT.get_child_from_name(node, 'PointList')[1][0]
  cloud = _get_cloud(dmesh, pl, comm)
  return cloud

def apply_periodicity(cloud, periodic):
  coords = cloud['coords']
  cx = coords[0::3]
  cy = coords[1::3]
  cz = coords[2::3]
  cx_p, cy_p, cz_p = np_utils.transform_cart_vectors(cx,cy,cz, **periodic)
  coords_p = np_utils.interweave_arrays([cx_p, cy_p, cz_p])
  cloud['coords'] = coords_p


def recover_1to1_pairing(dist_tree, subset_paths, comm, periodic=None, **kwargs):

    # Steps are
    # 1.  Get input PL at faces (if they are vertex -> convert it)
    # 2.  Extract 2d face mesh. Apply periodicity if necessary
    # 3.  Get matching result at vertex
    # 4.  Convert matching result at faces
    # 5.  Go back to volumic numbering
    # 6.  Create output for matched faces
    # 7.  Check resulting faces vs input faces

    assert len(subset_paths) == 2

    clouds_path = subset_paths[0] + subset_paths[1]
    clouds = []

    cached_dmesh = {} #Use caching to avoid translate zone->dmesh 2 times
    for cloud_path in subset_paths[0]:
      cloud = get_vtx_cloud_from_subset(dist_tree, cloud_path, comm, cached_dmesh)
      if periodic is not None:
        apply_periodicity(cloud, periodic)
      clouds.append(cloud)
    for cloud_path in subset_paths[1]:
      cloud = get_vtx_cloud_from_subset(dist_tree, cloud_path, comm, cached_dmesh)
      clouds.append(cloud)

    PT.rm_nodes_from_name(dist_tree, ":CGNS#MultiPart") #Cleanup

    matching_vtx  = _point_merge(clouds, comm)
    matching_face = _convert_match_result_to_faces(matching_vtx, clouds, comm)

    # Conversion en numéro parent
    n_interface_vtx  = len(matching_vtx['np_cloud_pair']) // 2
    n_interface_face = len(matching_face['np_cloud_pair']) // 2
    for i_itrf in range(n_interface_vtx):
      for j, side in enumerate(['lgnum_cur', 'lgnum_opp']):
        i_cloud = matching_vtx['np_cloud_pair'][2*i_itrf+j]
        parent_vtx_num = clouds[i_cloud]['parent_vtx']
        parent_face_num = clouds[i_cloud]['parent_face']

        distri_vtx = par_utils.dn_to_distribution(parent_vtx_num.size, comm)
        distri_face = par_utils.dn_to_distribution(parent_face_num.size, comm)

        matching_vtx[side][i_itrf]  = EP.block_to_part(parent_vtx_num,  distri_vtx,  [matching_vtx[side][i_itrf]],  comm)[0]
        # We can have less face interface than vertex interface (we removed empty face interface)
        if i_itrf < n_interface_face:
          matching_face[side][i_itrf] = EP.block_to_part(parent_face_num, distri_face, [matching_face[side][i_itrf]], comm)[0]


    # Add created nodes in tree
    output_loc = kwargs.get("location", "FaceCenter")
    if output_loc == 'Vertex':
      cloud_pair = matching_vtx['np_cloud_pair']
      gnum_cur   = matching_vtx['lgnum_cur']
      gnum_opp   = matching_vtx['lgnum_opp']
    elif output_loc == 'FaceCenter':
      cloud_pair = matching_face['np_cloud_pair']
      gnum_cur   = matching_face['lgnum_cur']
      gnum_opp   = matching_face['lgnum_opp']

    if periodic is not None:
      perio_opp = {'translation'     : - periodic.get('translation', np.zeros(3, np.float32)),
                   'rotation_center' :   periodic.get('rotation_center', np.zeros(3, np.float32)),
                   'rotation_angle'  : - periodic.get('rotation_angle', np.zeros(3, np.float32))}
    else:
      perio_opp = None

    n_spawn = {path:0 for path in clouds_path}
    for i_interface in range(cloud_pair.size // 2):

      jn_distri = par_utils.dn_to_distribution(gnum_cur[i_interface].size, comm)
      for j in range(2):
        if j == 0:
          origin_path_cur = clouds_path[cloud_pair[2*i_interface]]
          origin_path_opp = clouds_path[cloud_pair[2*i_interface+1]]
          _gnum_cur = gnum_cur[i_interface]
          _gnum_opp = gnum_opp[i_interface]
          _periodic = periodic
        else:
          origin_path_cur = clouds_path[cloud_pair[2*i_interface+1]]
          origin_path_opp = clouds_path[cloud_pair[2*i_interface]]
          _gnum_cur = gnum_opp[i_interface]
          _gnum_opp = gnum_cur[i_interface]
          _periodic = perio_opp

        leaf_name_cur = PT.path_tail(origin_path_cur)
        leaf_name_opp = PT.path_tail(origin_path_opp)
        zone_cur_path = PT.path_head(origin_path_cur, 2)
        zone_opp_path = PT.path_head(origin_path_opp, 2)
        zone_cur  = PT.get_node_from_path(dist_tree, zone_cur_path)
        zgc = PT.update_child(zone_cur, 'ZoneGridConnectivity', 'ZoneGridConnectivity_t')

        jn_name_cur = f"{leaf_name_cur}_{n_spawn[origin_path_cur]}"
        jn_name_opp = f"{leaf_name_opp}_{n_spawn[origin_path_opp]}"

        jn = PT.new_GridConnectivity(jn_name_cur,
                                     zone_opp_path,
                                     'Abutting1to1',
                                     loc=output_loc,
                                     point_list = _gnum_cur.reshape((1,-1), order='F'),
                                     point_list_donor = _gnum_opp.reshape((1,-1), order='F'),
                                     parent=zgc)
        PT.new_child(jn, "GridConnectivityDonorName", "Descriptor_t", jn_name_opp)

        if periodic is not None:
          PT.new_GridConnectivityProperty(_periodic, jn)

        MT.newDistribution({"Index" : jn_distri.copy()}, jn)

        to_copy = lambda n: PT.get_label(n) in ['FamilyName_t', 'AdditionalFamilyName_t']
        origin_node = PT.get_node_from_path(dist_tree, origin_path_cur)
        for node in PT.get_children_from_predicate(origin_node, to_copy):
          PT.add_child(jn, node)

      n_spawn[origin_path_cur] += 1
      n_spawn[origin_path_opp] += 1


    # Cleanup : remove input node or transform it to keep only unmatched faces
    for i_cloud, cloud_path in enumerate(clouds_path):
        spawn = np.where(cloud_pair == i_cloud)[0]
        itrf_id, side = np.divmod(spawn, 2) #  Convert into num iterface + pos (O or 1)
        input_face = PT.get_node_from_path(dist_tree, f"{cloud_path}/PointList")[1][0]
        output_faces = []
        for j,s in zip(itrf_id, side):
            output_faces.append(matching_face[['lgnum_cur', 'lgnum_opp'][s]][j])
        # Search input_face that are not in output face
        unfound = dist_set_difference(input_face, output_faces, comm)
        if comm.allreduce(unfound.size, MPI.SUM) > 0:
          input_node = PT.get_node_from_path(dist_tree, cloud_path)
          PT.set_name(input_node, f"{PT.get_name(input_node)}_unmatched")
          PT.update_child(input_node, 'GridLocation', value='FaceCenter')
          PT.update_child(input_node, 'PointList', value=unfound.reshape((1,-1), order='F'))
          MT.newDistribution({'Index':  par_utils.dn_to_distribution(unfound.size, comm)}, input_node)
        else:
          PT.rm_node_from_path(dist_tree, cloud_path)


def recover_1to1_pairing_from_families(dist_tree, families, comm, periodic=None, **kwargs):
  """Find the matching faces between cgns nodes belonging to the two provided families.

  For each one of the two families, all the BC_t or GridConnectivity_t nodes related to the family
  through a FamilyName/AdditionalFamilyName node will be included in the pairing process.

  If the interface is periodic, the transformation from the first to the second family
  entities must be specified using the ``periodic`` argument; a dictionnary with keys
  ``'translation'``, ``'rotation_center'`` and/or ``'rotation_angle'`` is expected.
  Each key maps to a 3-sized numpy array, with missing keys defaulting zero vector.

  Input tree is modified inplace : relevant GridConnectivity_t with PointList and PointListDonor
  data are created.
  If all the original elements are successfully paired, the original nodes are removed. Otherwise,
  unmatched faces remains in their original node which is suffixed by '_unmatched'.

  This function allows the additional optional parameters:

  - ``location`` (default = 'FaceCenter') -- Controls the output GridLocation of
    the created interfaces. 'FaceCenter' or 'Vertex' are admitted.

  Args:
    dist_tree (CGNSTree): Input distributed tree. Only U-NGon connectivities are managed.
    families (tuple of str): Name of the two families to connect.
    comm       (MPIComm): MPI communicator
    periodic (dic, optional): Transformation from first to second family if the interface is periodic.
      None otherwise. Defaults to None.
    **kwargs: Additional options

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #recover1to1@start
        :end-before: #recover1to1@end
        :dedent: 2
  """

  is_subset_container = lambda n: PT.get_label(n) in ['ZoneBC_t', 'ZoneGridConnectivity_t']
  is_subset           = lambda n: PT.get_label(n) in ['BC_t', 'GridConnectivity_t', 'GridConnectivity1to1_t']

  assert isinstance(families, (list, tuple)) and len(families) == 2

  subset_path = (list(), list())
  for zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    zone = PT.get_node_from_path(dist_tree, zone_path)
    for container in PT.get_children_from_predicate(zone, is_subset_container):
      for subset in PT.get_children_from_predicate(container, is_subset):
        path = f'{zone_path}/{PT.get_name(container)}/{PT.get_name(subset)}'
        for i_fam, family in enumerate(families):
          if PT.predicate.belongs_to_family(subset, family, True):
            subset_path[i_fam].append(path)

  recover_1to1_pairing(dist_tree, subset_path, comm, periodic, **kwargs)

