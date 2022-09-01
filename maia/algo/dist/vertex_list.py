import mpi4py.MPI as MPI
import numpy as np
import itertools

import Pypdm.Pypdm        as PDM
import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia       import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils import py_utils, np_utils, par_utils

from maia.algo.dist             import matching_jns_tools as MJT
from maia.transfer.dist_to_part import data_exchange      as MBTP

def face_ids_to_vtx_ids(face_ids, ngon, comm):
  """
  From an array of face ids, search in the distributed NGon node
  the id of vertices belonging to the faces.

  Return a tuple (vtx_offset, vtx_list).
  The offset array indicates to which face the vertices belong.
  Note that vertex ids can appear twice (or more) in vtx_list if they are shared by multiple faces
  """

  distri_ngon  = I.getVal(MT.getDistribution(ngon, 'Element')).astype(pdm_dtype)

  pdm_distrib = par_utils.partial_to_full_distribution(distri_ngon, comm)
  dist_data = PT.get_child_from_name(ngon, 'ElementConnectivity')[1]
  b_stride = np.diff(PT.get_child_from_name(ngon, 'ElementStartOffset')[1]).astype(np.int32)

  # Get the vertex associated to the faces in FaceList
  part_data_pl = dict()
  BTP = PDM.BlockToPart(pdm_distrib, comm, [face_ids.astype(pdm_dtype)], 1)
  part_data_stride, part_data = BTP.exchange_field(dist_data, b_stride)

  face_offset_l = np_utils.sizes_to_indices(part_data_stride[0])

  return face_offset_l, part_data[0]

def filter_vtx_coordinates(grid_coords_node, distri_vtx, requested_vtx_ids, comm):
  """
  Get the coordinates of requested vertices ids (wraps BlockToPart) and
  return it as a numpy (n,3) array
  """
  dist_data = dict()
  for data in PT.iter_children_from_label(grid_coords_node, 'DataArray_t'):
    dist_data[I.getName(data)] = data[1]

  part_data = MBTP.dist_to_part(distri_vtx.astype(pdm_dtype), dist_data, [np.asarray(requested_vtx_ids, dtype=pdm_dtype)], comm)

  cx, cy, cz = part_data['CoordinateX'][0], part_data['CoordinateY'][0], part_data['CoordinateZ'][0]

  return np.array([cx,cy,cz], order='F').transpose()

def get_extended_pl(pl, pl_d, face_vtx_idx_pl, face_vtx_pl, comm, faces_to_skip=None):
  """
  Extend a local list of face ids (and its donor) by adding face ids of other procs
  that are connected to the faces by at least one vertex.

  faces_to_skip is either None, or a boolean array of size len(pl).
  If faces_to_skip[i]==True, it means the face is not considered
  """

  pl_vtx, pl_vtx_face_idx, pl_vtx_face   = np_utils.reverse_connectivity(pl  , face_vtx_idx_pl, face_vtx_pl)
  _     , _              , pl_vtx_face_d = np_utils.reverse_connectivity(pl_d, face_vtx_idx_pl, face_vtx_pl)
  if faces_to_skip is not None:
    idx_to_extract = np_utils.arange_with_jumps(face_vtx_idx_pl,faces_to_skip)
    restricted_pl_vtx = np.unique(face_vtx_pl[idx_to_extract]).astype(pdm_dtype)
  else:
    restricted_pl_vtx = pl_vtx.astype(pdm_dtype)

  # Exchange to locally have the list of *all* jn faces related to vertex
  p_stride = np.diff(pl_vtx_face_idx).astype(np.int32)

  part_data = {'vtx_to_face'   : [pl_vtx_face],
               'vtx_to_face_d' : [pl_vtx_face_d]}

  PTB = PDM.PartToBlock(comm, [pl_vtx.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=2)
  dist_data = dict()
  PTB.PartToBlock_Exchange(dist_data, part_data, [p_stride])


  #Recreate stride for all vertex
  first, count, total = PTB.getBeginNbEntryAndGlob()
  b_stride = np.zeros(count, np.int32)
  b_stride[PTB.getBlockGnumCopy() - first - 1] = dist_data['vtx_to_face#PDM_Stride']

  dist_data.pop('vtx_to_face#PDM_Stride')
  dist_data.pop('vtx_to_face_d#PDM_Stride')
  part_data = dict()

  BTP = PDM.BlockToPart(PTB.getDistributionCopy(), comm, [restricted_pl_vtx], 1)
  BTP.BlockToPart_Exchange2(dist_data, part_data, BlkStride=b_stride)

  extended_pl, unique_idx = np.unique(part_data["vtx_to_face"][0], return_index=True)
  extended_pl_d = part_data["vtx_to_face_d"][0][unique_idx]

  return extended_pl, extended_pl_d

def _search_by_intersection(pl_face_vtx_idx, pl_face_vtx, pld_face_vtx):
  """
  Take two face_vtx arrays of matching faces and try to construct the matching
  vertices list using face intersections (topologic).
  Return two array of vertices : first one is simply unique(pl_face_vtx), second
  one is same size and stores matching vertices *or* 0 if not found.
  Also return an bool array of size n_face indicating if face has been treated
  (ie all of its vertices have been determined) or not.

  Intersection method assumes that 2 matching faces have inverted
  face_vtx ordering, but does not  necessarily start with same vtx,
  ie A B C D -> C' B' A' D'. The idea of the method is to find some groups
  of faces and maching faces sharing a unique sequence of vertices,
  in order to identificate the common starting point.
  """

  n_face = len(pl_face_vtx_idx) - 1
  pl_vtx_local     = np.unique(pl_face_vtx)
  pl_vtx_local_opp = np.zeros_like(pl_vtx_local)
  face_is_treated  = np.zeros(n_face, dtype=bool)

  # Build connectivity vtx -> list of faces to which the vtx belongs
  r_pl, vtx_face_idx, vtx_face = np_utils.reverse_connectivity(np.arange(n_face), pl_face_vtx_idx, pl_face_vtx)

  #Invert dictionnary to have couple of faces -> list of shared vertices
  interfaces_to_nodes = dict()
  for ivtx, vtx in enumerate(r_pl):
    faces = vtx_face[vtx_face_idx[ivtx]: vtx_face_idx[ivtx+1]]
    for pair in itertools.combinations(sorted(faces), 2):
      try:
        interfaces_to_nodes[pair].append(vtx)
      except KeyError:
        interfaces_to_nodes[pair] = [vtx]

  vtx_g_to_l = {v:i for i,v in enumerate(pl_vtx_local)}

  # TODO re-write this algorithm
  # too long
  # for + face_is_treated.all() : it is at least quadratic => replace face_is_treated by stack
  # Non-numpy Python is slow
  for interface, vtx in interfaces_to_nodes.items():
    # For each couple of faces, we have a list of shared vertices : (fA,fB) -> [vtx0, .. vtxN]
    fA_idx = slice(pl_face_vtx_idx[interface[0]],pl_face_vtx_idx[interface[0]+1])
    fB_idx = slice(pl_face_vtx_idx[interface[1]],pl_face_vtx_idx[interface[1]+1])

    #Build the list of shared vertices for the two *opposite* faces
    opp_face_vtx_a = pld_face_vtx[fA_idx]
    opp_face_vtx_b = pld_face_vtx[fB_idx]
    opp_vtx = np.intersect1d(opp_face_vtx_a, opp_face_vtx_b)

    # If vertices are following, we can retrieve order. We may loop in normal
    # or reverse order depending on which vtx appears first
    subset_vtx = py_utils.get_ordered_subset(vtx, pl_face_vtx[fA_idx])
    if subset_vtx is not None:
      subset_vtx_opp = py_utils.get_ordered_subset(opp_vtx, opp_face_vtx_a)
    else:
      subset_vtx = py_utils.get_ordered_subset(vtx, pl_face_vtx[fB_idx])
      if subset_vtx is not None:
        subset_vtx_opp = py_utils.get_ordered_subset(opp_vtx, opp_face_vtx_b)

    # Skip non continous vertices and treat faces if possible
    if subset_vtx is not None:
      l_vertices = [vtx_g_to_l[v] for v in subset_vtx]
      assert subset_vtx_opp is not None
      assert len(opp_vtx) == len(l_vertices)
      pl_vtx_local_opp[l_vertices] = subset_vtx_opp[::-1]

      for face in interface:
        if not face_is_treated[face]:
          face_vtx     = pl_face_vtx[pl_face_vtx_idx[face]:pl_face_vtx_idx[face+1]]
          opp_face_vtx = pld_face_vtx[pl_face_vtx_idx[face]:pl_face_vtx_idx[face+1]]

          ordered_vtx     = np_utils.roll_from(face_vtx, start_value = subset_vtx[0])
          ordered_vtx_opp = np_utils.roll_from(opp_face_vtx, start_value = subset_vtx_opp[-1], reverse=True)

          pl_vtx_local_opp[[vtx_g_to_l[k] for k in ordered_vtx]] = ordered_vtx_opp

      face_is_treated[list(interface)] = True

  #We may do the where first to extract the untreated tests, and remove the treated faces
  # at each iteration. Stop when array is empty
  someone_changed = True
  while (not face_is_treated.all() and someone_changed):
    someone_changed = False
    for face in np.where(~face_is_treated)[0]:
      face_vtx   = pl_face_vtx [pl_face_vtx_idx[face]:pl_face_vtx_idx[face+1]]
      l_vertices = [vtx_g_to_l[v] for v in face_vtx]
      for i, vtx_opp in enumerate(pl_vtx_local_opp[l_vertices]):
        #Get any already deduced opposed vertex
        if vtx_opp != 0:
          opp_face_vtx = pld_face_vtx[pl_face_vtx_idx[face]:pl_face_vtx_idx[face+1]]
          ordered_vtx     = np_utils.roll_from(face_vtx, start_value = pl_vtx_local[l_vertices[i]])
          ordered_vtx_opp = np_utils.roll_from(opp_face_vtx, start_value = vtx_opp, reverse=True)

          pl_vtx_local_opp[[vtx_g_to_l[k] for k in ordered_vtx]] = ordered_vtx_opp
          face_is_treated[face] = True
          someone_changed = True
          break

  return pl_vtx_local, pl_vtx_local_opp, face_is_treated

def _search_with_geometry(zone, zone_d, gc_prop, pl_face_vtx_idx, pl_face_vtx, pld_face_vtx, comm):
  """
  Take two face_vtx arrays describing one or more matching faces (face link is given
  by pl_face_vtx_idx array), and try to construct the matching
  vertices list using face intersections (geometric).
  Return two array of vertices : first one is simply unique(pl_face_vtx), second
  one is same size and stores matching vertices.

  Intersection method assumes that 2 matching faces have inverted
  face_vtx ordering, but does not  necessarily start with same vtx,
  ie A B C D -> C' B' A' D'. The idea of the method is to identificate the
  starting vertex by sorting the coordinates with the same rule.
  Note that a result will be return even if the faces are not truly matching!
  """
  pl_vtx_local     = np.unique(pl_face_vtx)
  pl_vtx_local_opp = np.zeros_like(pl_vtx_local)

  vtx_g_to_l = {v:i for i,v in enumerate(pl_vtx_local)}

  assert len(pl_face_vtx) == len(pld_face_vtx) == pl_face_vtx_idx[-1]
  n_face = len(pl_face_vtx_idx) - 1
  n_face_vtx = len(pl_face_vtx)

  received_coords     = filter_vtx_coordinates(PT.get_child_from_label(zone, 'GridCoordinates_t'),
                                            I.getVal(MT.getDistribution(zone, 'Vertex')),
                                            pl_face_vtx, comm)
  opp_received_coords = filter_vtx_coordinates(PT.get_child_from_label(zone_d, 'GridCoordinates_t'),
                                            I.getVal(MT.getDistribution(zone_d, 'Vertex')),
                                            pld_face_vtx, comm)

  # TODO extract in apply_periodic_transformation
  #Apply transformation
  if gc_prop is not None:
    gc_periodic = PT.get_child_from_label(gc_prop, 'Periodic_t')
    translation     = PT.get_child_from_name(gc_periodic, 'Translation')[1]
    rotation_center = PT.get_child_from_name(gc_periodic, 'RotationCenter')[1]
    rotation_angle  = PT.get_child_from_name(gc_periodic, 'RotationAngle')[1]
    
    opp_received_coords = np_utils.transform_cart_matrix(opp_received_coords.T, translation, rotation_center, rotation_angle).T



  #Work locally on each original face to find the starting vtx
  for iface in range(n_face):
    vtx_idx     = slice(pl_face_vtx_idx[iface], pl_face_vtx_idx[iface+1])

    # Unique should sort array following the same key (axis=0 is important!)
    _, indices, counts = np.unique(received_coords[vtx_idx], axis = 0, return_index = True, return_counts = True)
    _, opp_indices = np.unique(opp_received_coords[vtx_idx], axis = 0, return_index = True)
    # Search first unique element (tie breaker) and use it as starting vtx
    idx = 0
    counts_it = iter(counts)
    while (next(counts_it) != 1):
      idx += 1
    first_vtx     = indices[idx]
    opp_first_vtx = opp_indices[idx]

    ordered_vtx     = np_utils.roll_from(pl_face_vtx[vtx_idx], start_idx = first_vtx)
    ordered_vtx_opp = np_utils.roll_from(pld_face_vtx[vtx_idx], start_idx = opp_first_vtx, reverse=True)

    pl_vtx_local_opp[[vtx_g_to_l[k] for k in ordered_vtx]] = ordered_vtx_opp

  return pl_vtx_local, pl_vtx_local_opp

def get_pl_isolated_faces(ngon_node, pl, vtx_distri, comm):
  """
  Take a list of face ids and search the faces who are isolated, ie who have no common
  edge or vertices with the other faces of the pointlist
  Return the array indices of theses faces
  """
  pl_face_vtx_idx, pl_face_vtx = face_ids_to_vtx_ids(pl, ngon_node, comm)
  pdm_vtx_distrib = par_utils.partial_to_full_distribution(vtx_distri, comm)
  PTB = PDM.PartToBlock(comm, [pl_face_vtx.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=2, userDistribution=pdm_vtx_distrib)
  block_gnum  = PTB.getBlockGnumCopy()
  vtx_n_occur = PTB.getBlockGnumCountCopy()

  vtx_n_occur_full = np.zeros(vtx_distri[1] - vtx_distri[0], np.int32)
  vtx_n_occur_full[block_gnum-vtx_distri[0]-1] = vtx_n_occur
  dist_data = {'n_occur' : vtx_n_occur_full}
  part_data = MBTP.dist_to_part(vtx_distri.astype(pdm_dtype), dist_data, [pl_face_vtx.astype(pdm_dtype)], comm)

  #This is the number of total occurence of all the vertices of each face. A face is isolated if each vertex appears
  # (globally) only once ie if this total equal the number of vertices of the face
  n_vtx_per_face = np.add.reduceat(part_data['n_occur'][0], indices=pl_face_vtx_idx[:-1])
  isolated_face     = np.where(n_vtx_per_face == np.diff(pl_face_vtx_idx))[0]

  return isolated_face

def generate_jn_vertex_list(dist_tree, jn_path, comm):
  """
  From a FaceCenter join (given by its path in the tree), create the distributed arrays VertexList
  and VertexListDonor such that vertices are matching 1 to 1.
  Return the two index arrays and the partial distribution array, which is
  identical for both of them
  """
  jn = I.getNodeFromPath(dist_tree, jn_path)
  assert PT.Subset.GridLocation(jn) == 'FaceCenter'

  base_name, zone_name = jn_path.split('/')[0:2]
  zone   = I.getNodeFromPath(dist_tree, base_name + '/' + zone_name)
  zone_d = I.getNodeFromPath(dist_tree, PT.getZoneDonorPath(base_name, jn))

  ngon_node   = PT.Zone.NGonNode(zone)
  vtx_distri  = I.getVal(MT.getDistribution(zone, 'Vertex')).astype(pdm_dtype)
  face_distri = I.getVal(MT.getDistribution(ngon_node, 'Element')).astype(pdm_dtype)

  ngon_node_d   = PT.Zone.NGonNode(zone_d)
  vtx_distri_d  = I.getVal(MT.getDistribution(zone_d, 'Vertex'))
  face_distri_d = I.getVal(MT.getDistribution(ngon_node_d, 'Element'))

  distri_jn = I.getVal(MT.getDistribution(jn, 'Index'))
  pl   = PT.get_child_from_name(jn, 'PointList'     )[1][0]
  pl_d = PT.get_child_from_name(jn, 'PointListDonor')[1][0]


  dn_vtx  = [vtx_distri[1] - vtx_distri[0],   vtx_distri_d[1] - vtx_distri_d[0]]
  dn_face = [face_distri[1] - face_distri[0], face_distri_d[1] - face_distri_d[0]]

  shifted_eso = lambda ng: (I.getNodeFromPath(ng, 'ElementStartOffset')[1] - I.getNodeFromPath(ng, 'ElementStartOffset')[1][0]).astype(np.int32)
  dface_vtx_idx = [shifted_eso(ng)  for ng in [ngon_node, ngon_node_d]]
  dface_vtx     = [I.getNodeFromPath(ng, 'ElementConnectivity')[1] for ng in [ngon_node, ngon_node_d]]

  isolated_face_loc = get_pl_isolated_faces(ngon_node, pl, vtx_distri, comm)
  not_isolated_face_loc = np.arange(pl.size)[np_utils.others_mask(pl, isolated_face_loc)]

  solo_face = comm.allreduce(isolated_face_loc.size > 0, MPI.LOR)
  conn_face = comm.allreduce(not_isolated_face_loc.size > 0, MPI.LOR)

  pl_vtx_l = []
  pld_vtx_l = []

  if solo_face:
    gc_prop = PT.get_child_from_label(jn, 'GridConnectivityProperty_t')
    _, pld_face_vtx = face_ids_to_vtx_ids(pl_d, ngon_node_d, comm)

    pl_face_vtx_idx, pl_face_vtx = face_ids_to_vtx_ids(pl, ngon_node, comm)
    pl_face_vtx_idx_e, pl_face_vtx_e  = np_utils.jagged_extract(pl_face_vtx_idx, pl_face_vtx,  isolated_face_loc)
    pl_face_vtx_idx_e, pld_face_vtx_e = np_utils.jagged_extract(pl_face_vtx_idx, pld_face_vtx, isolated_face_loc)
    pl_vtx_local, pl_vtx_local_opp = \
        _search_with_geometry(zone, zone_d, gc_prop, pl_face_vtx_idx_e, pl_face_vtx_e, pld_face_vtx_e, comm)
    pl_vtx_l.append(pl_vtx_local)
    pld_vtx_l.append(pl_vtx_local_opp)

  if conn_face:
    interface_dn_face  = not_isolated_face_loc.size
    interface_ids_face = np_utils.interweave_arrays([pl[not_isolated_face_loc],pl_d[not_isolated_face_loc]]).astype(pdm_dtype)

    vtx_interfaces = PDM.interface_face_to_vertex(1, #n_interface,
                                                  2,
                                                  False,
                                                  [interface_dn_face],
                                                  [interface_ids_face],
                                                  [(0,1)],
                                                  dn_vtx,
                                                  dn_face,
                                                  dface_vtx_idx,
                                                  dface_vtx,
                                                  comm)
    dn_vtx_jn = vtx_interfaces[0]['interface_dn_vtx']
    interface_ids_vtx = vtx_interfaces[0]['np_interface_ids_vtx'] # Can be void because of realloc

    if interface_ids_vtx is not None:
      pl_vtx_local  = np.copy(interface_ids_vtx[::2]) #Copy is needed to have aligned memory
      pld_vtx_local = np.copy(interface_ids_vtx[1::2])
      assert pl_vtx_local.size == dn_vtx_jn
      assert pld_vtx_local.size == dn_vtx_jn
    else:
      pl_vtx_local  = np.empty(0, dtype=pdm_dtype)
      pld_vtx_local = np.empty(0, dtype=pdm_dtype)

    pl_vtx_l.append(pl_vtx_local)
    pld_vtx_l.append(pld_vtx_local)

  #Final part_to_block will merge gnum from two method and reequilibrate
  PTB = PDM.PartToBlock(comm, pl_vtx_l, pWeight=[np.ones(pl.size, float) for pl in pl_vtx_l], partN=len(pl_vtx_l),
                          t_distrib=0, t_post=2)
  pl_vtx = PTB.getBlockGnumCopy()
  _, pld_vtx = PTB.exchange_field(pld_vtx_l, [np.ones(pl.size, np.int32) for pl in pl_vtx_l])
  assert pld_vtx.size == pl_vtx.size
  dn_vtx_jn = pld_vtx.size
  distri = par_utils.gather_and_shift(dn_vtx_jn, comm)
  distri_jn_vtx = distri[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

  return pl_vtx, pld_vtx, distri_jn_vtx

def _generate_jns_vertex_list(dist_tree, interface_pathes, comm):
  """
  Such as generate_jn_vertex_list, create the distributed arrays VertexList
  and VertexListDonor such that vertices are matching 1 to 1 from FaceCenter interfaces.
  This function manage several interface at the same time, but will no work if isolated faces
  (requiring geometric treatment) are present.
  """
  shifted_eso = lambda ng: (I.getNodeFromPath(ng, 'ElementStartOffset')[1] - I.getNodeFromPath(ng, 'ElementStartOffset')[1][0]).astype(np.int32)

  # Collect zones data
  zone_to_id = {}
  dn_vtx = []
  dn_face = []
  dface_vtx_idx = []
  dface_vtx = []
  for i, zone_path in enumerate(PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t')):
    zone_to_id[zone_path] = i

    zone = I.getNodeFromPath(dist_tree, zone_path)
    ngon = PT.Zone.NGonNode(zone)

    face_distri = MT.getDistribution(ngon, 'Element')[1].astype(pdm_dtype)
    vtx_distri  = MT.getDistribution(zone, 'Vertex')[1].astype(pdm_dtype)

    dn_vtx.append(vtx_distri[1] - vtx_distri[0])
    dn_face.append(face_distri[1] - face_distri[0])

    eso = PT.get_child_from_name(ngon, 'ElementStartOffset')[1]
    dface_vtx_idx.append(shifted_eso(ngon))
    dface_vtx.append(PT.get_child_from_name(ngon, 'ElementConnectivity')[1])

  # Collect interface data
  interface_dn_face = []
  interface_ids_face = []
  interface_dom_face = []
  for interface_path in interface_pathes:
    gc = I.getNodeFromPath(dist_tree, interface_path)
    pl  = PT.get_child_from_name(gc, 'PointList')[1][0]
    pld = PT.get_child_from_name(gc, 'PointListDonor')[1][0]

    interface_dn_face.append(pl.size)
    interface_ids_face.append(np_utils.interweave_arrays([pl,pld]))
    cur_zone_path = '/'.join(interface_path.split('/')[:2])
    opp_zone_path = PT.getZoneDonorPath(cur_zone_path.split('/')[0], gc)
    interface_dom_face.append((zone_to_id[cur_zone_path], zone_to_id[opp_zone_path]))

  #Call function
  vtx_interfaces = PDM.interface_face_to_vertex(len(interface_pathes),
                                                len(zone_to_id),
                                                False,
                                                interface_dn_face,
                                                interface_ids_face,
                                                interface_dom_face,
                                                dn_vtx,
                                                dn_face,
                                                dface_vtx_idx,
                                                dface_vtx,
                                                comm)
  
  # Unpack results
  all_pl_vtx = []
  all_pld_vtx = []
  all_distri_vtx = []
  for vtx_interface in vtx_interfaces:
    dn_vtx_jn = vtx_interface['interface_dn_vtx']
    distri = par_utils.gather_and_shift(dn_vtx_jn, comm)
    all_distri_vtx.append(distri[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]])

    interface_ids_vtx = vtx_interface['np_interface_ids_vtx'] # Can be void because of realloc
    if interface_ids_vtx is not None:
      pl_vtx  = np.copy(interface_ids_vtx[::2]) #Copy is needed to have aligned memory
      pld_vtx = np.copy(interface_ids_vtx[1::2])
      assert pl_vtx.size == pld_vtx.size == dn_vtx_jn
    else:
      pl_vtx  = np.empty(0, dtype=pdm_dtype)
      pld_vtx = np.empty(0, dtype=pdm_dtype)

    all_pl_vtx.append(pl_vtx)
    all_pld_vtx.append(pld_vtx)
    
  return all_pl_vtx, all_pld_vtx, all_distri_vtx
    

def generate_jns_vertex_list(dist_tree, comm, have_isolated_faces=False):
  """
  For each 1to1 FaceCenter matching join found in the distributed tree,
  create a corresponding 1to1 Vertex matching join.

  Input tree is modified inplace: Vertex ``GridConnectivity_t`` nodes
  are stored under distinct containers named from the original ones, suffixed
  with `#Vtx`. Similarly, vertex GC nodes uses the original name suffixed
  with `#Vtx`.

  Only unstructured-NGon based meshes are supported.

  Args:
    dist_tree  (CGNSTree): Distributed tree
    comm       (`MPIComm`) : MPI communicator
    have_isolated_faces (bool, optional) : Indicate if original joins includes
      faces who does not share any edge with other external (join) faces.
      If False, disable the special treatement needed by such faces (better performances,
      but will fail if isolated faces were actually present).
      Defaults to False.

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #generate_jns_vertex_list@start
        :end-before: #generate_jns_vertex_list@end
        :dedent: 2
  """
  #Build join ids to identify opposite joins
  MJT.add_joins_donor_name(dist_tree, comm)

  match_jns = MJT.get_matching_jns(dist_tree)
  interface_pathes_cur = [pair[0] for pair in match_jns]
  interface_pathes_opp = [pair[1] for pair in match_jns]
    
  if have_isolated_faces:
    #Filter interfaces having isolated faces; they will be treated one by one, while other will be grouped
    have_isolated = []
    for interface_path_cur in interface_pathes_cur:
      zone_path = '/'.join(interface_path_cur.split('/')[:2])
      zone_node = I.getNodeFromPath(dist_tree, zone_path)
      ngon_node = PT.Zone.NGonNode(zone_node)
      n_isolated = get_pl_isolated_faces(ngon_node, 
                                         I.getNodeFromPath(dist_tree, interface_path_cur + '/PointList')[1][0],
                                         MT.getDistribution(zone_node, 'Vertex')[1],
                                         comm).size
      have_isolated.append(bool(comm.allreduce(n_isolated, MPI.SUM) > 0))
    
    itrf_cur_with_iso    = [itrf for i,itrf in enumerate(interface_pathes_cur) if have_isolated[i]]
    itrf_cur_without_iso = [itrf for i,itrf in enumerate(interface_pathes_cur) if not have_isolated[i]]

    itrf_opp_with_iso    = [itrf for i,itrf in enumerate(interface_pathes_opp) if have_isolated[i]]
    itrf_opp_without_iso = [itrf for i,itrf in enumerate(interface_pathes_opp) if not have_isolated[i]]
  else:
    itrf_cur_with_iso    = []
    itrf_cur_without_iso = interface_pathes_cur
    itrf_opp_with_iso    = []
    itrf_opp_without_iso = interface_pathes_opp

  interface_pathes_cur = itrf_cur_without_iso + itrf_cur_with_iso
  interface_pathes_opp = itrf_opp_without_iso + itrf_opp_with_iso

  all_pl_vtx, all_pld_vtx, all_distri_vtx = _generate_jns_vertex_list(dist_tree, itrf_cur_without_iso, comm)
  for interface_path_cur in itrf_cur_with_iso:
    r = generate_jn_vertex_list(dist_tree, interface_path_cur, comm)
    for j, l in enumerate([all_pl_vtx, all_pld_vtx, all_distri_vtx]):
      l.append(r[j])


  # Create in tree
  for i, interface_path in enumerate(zip(interface_pathes_cur, interface_pathes_opp)):
    pl_vtx, pl_vtx_opp, distri_jn = all_pl_vtx[i], all_pld_vtx[i], all_distri_vtx[i] #Get results
    for j, gc_path in enumerate(interface_path):
      base_name, zone_name, zgc_name, gc_name = gc_path.split('/')
      zone = I.getNodeFromPath(dist_tree, base_name + '/' + zone_name)
      zgc  = PT.get_child_from_name(zone, zgc_name)
      gc = I.getNodeFromPath(dist_tree, gc_path)

      if j == 1: #Swap pl/pld for opposite jn
        pl_vtx, pl_vtx_opp = pl_vtx_opp, pl_vtx
      jn_vtx = I.newGridConnectivity(I.getName(gc)+'#Vtx', I.getValue(gc), ctype='Abutting1to1', parent=zgc)
      I.newGridLocation('Vertex', jn_vtx)
      I.newPointList('PointList',      pl_vtx.reshape(1,-1), parent=jn_vtx)
      I.newPointList('PointListDonor', pl_vtx_opp.reshape(1,-1), parent=jn_vtx)
      MT.newDistribution({'Index' : distri_jn}, jn_vtx)

      I._addChild(jn_vtx, PT.get_child_from_label(gc, 'GridConnectivityProperty_t'))
      I._addChild(jn_vtx, PT.get_child_from_name(gc, 'DistInterfaceId'))
      I._addChild(jn_vtx, PT.get_child_from_name(gc, 'DistInterfaceOrd'))
      donor_name_node = PT.get_child_from_name(gc, 'GridConnectivityDonorName')
      if donor_name_node is not None:
        I.newDescriptor('GridConnectivityDonorName', I.getValue(donor_name_node)+'#Vtx', parent=jn_vtx)


