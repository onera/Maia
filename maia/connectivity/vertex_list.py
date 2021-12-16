import mpi4py.MPI as MPI
import numpy as np
from math import cos, sin
import itertools

import Converter.Internal as I
import Pypdm.Pypdm        as PDM

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.sids import Internal_ext as IE
from maia.sids import sids
from maia.utils import py_utils
from maia.utils.parallel import utils as par_utils

from maia.transform.dist_tree import add_joins_ordinal as AJO

from maia.tree_exchange.dist_to_part import data_exchange as MBTP

def face_ids_to_vtx_ids(face_ids, ngon, comm):
  """
  From an array of face ids, search in the distributed NGon node
  the id of vertices belonging to the faces.

  Return a tuple (vtx_offset, vtx_list).
  The offset array indicates to which face the vertices belong.
  Note that vertex ids can appear twice (or more) in vtx_list if they are shared by multiple faces
  """

  distri_ngon  = I.getVal(IE.getDistribution(ngon, 'Element')).astype(pdm_dtype)

  pdm_distrib = par_utils.partial_to_full_distribution(distri_ngon, comm)
  dist_data = {'FaceVtx' : I.getNodeFromName1(ngon, 'ElementConnectivity')[1]}
  b_stride = np.diff(I.getNodeFromName1(ngon, 'ElementStartOffset')[1]).astype(np.int32)

  # Get the vertex associated to the faces in FaceList
  part_data_pl = dict()
  BTP = PDM.BlockToPart(pdm_distrib, comm, [face_ids.astype(pdm_dtype)], 1)
  BTP.BlockToPart_Exchange2(dist_data, part_data_pl, 1, b_stride)

  face_offset_l = py_utils.sizes_to_indices(part_data_pl['FaceVtx#PDM_Stride'][0])

  return face_offset_l, part_data_pl['FaceVtx'][0]

def filter_vtx_coordinates(grid_coords_node, distri_vtx, requested_vtx_ids, comm):
  """
  Get the coordinates of requested vertices ids (wraps BlockToPart) and
  return it as a numpy (n,3) array
  """
  dist_data = dict()
  for data in I.getNodesFromType1(grid_coords_node, 'DataArray_t'):
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

  pl_vtx, pl_vtx_face_idx, pl_vtx_face   = py_utils.reverse_connectivity(pl  , face_vtx_idx_pl, face_vtx_pl)
  _     , _              , pl_vtx_face_d = py_utils.reverse_connectivity(pl_d, face_vtx_idx_pl, face_vtx_pl)
  if faces_to_skip is not None:
    idx_to_extract = py_utils.arange_with_jumps(face_vtx_idx_pl,faces_to_skip)
    restricted_pl_vtx = np.unique(face_vtx_pl[idx_to_extract]).astype(pdm_dtype)
  else:
    restricted_pl_vtx = pl_vtx.astype(pdm_dtype)

  # Exchange to locally have the list of *all* jn faces related to vertex
  p_stride = np.diff(pl_vtx_face_idx).astype(np.int32)

  part_data = {'vtx_to_face'   : [pl_vtx_face],
               'vtx_to_face_d' : [pl_vtx_face_d]}

  PTB = PDM.PartToBlock(comm, [pl_vtx.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=2, t_stride=1)
  dist_data = dict()
  PTB.PartToBlock_Exchange(dist_data, part_data, [p_stride])


  #Recreate stride for all vertex
  first, count, total = PTB.getBeginNbEntryAndGlob()
  b_stride = np.zeros(count, np.int32)
  b_stride[PTB.getBlockGnumCopy() - first - 1] = dist_data['vtx_to_face#Stride']

  dist_data.pop('vtx_to_face#Stride')
  dist_data.pop('vtx_to_face_d#Stride')
  part_data = dict()

  BTP = PDM.BlockToPart(PTB.getDistributionCopy(), comm, [restricted_pl_vtx], 1)
  BTP.BlockToPart_Exchange2(dist_data, part_data, 1, b_stride)

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
  r_pl, vtx_face_idx, vtx_face = py_utils.reverse_connectivity(np.arange(n_face), pl_face_vtx_idx, pl_face_vtx)

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

          ordered_vtx     = py_utils.roll_from(face_vtx, start_value = subset_vtx[0])
          ordered_vtx_opp = py_utils.roll_from(opp_face_vtx, start_value = subset_vtx_opp[-1], reverse=True)

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
          ordered_vtx     = py_utils.roll_from(face_vtx, start_value = pl_vtx_local[l_vertices[i]])
          ordered_vtx_opp = py_utils.roll_from(opp_face_vtx, start_value = vtx_opp, reverse=True)

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
  ie A B C D -> D' B' A' C'. The idea of the method is to identificate the
  starting vertex by sorting the coordinates with the same rule.
  Note that a result will be return even if the faces are not truly matching!
  """
  pl_vtx_local     = np.unique(pl_face_vtx)
  pl_vtx_local_opp = np.zeros_like(pl_vtx_local)

  vtx_g_to_l = {v:i for i,v in enumerate(pl_vtx_local)}

  assert len(pl_face_vtx) == len(pld_face_vtx) == pl_face_vtx_idx[-1]
  n_face = len(pl_face_vtx_idx) - 1
  n_face_vtx = len(pl_face_vtx)

  received_coords     = filter_vtx_coordinates(I.getNodeFromType1(zone, 'GridCoordinates_t'),
                                            I.getVal(IE.getDistribution(zone, 'Vertex')),
                                            pl_face_vtx, comm)
  opp_received_coords = filter_vtx_coordinates(I.getNodeFromType1(zone_d, 'GridCoordinates_t'),
                                            I.getVal(IE.getDistribution(zone_d, 'Vertex')),
                                            pld_face_vtx, comm)

  # TODO extract in apply_periodic_transformation
  #Apply transformation
  if gc_prop is not None:
    gc_periodic = I.getNodeFromType1(gc_prop, 'Periodic_t')
    translation         = I.getNodeFromName1(gc_periodic, 'Translation')[1]
    rotation_center     = I.getNodeFromName1(gc_periodic, 'RotationCenter')[1]
    alpha, beta, gamma  = I.getNodeFromName1(gc_periodic, 'RotationAngle')[1]
    rotation_matx = np.array([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])
    rotation_maty = np.array([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
    rotation_matz = np.array([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])
    rotation_mat  = np.dot(rotation_matx, np.dot(rotation_maty, rotation_matz))

    # TODO either rotation_center == 0, or translate before rotation
    assert (rotation_center == np.zeros(3, dtype=np.float64)).all()
    opp_received_coords = np.dot(rotation_mat, opp_received_coords.T).T + translation


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

    ordered_vtx     = py_utils.roll_from(pl_face_vtx[vtx_idx], start_idx = first_vtx)
    ordered_vtx_opp = py_utils.roll_from(pld_face_vtx[vtx_idx], start_idx = opp_first_vtx, reverse=True)

    pl_vtx_local_opp[[vtx_g_to_l[k] for k in ordered_vtx]] = ordered_vtx_opp

  return pl_vtx_local, pl_vtx_local_opp

def generate_jn_vertex_list(dist_tree, jn_path, comm):
  """
  From a FaceCenter join (given by its path in the tree), create the distributed arrays VertexList
  and VertexListDonor such that vertices are matching 1 to 1.
  Return the two index arrays and the partial distribution array, which is
  identical for both of them
  """
  jn = I.getNodeFromPath(dist_tree, jn_path)
  assert sids.GridLocation(jn) == 'FaceCenter'

  base_name, zone_name = jn_path.split('/')[0:2]
  zone   = I.getNodeFromPath(dist_tree, base_name + '/' + zone_name)
  zone_d = I.getNodeFromPath(dist_tree, IE.getZoneDonorPath(base_name, jn))

  ngon_node   = sids.Zone.NGonNode(zone)
  vtx_distri  = I.getVal(IE.getDistribution(zone, 'Vertex'))
  face_distri = I.getVal(IE.getDistribution(ngon_node, 'Element'))

  ngon_node_d   = sids.Zone.NGonNode(zone_d)
  vtx_distri_d  = I.getVal(IE.getDistribution(zone_d, 'Vertex'))
  face_distri_d = I.getVal(IE.getDistribution(ngon_node_d, 'Element'))

  distri_jn = I.getVal(IE.getDistribution(jn, 'Index'))
  pl   = I.getNodeFromName1(jn, 'PointList'     )[1][0]
  pl_d = I.getNodeFromName1(jn, 'PointListDonor')[1][0]

  interface_dn_face  = distri_jn[1] - distri_jn[0]
  interface_ids_face = py_utils.interweave_arrays([pl,pl_d]).astype(pdm_dtype)

  dn_vtx  = [vtx_distri[1] - vtx_distri[0],   vtx_distri_d[1] - vtx_distri_d[0]]
  dn_face = [face_distri[1] - face_distri[0], face_distri_d[1] - face_distri_d[0]]

  shifted_eso = lambda ng: I.getNodeFromPath(ng, 'ElementStartOffset')[1] - I.getNodeFromPath(ng, 'ElementStartOffset')[1][0]
  dface_vtx_idx = [shifted_eso(ng)  for ng in [ngon_node, ngon_node_d]]
  dface_vtx     = [I.getNodeFromPath(ng, 'ElementConnectivity')[1] for ng in [ngon_node, ngon_node_d]]

  # We have to catch joins having isolated faces, because those one will need geometric treatment
  # Count the number of appareance of each vertex of the jn
  pl_face_vtx_idx, pl_face_vtx = face_ids_to_vtx_ids(pl, ngon_node, comm)
  pdm_vtx_distrib = par_utils.partial_to_full_distribution(vtx_distri, comm)
  PTB = PDM.PartToBlock(comm, [pl_face_vtx.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=2, t_stride=0, userDistribution=pdm_vtx_distrib)
  block_gnum  = PTB.getBlockGnumCopy()
  vtx_n_occur = PTB.getBlockGnumCountCopy()

  vtx_n_occur_full = np.zeros(vtx_distri[1] - vtx_distri[0], np.int32)
  vtx_n_occur_full[block_gnum-vtx_distri[0]-1] = vtx_n_occur
  dist_data = {'n_occur' : vtx_n_occur_full}
  part_data = MBTP.dist_to_part(vtx_distri.astype(pdm_dtype), dist_data, [pl_face_vtx.astype(pdm_dtype)], comm)

  n_vtx_per_face = np.add.reduceat(part_data['n_occur'][0], indices=pl_face_vtx_idx[:-1]) #Number of total occurence of all the vertices of each face
  solo_face_l = np.any(n_vtx_per_face == np.diff(pl_face_vtx_idx))
  solo_face = comm.allreduce(solo_face_l, MPI.LOR)
  if solo_face:
    gc_prop = I.getNodeFromType1(jn, 'GridConnectivityProperty_t')
    _, pld_face_vtx = face_ids_to_vtx_ids(pl_d, ngon_node_d, comm)
    pl_vtx_local, pl_vtx_local_opp = \
        _search_with_geometry(zone, zone_d, gc_prop, pl_face_vtx_idx, pl_face_vtx, pld_face_vtx, comm)

    # Not sure if we can have vertices defined twice... merge it to be sure (and reequilibrate)
    # Since we did not extract isolated faces, we have the whole join and thus we can have multiple vertices
    part_data = {'pl_vtx_opp' : [pl_vtx_local_opp]}
    PTB = PDM.PartToBlock(comm, [pl_vtx_local], pWeight=[np.ones(pl_vtx_local.size, float)], partN=1,
                          t_distrib=0, t_post=2, t_stride=1)
    dist_data = dict()
    PTB.PartToBlock_Exchange(dist_data, part_data, pStrid=[np.ones(pl_vtx_local.size, np.int32)])
    #Extract duplicated
    idx = np.cumsum(dist_data['pl_vtx_opp#Stride']) - 1

    pl_vtx = PTB.getBlockGnumCopy()
    pld_vtx = dist_data['pl_vtx_opp'][idx]
    dn_vtx_jn = pl_vtx.size

  else:
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
      pl_vtx  = interface_ids_vtx[::2]
      pld_vtx = interface_ids_vtx[1::2]
      assert pl_vtx.size == dn_vtx_jn
      assert pld_vtx.size == dn_vtx_jn
    else:
      pl_vtx  = np.empty(0, dtype=pdm_dtype)
      pld_vtx = np.empty(0, dtype=pdm_dtype)

  distri = par_utils.gather_and_shift(dn_vtx_jn, comm)
  distri_jn_vtx = distri[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

  return pl_vtx, pld_vtx, distri_jn_vtx

def generate_jn_vertex_listO(dist_tree, jn_path, comm):
  """
  From a FaceCenter join (given by its path in the tree), create the distributed arrays VertexList
  and VertexListDonor such that vertices are matching 1 to 1.
  Return the two index arrays and the partial distribution array, which is
  identical for both of them
  """

  jn = I.getNodeFromPath(dist_tree, jn_path)
  assert sids.GridLocation(jn) == 'FaceCenter'
  distri_jn = I.getVal(IE.getDistribution(jn, 'Index'))

  base_name, zone_name = jn_path.split('/')[0:2]
  zone = I.getNodeFromPath(dist_tree, base_name + '/' + zone_name)
  zone_d = I.getNodeFromPath(dist_tree, IE.getZoneDonorPath(base_name, jn))

  ngon   = [elem for elem in I.getNodesFromType1(zone,   'Elements_t') if elem[1][0] == 22][0]
  ngon_d = [elem for elem in I.getNodesFromType1(zone_d, 'Elements_t') if elem[1][0] == 22][0]

  pl   = I.getNodeFromName1(jn, 'PointList'     )[1][0]
  pl_d = I.getNodeFromName1(jn, 'PointListDonor')[1][0]

  #First pass with topologic treatment
  face_offset, pl_face_vtx  = face_ids_to_vtx_ids(pl  , ngon  , comm)
  face_offset, pld_face_vtx = face_ids_to_vtx_ids(pl_d, ngon_d, comm)

  pl_vtx_local, pl_vtx_local_opp, face_is_treated = _search_by_intersection(face_offset, pl_face_vtx, pld_face_vtx)

  undermined_vtx = (pl_vtx_local_opp == 0)
  pl_vtx_local_list     = [pl_vtx_local[~undermined_vtx].astype(pdm_dtype)]
  pl_vtx_local_opp_list = [pl_vtx_local_opp[~undermined_vtx].astype(pdm_dtype)]

  #Second pass with extended PL to try to catch locally isolated faces, having
  # neighbor faces in the same jn on other procs
  completed = comm.allreduce(not undermined_vtx.any(), op=MPI.LAND)
  if not completed and distri_jn[2] != 1:

    extended_pl, extended_pl_d = get_extended_pl(pl, pl_d, face_offset, pl_face_vtx, comm, face_is_treated)

    face_offset_e, pl_face_vtx_e  = face_ids_to_vtx_ids(extended_pl,   ngon, comm)
    face_offset_e, pld_face_vtx_e = face_ids_to_vtx_ids(extended_pl_d, ngon_d, comm)

    pl_vtx_local2, pl_vtx_local_opp2, face_is_treated2 = _search_by_intersection(face_offset_e, pl_face_vtx_e, pld_face_vtx_e)

    undermined_vtx = (pl_vtx_local_opp2 == 0)
    pl_vtx_local_list    .append(pl_vtx_local2[~undermined_vtx].astype(pdm_dtype))
    pl_vtx_local_opp_list.append(pl_vtx_local_opp2[~undermined_vtx].astype(pdm_dtype))

    completed = comm.allreduce(not undermined_vtx.any(), op=MPI.LAND)

    #Update face_is_treated array
    original_face_pos = {face:iface for iface,face in enumerate(extended_pl)} #Position of old face in extended array
    for iface,face in enumerate(pl):
      if not face_is_treated[iface] and face_is_treated2[original_face_pos[face]]:
        face_is_treated[iface] = True

  # Last pass : some faces remain untreated if they are totaly isolated in the original gc (in particular
  # if the original gc has only one face). For these we do a geometric treatment
  if not completed:
    #Extract faces already treated on topologic pass
    face_offset_e = py_utils.sizes_to_indices(np.diff(face_offset)[~face_is_treated])

    vtx_to_extract = py_utils.arange_with_jumps(face_offset,face_is_treated)
    pl_face_vtx_e  = pl_face_vtx[vtx_to_extract]
    pld_face_vtx_e = pld_face_vtx[vtx_to_extract]

    gc_prop = I.getNodeFromType1(jn, 'GridConnectivityProperty_t')
    pl_vtx_local3, pl_vtx_local_opp3 = \
        _search_with_geometry(zone, zone_d, gc_prop, face_offset_e, pl_face_vtx_e, pld_face_vtx_e, comm)
    assert np.all(pl_vtx_local_opp3 != 0)
    pl_vtx_local_list    .append(pl_vtx_local3.astype(pdm_dtype))
    pl_vtx_local_opp_list.append(pl_vtx_local_opp3.astype(pdm_dtype))


  #Now merge vertices appearing more than once
  #Careful, the distribution may be unequilibred since it is computed w.r.t vtx id
  #and not using number of vertices
  part_data = {'pl_vtx_opp' : pl_vtx_local_opp_list}
  PTB = PDM.PartToBlock(comm, pl_vtx_local_list, pWeight=None, partN=len(pl_vtx_local_list),
                        t_distrib=0, t_post=1, t_stride=0)
  pl_vtx = PTB.getBlockGnumCopy()
  dist_data = dict()
  PTB.PartToBlock_Exchange(dist_data, part_data)
  distri_jn_vtx_full =  par_utils.gather_and_shift(len(pl_vtx), comm, dtype=pdm_dtype)
  distri_jn_vtx =  distri_jn_vtx_full[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

  return pl_vtx, dist_data['pl_vtx_opp'], distri_jn_vtx

def generate_jns_vertex_list(dist_tree, comm):
  """
  For each 1to1 FaceCenter matching join found in the distributed tree,
  create a corresponding 1to1 Vertex matching join
  """
  #Build join ids to identify opposite joins
  if I.getNodeFromName(dist_tree, 'OrdinalOpp') is None:
    AJO.add_joins_ordinal(dist_tree, comm)

  ordinal_to_jn = {}

  query = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', \
      lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] and sids.GridLocation(n) == 'FaceCenter']

  for base, zone, zgc, gc in IE.getNodesWithParentsByMatching(dist_tree, query):
    jn_ordinal     = I.getNodeFromName1(gc, 'Ordinal')[1][0]
    jn_ordinal_opp = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]
    jn_key = min(jn_ordinal, jn_ordinal_opp)

    zgc_vtx = I.createUniqueChild(zone, I.getName(zgc)+'#Vtx', 'ZoneGridConnectivity_t')

    #If opposite join have already been treated, get it and switch pl-pld
    try:
      pl_vtx_opp, pl_vtx, distri_jn = ordinal_to_jn[jn_key]
    #Otherwise, treat join
    except KeyError:
      gc_path = '/'.join([I.getName(node) for node in [base, zone, zgc, gc]])
      pl_vtx, pl_vtx_opp, distri_jn = generate_jn_vertex_list(dist_tree, gc_path, comm)
      ordinal_to_jn[jn_key] = (pl_vtx, pl_vtx_opp, distri_jn)

    jn_vtx = I.newGridConnectivity(I.getName(gc)+'#Vtx', I.getValue(gc), ctype='Abutting1to1', parent=zgc_vtx)
    I.newGridLocation('Vertex', jn_vtx)
    I.newPointList('PointList',      pl_vtx.reshape(1,-1), parent=jn_vtx)
    I.newPointList('PointListDonor', pl_vtx_opp.reshape(1,-1), parent=jn_vtx)
    I.newIndexArray('PointList#Size', [1, distri_jn[2]], parent=jn_vtx)
    IE.newDistribution({'Index' : distri_jn}, jn_vtx)

    I._addChild(jn_vtx, I.getNodeFromType1(gc, 'GridConnectivityProperty_t'))
