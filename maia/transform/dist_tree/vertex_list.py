import mpi4py.MPI as MPI
import numpy as np
import itertools

import Converter.Internal as I
import Pypdm.Pypdm        as PDM

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.sids import Internal_ext as IE
from maia.sids import sids
from maia.utils import py_utils
from maia.utils.parallel import utils as par_utils

from maia.tree_exchange.dist_to_part import data_exchange as MBTP

def _is_subset_l(subset, L):
  """Return True is subset list is included in L, allowing looping"""
  extended_l = list(L) + list(L)[:len(subset)-1]
  return max([subset == extended_l[i:i+len(subset)] for i in range(len(L))])

def _is_before(l, a, b):
  """Return True is element a is present in list l before element b"""
  for e in l:
    if e==a:
      return True
    if e==b:
      return False
  return False

def _roll_from(array, start_idx = None, start_value = None, reverse = False):
  """
  Return a new array starting from given index (or value), in normal or reversed order
  """
  assert (start_idx is None) != (start_value is None)
  if start_idx is None:
    start_idx = np.where(array == start_value)[0][0]

  return np.roll(array, -start_idx) if not reverse else np.roll(array[::-1], start_idx + 1)

def facelist_to_vtxlist_local(pls, ngon, comm):
  """
  From a list of FaceCenter PointList, search in the distributed NGon node
  the id of vertices belonging to the faces.

  Return a list (size len(pls)) of tuples (vtx_offset, vtx_list). Offset array
  indicates to which face the vertices belongs.
  Note that vertices ids can appears twice (or more) in vtx_list if they are
  shared by some faces
  """

  distri_ngon  = IE.getDistribution(ngon, 'Element').astype(pdm_dtype)

  pdm_distrib = par_utils.partial_to_full_distribution(distri_ngon, comm)
  dist_data = {'FaceVtx' : I.getNodeFromName1(ngon, 'ElementConnectivity')[1]}
  b_stride = np.diff(I.getNodeFromName1(ngon, 'ElementStartOffset')[1]).astype(np.int32)

  # Get the vertex associated to the faces in FaceList
  part_data_pl = dict()
  BTP = PDM.BlockToPart(pdm_distrib, comm, [pl.astype(pdm_dtype) for pl in pls], len(pls))
  BTP.BlockToPart_Exchange2(dist_data, part_data_pl, 1, b_stride)

  face_offset_l = [py_utils.sizes_to_indices(p_sizes) for p_sizes in part_data_pl['FaceVtx#PDM_Stride']]

  return list(zip(face_offset_l, part_data_pl['FaceVtx']))

def get_vtx_coordinates(grid_coords_n, distri_vtx, requested_vtx, comm):
  """
  Get the coordinates of requested vertices ids (wraps BlockToPart) and
  return it as a numpy (n,3) array
  """
  dist_data = dict()
  for data in I.getNodesFromType1(grid_coords_n, 'DataArray_t'):
    dist_data[I.getName(data)] = data[1]

  part_data = MBTP.dist_to_part(distri_vtx, dist_data, [np.asarray(requested_vtx, dtype=pdm_dtype)], comm)

  cx, cy, cz = part_data['CoordinateX'][0], part_data['CoordinateY'][0], part_data['CoordinateZ'][0]

  return np.array([cx,cy,cz], order='F').transpose()

def get_extended_pl(pl, pl_d, face_vtx_idx_pl, face_vtx_pl, comm, faces_to_skip=None):
  """
  Extend a distributed face PointList (and its donor) by adding the faces present elsewhere
  in the PL and connected to the pl faces by one (or more) vertices. 
  If faces_to_skip is not None, ignore the faces for which the value is True when searching
  neighbors.
  """

  pl_vtx = np.unique(face_vtx_pl).astype(pdm_dtype)
  if faces_to_skip is not None:
    idx_to_extract = py_utils.multi_arange(face_vtx_idx_pl[:-1][~faces_to_skip], \
                                           face_vtx_idx_pl[1:][~faces_to_skip])
    restricted_pl_vtx = np.unique(face_vtx_pl[idx_to_extract]).astype(pdm_dtype)
  else:
    restricted_pl_vtx = pl_vtx

  #Map each vertex of pl_vtx to the couple (face, face_opp) if its belong to face
  pl_vtx_local_dict = {key: [] for key in pl_vtx}
  for iface, face_pair in enumerate(zip(pl, pl_d)):
    for vtx in face_vtx_pl[face_vtx_idx_pl[iface]:face_vtx_idx_pl[iface+1]]:
      pl_vtx_local_dict[vtx].append(face_pair)

  # Exchange to locally have the list of *all* jn faces related to vertex
  p_stride = np.array([len(pl_vtx_local_dict[vtx]) for vtx in pl_vtx], dtype=np.int32)

  part_data = dict()
  part_data["vtx_to_face"]   = [np.empty(np.sum(p_stride), dtype=np.int)]
  part_data["vtx_to_face_d"] = [np.empty(np.sum(p_stride), dtype=np.int)]
  offset = 0
  for vtx in pl_vtx:
    n = len(pl_vtx_local_dict[vtx])
    part_data["vtx_to_face"][0][offset:offset+n] = [t[0] for t in pl_vtx_local_dict[vtx]]
    part_data["vtx_to_face_d"][0][offset:offset+n] = [t[1] for t in pl_vtx_local_dict[vtx]]
    offset += n

  PTB = PDM.PartToBlock(comm, [pl_vtx], pWeight=None, partN=1,
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
  face_is_treated  = np.zeros(n_face, dtype=np.bool)

  # Build dict vtx -> list of faces to which its belong
  pl_vtx_local_dict = {key: [] for key in pl_vtx_local}
  for iface in range(n_face):
    for vtx in pl_face_vtx[pl_face_vtx_idx[iface]:pl_face_vtx_idx[iface+1]]:
      pl_vtx_local_dict[vtx].append(iface)

  #Invert dictionnary to have couple of faces -> list of shared vertices
  interfaces_to_nodes = dict()
  for key, val in pl_vtx_local_dict.items():
    for pair in itertools.combinations(sorted(val), 2):
      try:
        interfaces_to_nodes[pair].append(key)
      except KeyError:
        interfaces_to_nodes[pair] = [key]

  vtx_g_to_l = {v:i for i,v in enumerate(pl_vtx_local)}

  for interface, vtx in interfaces_to_nodes.items():
    # For each couple of faces, we have a list of shared vertices : (fA,fB) -> [vtx0, .. vtxN]
    fA_idx = slice(pl_face_vtx_idx[interface[0]],pl_face_vtx_idx[interface[0]+1])
    fB_idx = slice(pl_face_vtx_idx[interface[1]],pl_face_vtx_idx[interface[1]+1])
    step = 0
    #Build the list of shared vertices for the two *opposite* faces
    opp_face_vtx_a = pld_face_vtx[fA_idx]
    opp_face_vtx_b = pld_face_vtx[fB_idx]
    opp_vtx = np.intersect1d(opp_face_vtx_a, opp_face_vtx_b)

    # If vertices are following, we can retrieve order. We may loop in normal
    # or reverse order depending on which vtx appears first
    if _is_subset_l(vtx, pl_face_vtx[fA_idx]):
      step = -1 if _is_before(opp_face_vtx_a, opp_vtx[0], opp_vtx[-1]) else 1
    elif _is_subset_l(vtx[::-1], pl_face_vtx[fA_idx]):
      step = -1 if not _is_before(opp_face_vtx_a, opp_vtx[0], opp_vtx[-1]) else 1
    elif _is_subset_l(vtx, pl_face_vtx[fB_idx]):
      step = -1 if not _is_before(opp_face_vtx_b, opp_vtx[0], opp_vtx[-1]) else 1
    elif _is_subset_l(vtx[::-1], pl_face_vtx[fB_idx]):
      step = -1 if _is_before(opp_face_vtx_b, opp_vtx[0], opp_vtx[-1]) else 1

    # Skip non continous vertices and treat faces if possible
    if step != 0:
      l_vertices = [vtx_g_to_l[v] for v in vtx]
      assert len(opp_vtx) == len(l_vertices)
      pl_vtx_local_opp[l_vertices] = opp_vtx[::step]

      for face in interface:
        if not face_is_treated[face]:
          face_vtx     = pl_face_vtx[pl_face_vtx_idx[face]:pl_face_vtx_idx[face+1]]
          opp_face_vtx = pld_face_vtx[pl_face_vtx_idx[face]:pl_face_vtx_idx[face+1]]

          ordered_vtx     = _roll_from(face_vtx, start_value = vtx[0])
          ordered_vtx_opp = _roll_from(opp_face_vtx, start_value = pl_vtx_local_opp[l_vertices[0]], reverse=True)

          pl_vtx_local_opp[[vtx_g_to_l[k] for k in ordered_vtx]] = ordered_vtx_opp

      face_is_treated[list(interface)] = True

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
          ordered_vtx     = _roll_from(face_vtx, start_value = pl_vtx_local[l_vertices[i]])
          ordered_vtx_opp = _roll_from(opp_face_vtx, start_value = vtx_opp, reverse=True)

          pl_vtx_local_opp[[vtx_g_to_l[k] for k in ordered_vtx]] = ordered_vtx_opp
          face_is_treated[face] = True
          someone_changed = True
          break

  return pl_vtx_local, pl_vtx_local_opp, face_is_treated

def _search_with_geometry(zone, pl_face_vtx_idx, pl_face_vtx, pld_face_vtx, comm):
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

  Todo : manage transformation;
         manage multizone;
  """
  pl_vtx_local     = pl_face_vtx
  pl_vtx_local_opp = np.zeros_like(pl_vtx_local)

  vtx_g_to_l = {v:i for i,v in enumerate(pl_vtx_local)}
  grid_coords_n = I.getNodeFromType1(zone, 'GridCoordinates_t')
  distri_vtx = IE.getDistribution(zone, 'Vertex')

  assert len(pl_face_vtx) == len(pld_face_vtx) == pl_face_vtx_idx[-1]
  n_face = len(pl_face_vtx_idx) - 1
  n_face_vtx = len(pl_face_vtx)
  requested_vtx = np.concatenate([pl_face_vtx, pld_face_vtx])

  received_coords = get_vtx_coordinates(grid_coords_n, distri_vtx, requested_vtx, comm)

  for iface in range(n_face):

    vtx_idx     = slice(pl_face_vtx_idx[iface], pl_face_vtx_idx[iface+1])
    opp_vtx_idx = slice(pl_face_vtx_idx[iface] + n_face_vtx, pl_face_vtx_idx[iface+1] + n_face_vtx)
    face_vtx_coords     = received_coords[vtx_idx]
    opp_face_vtx_coords = received_coords[opp_vtx_idx]

    # Unique should sort array following the same key (axis=0 is important!)
    _, indices, counts = np.unique(face_vtx_coords, axis = 0, return_index = True, return_counts = True)
    _, opp_indices = np.unique(opp_face_vtx_coords, axis = 0, return_index = True)
    # Search first unique element (tie breaker) and use it as starting vtx
    idx = 0
    counts_it = iter(counts)
    while (next(counts_it) != 1):
      idx += 1
    first_vtx     = indices[idx]
    opp_first_vtx = opp_indices[idx]

    ordered_vtx     = _roll_from(pl_face_vtx[vtx_idx], start_idx = first_vtx)
    ordered_vtx_opp = _roll_from(pld_face_vtx[vtx_idx], start_idx = opp_first_vtx, reverse=True)

    pl_vtx_local_opp[[vtx_g_to_l[k] for k in ordered_vtx]] = ordered_vtx_opp

  pl_vtx_local, order = np.unique(pl_vtx_local, return_index = True)
  pl_vtx_local_opp = pl_vtx_local_opp[order]
  return pl_vtx_local, pl_vtx_local_opp


def generate_jn_vertex_list(zone, ngon, jn, comm):
  """
  From a FaceCenter join, create the distributed arrays VertexList
  and VertexListDonor such that vertices are matching 1 to 1.
  Return the two index arrays and the partial distribution array, which is
  identical for both of them

  For now only one zone is supported"""

  assert sids.GridLocation(jn) == 'FaceCenter'
  distri_jn = IE.getDistribution(jn, 'Index')

  pl   = I.getNodeFromName1(jn, 'PointList')[1][0]
  pl_d = I.getNodeFromName1(jn, 'PointListDonor')[1][0]

  face_offset, pl_face_vtx  = facelist_to_vtxlist_local([pl], ngon, comm)[0]
  face_offset, pld_face_vtx = facelist_to_vtxlist_local([pl_d], ngon, comm)[0]

  #Raccord single face
  if distri_jn[2] == 1:
    pl_vtx_local, pl_vtx_local_opp = _search_with_geometry(zone, face_offset, pl_face_vtx, pld_face_vtx, comm)
    assert np.all(pl_vtx_local_opp != 0)
    pl_vtx_local_list     = [pl_vtx_local.astype(pdm_dtype)]
    pl_vtx_local_opp_list = [pl_vtx_local_opp.astype(pdm_dtype)]

  else:
    pl_vtx_local, pl_vtx_local_opp, face_is_treated = _search_by_intersection(face_offset, pl_face_vtx, pld_face_vtx)

    undermined_vtx = (pl_vtx_local_opp == 0)
    pl_vtx_local_list     = [pl_vtx_local[~undermined_vtx].astype(pdm_dtype)]
    pl_vtx_local_opp_list = [pl_vtx_local_opp[~undermined_vtx].astype(pdm_dtype)]

    #Face isolée:  raccord avec plusieurs faces mais 1 seule connue par le proc (ou sans voisines)
    if comm.allreduce(not np.all(face_is_treated), op=MPI.LOR) and distri_jn[2] != 1:

      extended_pl, extended_pl_d = get_extended_pl(pl, pl_d, face_offset, pl_face_vtx, comm, face_is_treated)

      face_offset_e, pl_face_vtx_e  = facelist_to_vtxlist_local([extended_pl],   ngon, comm)[0]
      face_offset_e, pld_face_vtx_e = facelist_to_vtxlist_local([extended_pl_d], ngon, comm)[0]

      pl_vtx_local2, pl_vtx_local_opp2, _ = _search_by_intersection(face_offset_e, pl_face_vtx_e, pld_face_vtx_e)
      assert np.all(pl_vtx_local_opp2 != 0)
      pl_vtx_local_list    .append(pl_vtx_local2.astype(pdm_dtype))
      pl_vtx_local_opp_list.append(pl_vtx_local_opp2.astype(pdm_dtype))

  #Now merge vertices appearing more than once
  part_data = {'pl_vtx_opp' : pl_vtx_local_opp_list}
  PTB = PDM.PartToBlock(comm, pl_vtx_local_list, pWeight=None, partN=len(pl_vtx_local_list),
                        t_distrib=0, t_post=1, t_stride=0)
  pl_vtx = PTB.getBlockGnumCopy()
  dist_data = dict()
  PTB.PartToBlock_Exchange(dist_data, part_data)
  distri_jn_vtx_full =  par_utils.gather_and_shift(len(pl_vtx), comm, dtype=pdm_dtype)
  distri_jn_vtx =  distri_jn_vtx_full[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

  return pl_vtx, dist_data['pl_vtx_opp'], distri_jn_vtx

def generate_vertex_joins(dist_zone, comm):
  """For now only one zone is supported"""

  ngons = [elem for elem in I.getNodesFromType1(dist_zone, 'Elements_t') if elem[1][0] == 22]
  assert len(ngons) == 1
  ngon = ngons[0]

  for zgc in I.getNodesFromType1(dist_zone, 'ZoneGridConnectivity_t'):
    zgc_vtx = I.newZoneGridConnectivity(I.getName(zgc) + '#Vtx', dist_zone)
    #Todo : do not apply algo to opposite jn, juste invert values !
    for gc in I.getNodesFromType1(zgc, 'GridConnectivity_t'):
      pl_vtx, pl_vtx_opp, distri_jn = generate_jn_vertex_list(dist_zone, ngon, gc, comm)

      jn_vtx = I.newGridConnectivity(I.getName(gc)+'#Vtx', I.getValue(gc), ctype='Abutting1to1', parent=zgc_vtx)
      I.newGridLocation('Vertex', jn_vtx)
      IE.newDistribution({'Index' : distri_jn}, jn_vtx)
      I.newPointList('PointList',      pl_vtx.reshape(1,-1), parent=jn_vtx)
      I.newPointList('PointListDonor', pl_vtx_opp.reshape(1,-1), parent=jn_vtx)
      I.newIndexArray('PointList#Size', [1, distri_jn[2]], parent=jn_vtx)

