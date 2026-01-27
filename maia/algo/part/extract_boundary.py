import numpy   as np

from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT
from maia.utils     import np_utils, s_numbering, pr_utils
from maia.transfer  import utils as te_utils

from .point_cloud_utils import create_sub_numbering
from maia.pytree.maia.check_tree import check_cgns_part_tree
from maia import npy_pdm_gnum_dtype as pdm_dtype

def _struct2d_connectivity(zone: CGNSTree) -> Tuple[NDArray, NDArray]:
  n_vtx_i, n_vtx_j = PT.Zone.VertexSize(zone)
  n_vtx = n_vtx_i*n_vtx_j
  nf_i = n_vtx_i * (n_vtx_j-1)
  nf_j = n_vtx_j * (n_vtx_i-1)
  edge_vtx_idx = 2*np.arange(nf_i+nf_j+1, dtype=np.int32)
  edge_vtx = np.empty(edge_vtx_idx[-1], np.int32)
  _edge_vtx_i, _edge_vtx_j = edge_vtx[:2*nf_i], edge_vtx[2*nf_i:]
  # Fill IEdge
  _edge_vtx_i[0::2] = np.arange(1, n_vtx - n_vtx_i + 1)
  _edge_vtx_i[1::2] = np.arange(1 + n_vtx_i, 1 + n_vtx)
  # Swap bnd I edges
  _edge_vtx_i[0::2*n_vtx_i] += n_vtx_i
  _edge_vtx_i[1::2*n_vtx_i] -= n_vtx_i
  # Fill JEdge
  mask = np.ones(2*n_vtx, bool)
  mask[0::2*n_vtx_i] = False
  mask[2*n_vtx_i-1::2*n_vtx_i] = False
  _edge_vtx_j[:] = np_utils.repeated_arange(2, 1, n_vtx+1)[mask]
  # Swap bnd J edges
  n_edge_j = n_vtx_i - 1
  ymax = _edge_vtx_j[-2*n_edge_j:].reshape((-1,2))
  _edge_vtx_j[-2*n_edge_j:] = np.flip(ymax, axis=1).reshape(-1)
  #tmp = ymax[::2].copy()
  #ymax[0::2] = ymax[1::2]
  #ymax[1::2] = tmp
  return edge_vtx_idx, edge_vtx

def _struct3d_connectivity(zone: CGNSTree) -> Tuple[NDArray, NDArray]:
  nf_i, nf_j, nf_k = PT.Zone.FaceSize(zone)
  n_face_tot = nf_i + nf_j + nf_k

  bounds = [b + 1 for b in [0, nf_i, nf_i + nf_j, nf_i + nf_j + nf_k]]

  face_vtx_idx = 4*np.arange(0, n_face_tot+1, dtype=np.int32)
  face_vtx, _ = s_numbering.ngon_dconnectivity_from_gnum(bounds, PT.Zone.VertexSize(zone), dtype=np.int32)
  return face_vtx_idx, face_vtx

def _pr_to_face_pl(n_vtx_zone: Tuple[int, ...], pr: NDArray, input_loc: str) -> NDArray:
  """
  Transform a (partitioned) PointRange pr of any location input_loc into a PointList
  supported by the faces or edges. n_vtx_zone is the number of vertices of the zone to which the
  pr belongs. Output face are numbered using s_numb conventions (i faces, then j faces, then
  k faces in increasing i,j,k for each group)
  """
  # NB the hack in this function is to call compute_pointList_from_pointRanges with
  # outputloc == 'FaceCenter' even when we want to produce a 2D / EdgeCenter BC
  # We can do it if we extend input args *and* if we do the shift manually for JEdge
  # (since nFacesI evaluates to 0 in func)

  cell_dim = len(n_vtx_zone)
  bnd_axis = PT.Subset.normal_axis(PT.new_BC(point_range=pr, loc=input_loc))

  # It is safer to reuse slabs to manage all cases (eg input location or reversed pr)
  bc_size = pr_utils.transform_bnd_pr_size(pr, input_loc, "FaceCenter")

  slab = np.ones((3,2), order='F', dtype=np.int32)
  slab[0:cell_dim,0] = pr[:,0]
  slab[0:cell_dim,1] = bc_size + pr[:,0] - 1
  slab[bnd_axis,:] += pr_utils.normal_index_shift(pr, n_vtx_zone, bnd_axis, input_loc, "FaceCenter")

  _n_vtx_zone = n_vtx_zone[:cell_dim] + tuple(1 for _ in range(3-cell_dim))

  pl = pr_utils.compute_pointList_from_pointRanges([slab], _n_vtx_zone,  ['I', 'J', 'K'][bnd_axis]+'FaceCenter')
  if cell_dim == 2 and bnd_axis == 1:
    pl += n_vtx_zone[0]*(n_vtx_zone[1]-1)

  return pl

def _extract_sub_connectivity(array_idx: NDArray, array: NDArray, sub_elts: NDArray) -> Tuple[NDArray, ...]:
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


def extract_faces_mesh(zone: CGNSTree, face_ids: NDArray) -> Tuple[NDArray, ...]:
  """
  Extract a sub mesh from a U or S zone and a (flat) list of face ids to extract :
  create the sub ngon connectivity and extract the coordinates of vertices 
  belonging to the sub mesh.
  For S zone, faces to extract must be converted from i,j,k to index before processing
  """
  zone_dim = PT.Zone.CellDimension(zone)
  # NGon Extraction
  if PT.Zone.Type(zone) == 'Unstructured':
    if PT.Zone.has_ngon_elements(zone):
      bnd_elts = MT.Zone.EdgeNode(zone) if zone_dim == 2 else PT.Zone.NGonNode(zone)
      _face_vtx = MT.Element.connectivity(bnd_elts)
      face_vtx_idx = _face_vtx.displs
      face_vtx = _face_vtx.values
    else: # Zone has std elements
      sections_2d = PT.Zone.get_ordered_elements_per_dim(zone)[zone_dim-1]
      elem_size_list = [PT.Element.Size(elt) for elt in sections_2d]
      face_n_vtx_list = [PT.Element.NVtx(elt) for elt in sections_2d]
      elem_cnt_list = [PT.get_np_value(PT.find_node_from_name(elt, 'ElementConnectivity')) for elt in sections_2d]
      _, face_vtx = np_utils.concatenate_np_arrays(elem_cnt_list, dtype=np.int32)
      face_vtx_idx = np_utils.sizes_to_indices(np.repeat(face_n_vtx_list, elem_size_list), dtype=np.int32)
  elif PT.Zone.Type(zone) == 'Structured':
    # For S zone, create a NGon connectivity
    if zone_dim == 3:
      face_vtx_idx, face_vtx = _struct3d_connectivity(zone)
    elif zone_dim == 2:
      face_vtx_idx, face_vtx = _struct2d_connectivity(zone)


  ex_face_vtx_idx, ex_face_vtx, vtx_ids = _extract_sub_connectivity(face_vtx_idx, face_vtx, face_ids)
  
  # Vertex extraction
  cx, cy, cz = PT.Zone.coordinates(zone)
  assert (cx is not None) and (cy is not None) and (cz is not None)
  if PT.Zone.Type(zone) == 'Unstructured':
    ex_cx = cx[vtx_ids-1]
    ex_cy = cy[vtx_ids-1]
    ex_cz = cz[vtx_ids-1]
  elif PT.Zone.Type(zone) == 'Structured':
    if zone_dim == 2:
      i_idx, j_idx = s_numbering.index_to_ij(vtx_ids, PT.Zone.VertexSize(zone))
    else:
      i_idx, j_idx, k_idx = s_numbering.index_to_ijk(vtx_ids, PT.Zone.VertexSize(zone))
    ex_cx = cx[i_idx-1, j_idx-1].flatten() if zone_dim == 2 else cx[i_idx-1, j_idx-1, k_idx-1].flatten()
    ex_cy = cy[i_idx-1, j_idx-1].flatten() if zone_dim == 2 else cy[i_idx-1, j_idx-1, k_idx-1].flatten()
    ex_cz = cz[i_idx-1, j_idx-1].flatten() if zone_dim == 2 else cz[i_idx-1, j_idx-1, k_idx-1].flatten()

  return ex_cx, ex_cy, ex_cz, ex_face_vtx_idx, ex_face_vtx, vtx_ids


def extract_surf_from_bc_single(part_zones: List[CGNSTree], 
                                bc_predicate: Callable[[CGNSTree], bool], 
                                comm: MPIComm) -> List[CGNSTree]:
  """
  From a list of partitioned zones (coming from the same initial domain), get the list
  of faces (or edge, depending on zone dimension)
  belonging to any bc satisfiyng bc_predicate and extract the surfacic mesh.
  In addition, compute a new global numbering (over the procs and the part_zones) of the extracted
  faces and vertex (starting a 1 without gap)
  """
  for part_zone in part_zones:
    check_cgns_part_tree(part_zone)
  
  parent_face_lngn_l = []
  parent_vtx_lngn_l  = []
  ext_zones = []
  for zone in part_zones:
    zone_dim = PT.Zone.CellDimension(zone)
    wanted_loc = 'EdgeCenter' if zone_dim == 2 else 'FaceCenter'
    is_relevant_bc = PT.pred.label_is('BC_t') & PT.pred.NodePredicate(bc_predicate)

    bc_face_ids:List[NDArray]
    if PT.Zone.Type(zone) == 'Unstructured':
      bc_nodes = PT.get_children_from_predicates(zone, ['ZoneBC_t', is_relevant_bc & PT.pred.has_location(wanted_loc)])
      bc_face_ids = [PT.get_np_value(PT.find_child_from_name(bc_node, 'PointList'))[0] for bc_node in bc_nodes]
    else:
      n_vtx_z = PT.Zone.VertexSize(zone)
      bc_nodes = PT.get_children_from_predicates(zone, ['ZoneBC_t', is_relevant_bc])
      bc_face_ids = [_pr_to_face_pl(n_vtx_z, PT.get_np_value(PT.find_child_from_name(bc_node, 'PointRange')), PT.Subset.GridLocation(bc_node))[0] \
          for bc_node in bc_nodes]

    _, bc_face_ids_cat = np_utils.concatenate_np_arrays(bc_face_ids, np.int32)
    # Shift the bc_face_ids to make it start a 1
    if PT.Zone.Type(zone) == 'Unstructured':
      try:
        bc_face_ids_cat -= (PT.Zone.get_elt_range_per_dim(zone)[PT.Zone.CellDimension(zone)-1][0] - 1)
      except Exception:
        raise RuntimeError("Unable to extract unordered faces")

    cx, cy, cz, bc_face_vtx_idx, bc_face_vtx, bc_vtx_ids = extract_faces_mesh(zone, bc_face_ids_cat)

    vtx_ln_to_gn_zone = MT.globalnumbering_value(zone, 'Vertex')

    if PT.Zone.Type(zone) == 'Unstructured' and not PT.Zone.has_ngon_elements(zone):
      elt_2d_nodes = PT.Zone.get_ordered_elements_per_dim(zone)[zone_dim-1]
      elt_2d_gnums = [MT.globalnumbering_value(elt, "Sections") for elt in elt_2d_nodes]
      face_ln_to_gn_zone = np.concatenate(elt_2d_gnums) if len(elt_2d_nodes) else np.empty(0, dtype=pdm_dtype)
    else:
      face_ln_to_gn_zone = te_utils.get_entities_numbering(zone)[zone_dim-1] # Face if dim==3; Edge if dim == 2
      assert (face_ln_to_gn_zone) is not None

    parent_face_lngn_l.append(face_ln_to_gn_zone[bc_face_ids_cat-1])
    parent_vtx_lngn_l .append(vtx_ln_to_gn_zone[bc_vtx_ids-1]  )

    # Create extracted zone
    n_face = bc_face_vtx_idx.size - 1
    n_vtx = cz.size
    ext_zone = PT.new_Zone(PT.get_name(zone), type='Unstructured', size=[[n_vtx, n_face, 0]])
    PT.new_GridCoordinates(fields={f'Coordinate{d}' : c for d,c in zip('XYZ', (cx,cy,cz))}, parent=ext_zone)
    if zone_dim - 1 == 2:
      PT.new_NGonElements(erange=[1, n_face], eso=bc_face_vtx_idx, ec=bc_face_vtx, parent=ext_zone)
    else:
      PT.new_Elements(type='BAR_2', erange=[1, n_face], econn=bc_face_vtx, parent=ext_zone)
    ext_zones.append(ext_zone)

  # Compute extracted gnum from parents
  bc_face_lngn_l = create_sub_numbering(parent_face_lngn_l, comm)
  bc_vtx_lngn_l  = create_sub_numbering(parent_vtx_lngn_l, comm)

  for i, ext_zone in enumerate(ext_zones):
    PT.new_DiscreteData(loc='CellCenter', fields={'Parent' : parent_face_lngn_l[i]}, parent=ext_zone)
    MT.new_GlobalNumbering({'Vertex' : bc_vtx_lngn_l[i], 'Cell' : bc_face_lngn_l[i]}, parent=ext_zone)

  return ext_zones

def extract_surf_from_bc(part_tree: CGNSPartTree, 
                         bc_predicate: Callable[[CGNSTree], bool], 
                         comm: MPIComm) -> CGNSPartTree:
  # Light / local version of extract_part for WallDistance

  from maia.factory.dist_from_part     import get_parts_per_blocks
  ext_tree = PT.new_CGNSTree()
  for dist_zone_path, part_zones in get_parts_per_blocks(part_tree, comm).items():
    part_base = PT.find_child_from_name(part_tree, PT.utils.path_head(dist_zone_path))
    new_dim = [PT.Base.CellDimension(part_base)-1, 3]
    ext_base = PT.update_child(ext_tree, PT.get_name(part_base), PT.get_label(part_base), new_dim)
    ext_zones = extract_surf_from_bc_single(part_zones, bc_predicate, comm)
    for ext_zone in ext_zones:
      PT.add_child(ext_base, ext_zone)
  
  return ext_tree