import maia.pytree as PT
from   maia.utils  import np_utils, s_numbering, as_pdm_gnum
from   maia import npy_pdm_gnum_dtype as pdm_dtype
from   maia.factory  import dist_from_part
from   .point_cloud_utils  import create_sub_numbering

import numpy as np

LOC_TO_DIM   = {'Vertex':0,
                'EdgeCenter':1,
                'FaceCenter':2, 'IFaceCenter':2, 'JFaceCenter':2, 'KFaceCenter':2,
                'CellCenter':3}

DIMM_TO_DIMF = { 0: {'Vertex':'Vertex'},
               # 1: {'Vertex': None,    'EdgeCenter':None, 'FaceCenter':None, 'CellCenter':None},
                 2: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'CellCenter'},
                 3: {'Vertex':'Vertex', 'EdgeCenter':'EdgeCenter', 'FaceCenter':'FaceCenter', 'CellCenter':'CellCenter'}}

def discover_containers(part_zones, container_name, patch_name, patch_type, comm):
  mask_zone = ['MaskedZone', None, [], 'Zone_t']
  dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, container_name, comm, \
      child_list=['GridLocation', 'BCRegionName', 'GridConnectivityRegionName'])
  
  fields_query = lambda n: PT.get_label(n) in ['DataArray_t', patch_type]
  dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, [container_name, fields_query], comm)
  mask_container = PT.get_child_from_name(mask_zone, container_name)
  if mask_container is None:
    raise ValueError(f"[maia-extract_part] asked container \"{container_name}\" for exchange is not in tree")
  if PT.get_child_from_label(mask_container, 'DataArray_t') is None:
    return None, '', False
  patch_node     = PT.get_child_from_name(mask_container, patch_name)

  # > Manage BC and GC ZSR
  ref_zsr_node    = mask_container
  bc_descriptor_n = PT.get_child_from_name(mask_container, 'BCRegionName')
  gc_descriptor_n = PT.get_child_from_name(mask_container, 'GridConnectivityRegionName')
  assert not (bc_descriptor_n and gc_descriptor_n)
  if bc_descriptor_n is not None:
    bc_name      = PT.get_value(bc_descriptor_n)
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneBC_t', bc_name], comm, child_list=[patch_name, 'GridLocation_t'])
    ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneBC_t/{bc_name}')
    patch_node   = PT.get_child_from_predicates(ref_zsr_node, f'{patch_name}')
    assert patch_node is not None, 'Asked patch unfound for subregion extent.'
  elif gc_descriptor_n is not None:
    gc_name      = PT.get_value(gc_descriptor_n)
    dist_from_part.discover_nodes_from_matching(mask_zone, part_zones, ['ZoneGridConnectivity_t', gc_name], comm, child_list=[patch_name, 'GridLocation_t'])
    ref_zsr_node = PT.get_child_from_predicates(mask_zone, f'ZoneGridConnectivity_t/{gc_name}')
    patch_node   = PT.get_child_from_predicates(ref_zsr_node, f'{patch_name}')
    assert patch_node is not None, 'Asked patch unfound for subregion extent.'
  
  if PT.get_label(mask_container)=='ZoneSubRegion_t' and patch_node is None:
    raise ValueError('Asked patch unfound for ZSR container extent.')

  grid_location = PT.Subset.GridLocation(ref_zsr_node)
  partial_field = PT.get_child_from_name(ref_zsr_node, patch_name) is not None
  return mask_container, grid_location, partial_field

def local_pl_offset(part_zone, dim):
  """
  Return the shift related to the element of the dimension to apply to a point_list so it starts to 1.
  This function assumes that there will be only one type of element per dimension on zone
  (because isosurface and extract_part returns trees with this property).
  Works only for ngon/nface 3D meshes and elements meshes.
  """
  if   dim == 3:
    nface = PT.Zone.NFaceNode(part_zone)
    if nface is None:
      # Extract_part and isosurfaces use trees with NGon and NFace
      raise ValueError("NGon trees must have NFace connectivity to extract pl offset")
    return PT.Element.Range(nface)[0] - 1
  elif dim == 2:
    if PT.Zone.has_ngon_elements(part_zone):
      ngon = PT.Zone.NGonNode(part_zone)
      return PT.Element.Range(ngon)[0] - 1
    else:
      tri_or_quad_elts = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.Dimension(n)==2
      elt_n     = PT.get_child_from_predicate(part_zone, tri_or_quad_elts)
      return PT.Element.Range(elt_n)[0] - 1
  elif dim == 1:
    bar_elts  = lambda n: PT.get_label(n)=='Elements_t' and PT.Element.Dimension(n)==1
    elt_n     = PT.get_child_from_predicate(part_zone, bar_elts)
    return PT.Element.Range(elt_n)[0] - 1
  else:
    return 0

def get_relative_pl(container, part_zone):
  """Return the point_list node related to a container (from BC, GC or itself)."""
  if PT.get_label(container)=="FlowSolution_t":
    relative_n = container
  else:
    relative_n = PT.get_node_from_path(part_zone, PT.Subset.ZSRExtent(container, part_zone))
  return PT.get_child_from_name(relative_n, "PointList")

def get_partial_container_stride_and_order(part_zones, container_name, gridLocation, ptp, comm):
  """
  Return two list of arrays :
    - pl_gnum1 : the order to apply to container fields defined with a point_list to fit the gnum1_come_from order
    - stride   : stride of shape gnum1_come_from indexing which elt will get data
  """
  pl_gnum1 = list()
  stride   = list()

  for i_part, part_zone in enumerate(part_zones):
    container = PT.get_child_from_name(part_zone, container_name)
    if container is not None:
      # > Get the right node to get PL (if ZSR linked to BC or GC)
      point_list_n = get_relative_pl(container, part_zone)
      point_list   = PT.get_value(point_list_n)[0] - local_pl_offset(part_zone, LOC_TO_DIM[gridLocation]) # Gnum start at 1

    # Get p2p gnums
    part_gnum1_idx = ptp.get_gnum1_come_from() [i_part]['come_from_idx'] # Get partition order
    part_gnum1     = ptp.get_gnum1_come_from() [i_part]['come_from']     # Get partition order
    ref_lnum2      = ptp.get_referenced_lnum2()[i_part]                  # Get partition order

    if container is None or point_list.size==0 or ref_lnum2.size==0:
      stride_tmp   = np.zeros(part_gnum1_idx[-1],dtype=np.int32)
      pl_gnum1_tmp = np.empty(0,dtype=np.int32)
      stride  .append(stride_tmp)
      pl_gnum1.append(pl_gnum1_tmp)
    else:
      order    = np.argsort(ref_lnum2)                 # Sort order of point_list ()
      # Next lines create 2 arrays : pl_mask and true_idx
      # pl_mask (size=pl.size, bool) tell for each pl element if it appears in ref_lnum2
      # true_idx (size=pl_mask.sum(), int) give for each pl element its position in ref_lnum2
      # (if pl_mask is True)
      idx      = np.searchsorted(ref_lnum2,point_list,sorter=order)
      # Take is equivalent to order[idx], but if idx > order.size, last elt of order is taken
      pl_mask  = point_list==ref_lnum2[np.take(order, idx, mode='clip')]
      true_idx = idx[pl_mask]

      # Number of part1 elements in an element of part2 
      n_elt_of1_in2 = np.diff(part_gnum1_idx)[true_idx]

      sort_true_idx = np.argsort(true_idx)

      # PL in part2 order
      pl_gnum1_tmp = np.arange(0, point_list.shape[0], dtype=np.int32)[pl_mask][sort_true_idx]
      pl_gnum1_tmp = np.repeat(pl_gnum1_tmp, n_elt_of1_in2)
      pl_gnum1.append(pl_gnum1_tmp)

      # PL in gnum1 order
      pl_to_gnum1_start = part_gnum1_idx[true_idx]         
      pl_to_gnum1_stop  = pl_to_gnum1_start+n_elt_of1_in2
      pl_to_gnum1 = np_utils.multi_arange(pl_to_gnum1_start, pl_to_gnum1_stop)
      
      # Stride variable
      stride_tmp = np.zeros(part_gnum1_idx[-1], dtype=np.int32)
      stride_tmp[pl_to_gnum1] = 1
      stride.append(stride_tmp)
    
  return pl_gnum1, stride

def build_intersection_numbering(part_tree, extract_zones, mesh_dim, container_name, grid_location, etb, comm):

  parent_lnum_path = {'Vertex'     :'parent_lnum_vtx',
                      'IFaceCenter':'parent_lnum_cell',
                      'JFaceCenter':'parent_lnum_cell',
                      'KFaceCenter':'parent_lnum_cell',
                      'CellCenter' :'parent_lnum_cell'}
  LOC_TO_GNUM = {'Vertex'     :'Vertex',
                 'IFaceCenter':'Face',
                 'JFaceCenter':'Face',
                 'KFaceCenter':'Face',
                 'CellCenter' :'Cell',
                }

  part1_pr       = list()
  part1_in_part2 = list()
  partial_gnum   = list()
  for extract_zone in extract_zones:

    zone_name = PT.get_name(extract_zone)
    part_zone = PT.get_node_from_name_and_label(part_tree, zone_name, 'Zone_t')
    
    parent_part1_pl = etb[zone_name][parent_lnum_path[grid_location]]

    subset_n = PT.get_child_from_name(part_zone,container_name)
    if subset_n is not None:
      pr = PT.get_value(PT.Subset.getPatch(subset_n))
      i_ar = np.arange(min(pr[0]), max(pr[0])+1)
      j_ar = np.arange(min(pr[1]), max(pr[1])+1).reshape(-1,1)
      k_ar = np.arange(min(pr[2]), max(pr[2])+1).reshape(-1,1,1)
      part2_pl = s_numbering.ijk_to_index_from_loc(i_ar, j_ar, k_ar, grid_location, PT.Zone.VertexSize(part_zone)).flatten()

      lnum2 = np.searchsorted(part2_pl, parent_part1_pl) # Assume sorted
      mask  = parent_part1_pl==np.take(part2_pl,lnum2,mode='clip')
      lnum2 = lnum2[mask]
      part1_in_part2.append(lnum2)
      pl1   = np.isin(parent_part1_pl, part2_pl, assume_unique=True) # Assume unique because pr
      pl1   = np.where(pl1)[0]+1

      if pl1.size==0:
        part1_pr.append(np.empty(0, dtype=np.int32))
        partial_gnum.append(np.empty(0, dtype=pdm_dtype))
        continue # Pass if no recovering

      vtx_size = PT.Zone.VertexSize(extract_zone)
      if vtx_size.size==2:
        vtx_size = np.concatenate([vtx_size,np.array([1], dtype=vtx_size.dtype)])
      part1_ijk = s_numbering.index_to_ijk_from_loc(pl1, DIMM_TO_DIMF[mesh_dim][grid_location], vtx_size)
      part1_pr.append(np.array([[min(part1_ijk[0]),max(part1_ijk[0])],
                                [min(part1_ijk[1]),max(part1_ijk[1])],
                                [min(part1_ijk[2]),max(part1_ijk[2])]]))

      part2_elt_gnum = PT.maia.getGlobalNumbering(part_zone, LOC_TO_GNUM[grid_location])[1]
      
      partial_gnum.append(as_pdm_gnum(part2_elt_gnum[part2_pl[lnum2]-1]))
    else:
      part1_in_part2.append(np.empty(0, dtype=np.int32))
      part1_pr.append(np.empty(0, dtype=np.int32))
      partial_gnum.append(np.empty(0, dtype=pdm_dtype))

  part1_gnum1 = create_sub_numbering(partial_gnum, comm)

  return part1_pr, part1_gnum1, part1_in_part2
