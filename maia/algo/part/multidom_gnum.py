import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

import Pypdm.Pypdm as PDM

from maia           import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils     import np_utils, py_utils, par_utils, as_pdm_gnum
from maia.algo.dist import matching_jns_tools as MJT
from maia.factory   import dist_from_part     as DFP

from maia.algo                  import dist             as MAD

from maia.transfer.part_to_dist import data_exchange    as  PTB
from maia.transfer.dist_to_part import index_exchange   as IBTP

def _get_shifted_arrays(arrays_per_dom, comm):
  shifted_per_dom = []
  offset = np.zeros(len(arrays_per_dom)+1, dtype=pdm_gnum_dtype)
  for i_dom, arrays in enumerate(arrays_per_dom):
    offset[i_dom+1] = offset[i_dom] + par_utils.arrays_max(arrays, comm)
    shifted_per_dom.append([array + offset[i_dom] for array in arrays]) # Shift (with copy)
  return offset, shifted_per_dom

def get_shifted_ln_to_gn_from_loc(parts_per_dom, location, comm):
  """ Wraps _get_zone_ln_to_gn_from_loc around multiple domains,
  shifting lngn with previous values"""
  from .point_cloud_utils import _get_zone_ln_to_gn_from_loc
  lngns_per_dom = []
  for part_zones in parts_per_dom:
    lngns_per_dom.append([_get_zone_ln_to_gn_from_loc(part, location) for part in part_zones])
  return _get_shifted_arrays(lngns_per_dom, comm)

def get_mdom_gnum_vtx(parts_per_dom, comm, merge_jns=True):

  # Get gnum shifted for vertices
  vtx_mdom_offsets, shifted_lngn = get_shifted_ln_to_gn_from_loc(parts_per_dom.values(), 'Vertex', comm)

  if not merge_jns:
    return shifted_lngn

  # Now we want to give a common gnum to the vertices connected thought GC. The outline is: 
  # 1. go back to distribute vision of GC to build graph of connected vertices
  # 2. Give to each group of connected vertices a gnum
  # 3. Send this data to the partitionned GCs using a PartToPart.
  # 4. Use the recv data to update the shifted_lngn


  # 1. Go back to distribute vision of GC to build graph of connected vertices
  dist_tree_jn = DFP._get_joins_dist_tree(parts_per_dom, comm)

  # If there is some FaceCenter interfaces, convert it to Vertex
  tree_has_face_gc = False
  face_loc_query = lambda n : PT.get_label(n) == 'GridLocation_t' and PT.get_value(n) == 'FaceCenter'
  for dom_name, parts in parts_per_dom.items():
    dist_zone = PT.get_node_from_path(dist_tree_jn, dom_name)
    has_face_gc = PT.get_node_from_predicate(dist_zone, face_loc_query) is not None
    if has_face_gc:
      # Vtx gnum is needed for face->vtx conversion.
      # Connectivities should have been already added by _get_joins_dist_tree
      vtx_lngn_l  = [MT.getGlobalNumbering(part, 'Vertex')[1] for part in parts]
      vtx_distri  = PTB._lngn_to_distri(vtx_lngn_l, comm)

      MT.newDistribution({'Vertex' : vtx_distri}, parent=dist_zone)
      tree_has_face_gc = True

  if tree_has_face_gc:
    MJT.copy_donor_subset(dist_tree_jn) #Needed for Vertexlist
    MAD.generate_jns_vertex_list(dist_tree_jn, comm, True)

    # We need to put vtx joins on partitioned trees, but recovering donor
    # and order is useless
    for dom_name, parts in parts_per_dom.items():
      dist_zone = PT.get_node_from_path(dist_tree_jn, dom_name)
      IBTP.dist_pl_to_part_pl(dist_zone, parts, ['ZoneGridConnectivity_t/GridConnectivity_t'], 'Vertex', comm)


  zone_to_dom = {dom : i for i, dom in enumerate(parts_per_dom.keys())}

  interface_ids_v = []
  interface_dom = []
  interface_dn_v = []
  
  is_gc           = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] 
  is_vtx_gc       = lambda n: is_gc(n)     and PT.Subset.GridLocation(n) == 'Vertex'
  is_vtx_gc_intra = lambda n: is_vtx_gc(n) and not MT.conv.is_intra_gc(PT.get_name(n))

  for gc_path_cur in PT.predicates_to_paths(dist_tree_jn, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', is_vtx_gc]):
    gc_path_opp = MJT.get_jn_donor_path(dist_tree_jn, gc_path_cur)

    if gc_path_cur < gc_path_opp:

      pl  = as_pdm_gnum(PT.get_node_from_path(dist_tree_jn, gc_path_cur+'/PointList')[1][0])
      pld = as_pdm_gnum(PT.get_node_from_path(dist_tree_jn, gc_path_opp+"/PointList")[1][0])

      interface_dn_v.append(pl.size)
      interface_ids_v.append(np_utils.interweave_arrays([pl,pld]))
      interface_dom.append((zone_to_dom[PT.path_head(gc_path_cur,2)], 
                            zone_to_dom[PT.path_head(gc_path_opp,2)]))

  if len(interface_dn_v) == 0: # Early return if no joins
    return shifted_lngn

  graph_idx, graph_ids, graph_dom = PDM.interface_to_graph( len(interface_dn_v), False, 
      interface_dn_v, interface_ids_v, interface_dom, comm)


  # 2. Give to each group of connected vertices a gnum (in 1 ... Nb of groups of connected vtx)
  rank_offset = par_utils.gather_and_shift(graph_idx.size-1, comm)[comm.Get_rank()]
  vtx_group_id = np.repeat(np.arange(1, graph_idx.size), np.diff(graph_idx)) + rank_offset

  # 3. Send this data to the partitionned GCs using a PartToPart.

  vtx_ggnum_graph = [graph_ids + vtx_mdom_offsets[graph_dom]] #Domain gnum on graph side

  vtx_ggnum_parts = []
  for vtx_mdom_offset, parts in zip(vtx_mdom_offsets, parts_per_dom.values()):
    for part in parts:
      vtx_gnum = as_pdm_gnum(MT.getGlobalNumbering(part, 'Vertex')[1])
      for gc in PT.get_children_from_predicates(part, ['ZoneGridConnectivity_t', is_vtx_gc_intra]):
        pl = PT.get_child_from_name(gc, 'PointList')[1][0]
        vtx_ggnum_parts.append(vtx_gnum[pl-1] + vtx_mdom_offset) #Domain gnum on part side

  # Create PTP : indirection part1topart2 is just the identity
  PTP = PDM.PartToPart(comm, vtx_ggnum_graph, vtx_ggnum_parts, \
                       [np.arange(vtx_ggnum_graph[0].size+1, dtype=np.int32)], vtx_ggnum_graph)

  request = PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                      PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                      [vtx_group_id])
  _, vtx_group_id_recv = PTP.wait(request)

  # 4. Use the recv data to update the shifted_lngn

  count = 0
  for shifted_lngn_dom, parts in zip(shifted_lngn, parts_per_dom.values()):
    for shifted_lngn_part, part in zip(shifted_lngn_dom, parts):
      for gc in PT.get_children_from_predicates(part, ['ZoneGridConnectivity_t', is_vtx_gc_intra]):
        pl = PT.get_child_from_name(gc, 'PointList')[1][0]
        # We received id starting at 1 so shift it to the end of the internal gc gnums
        shifted_lngn_part[pl-1] = vtx_group_id_recv[count] + vtx_mdom_offsets[-1]
        count += 1

  # Finally we just have to fill holes
  from .point_cloud_utils import create_sub_numbering
  shifted_lngn_contiguous = create_sub_numbering(py_utils.to_flat_list(shifted_lngn), comm)
  shifted_lngn_contiguous = py_utils.to_nested_list(shifted_lngn_contiguous, 
      [len(parts) for parts in parts_per_dom.values()])
  return shifted_lngn_contiguous

