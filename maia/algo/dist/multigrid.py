import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP
from maia.transfer import utils as tr_utils

from maia.utils import s_numbering, par_utils, np_utils

from maia.typing import *

def _local_bounds(ini_start_loc:int, ini_end_loc:int, g_start:int, g_end:int) -> Tuple[int, int]:
  ini_size = ini_end_loc - ini_start_loc
  r_start = max(ini_start_loc, g_start) - ini_start_loc
  r_end   = min(ini_end_loc, g_end) - ini_start_loc
  
  return min(r_start, ini_size), max(r_end, 0)

def local_bounds(distri:NDArray, g_start:int, g_end:int) -> Tuple[int, int]:
  """ Compute the local start/end indices that should be used to extract a slice of 
  a distributed array, restricted to global [start:end[ interval """
  return _local_bounds(distri[0], distri[1], g_start, g_end)

def path_to_level(path:CGNSPath) -> int:
  # Return the MG level of a path, starting at base level
  return int(PT.utils.path_head(path, 1).rsplit('.LV',1)[1])

def n_level(tree) -> int:
  return max(path_to_level(PT.get_name(b)) for b in PT.get_all_CGNSBase_t(tree))
  
def single_level_tree(tree:CGNSTree, lvl:int) -> CGNSTree:
  # Return a containing only Bases of specified level
  return PT.new_node('CGNSTree', 'CGNSTree_t', children=[b for b in PT.get_all_CGNSBase_t(tree) if PT.get_name(b).endswith(f'.LV{lvl}')])

def update_path_level(path:CGNSPath, new_lvl:int) -> CGNSPath:
  cur_lvl = path_to_level(path)
  return PT.utils.update_path_elt(path, 0, lambda s:s[:-len(str(cur_lvl))]+str(new_lvl))


def convert_s_to_ngon(tree:CGNSDistTree, comm:MPIComm):
  """ Implements convert_s_to_ngon for multigrid trees """

  nb_lvl = n_level(tree)
  zone_path_to_vertex_size = {path: PT.Zone.VertexSize(PT.find_node_from_path(tree, path))
                              for path in PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')}

  maia.algo.dist.convert_s_to_ngon(tree, comm)

  for zone_path, n_vtx_s in zone_path_to_vertex_size.items():
    n_vtx_s_coarse = [s//2 + 1 for s in n_vtx_s]

    if path_to_level(zone_path) == nb_lvl:
      # Skip last level
      continue

    zone = PT.find_node_from_path(tree, zone_path)
    mg_cell_info = PT.find_child_from_name(zone, "MultiGridCellInfo")
    i_c_idx = PT.get_np_value(PT.find_child_from_name(mg_cell_info, "ICoarseIdx"))
    j_c_idx = PT.get_np_value(PT.find_child_from_name(mg_cell_info, "JCoarseIdx"))
    k_c_idx = PT.get_np_value(PT.find_child_from_name(mg_cell_info, "KCoarseIdx"))
    u_c_idx = s_numbering.ijk_to_index_from_loc(i_c_idx, j_c_idx, k_c_idx, "CellCenter", n_vtx_s_coarse)
    PT.rm_children_from_name(mg_cell_info, "*CoarseIdx")
    PT.new_DataArray("CoarseIdx", u_c_idx, parent=mg_cell_info)

    for bc in PT.get_nodes_from_label(zone, "BC_t"):
      mg_face_info = PT.find_child_from_name(bc, "MultiGridBCFaceInfo")
      i_c_idx = PT.get_np_value(PT.find_node_from_name(mg_face_info, "ICoarseIdx"))
      j_c_idx = PT.get_np_value(PT.find_node_from_name(mg_face_info, "JCoarseIdx"))
      k_c_idx = PT.get_np_value(PT.find_node_from_name(mg_face_info, "KCoarseIdx"))
      bc_loc = PT.get_str_value(PT.find_node_from_path(mg_face_info, "BCStructuredLocation"))
      u_c_idx = s_numbering.ijk_to_index_from_loc(i_c_idx, j_c_idx, k_c_idx, bc_loc, n_vtx_s_coarse)

      # Now BC and BCDS should be FaceCenter with same PointList
      assert np.array_equal(PT.get_np_value(PT.find_child_from_name(bc, 'PointList')),
                            PT.get_np_value(PT.find_child_from_name(mg_face_info, 'PointList')))
      # --> Erase everything
      ds = PT.new_BCData('DirichletData', {'CoarseIdx' : u_c_idx})
      PT.set_children(mg_face_info, [ds])

def merge_connected_zones(tree:CGNSDistTree, comm:MPIComm, **kwargs):
  """ Implements merge_connected_zones for multigrid trees """
                          
  nb_lvl = n_level(tree)
  # Use any level to get connected zones
  tree_lvl_0 = PT.new_node('CGNSTree', 'CGNSTree_t', children=[b for b in PT.get_children(tree) if PT.get_name(b).endswith('.LV0')])
  groups = PT.Tree.find_connected_zones(tree_lvl_0)

  for lvl in range(nb_lvl+1):
    for i, group in enumerate(groups):
      lvl_group = [update_path_level(path, lvl) for path in group]
      n_cell_group_cur = [PT.Zone.n_cell(PT.find_node_from_path(tree, p)) for p in lvl_group]
      n_face_group_cur = [PT.Zone.n_face(PT.find_node_from_path(tree, p)) for p in lvl_group]


      if lvl < nb_lvl:
        lvl_group_next    = [update_path_level(path, lvl+1) for path in group]
        n_cell_group_next = [PT.Zone.n_cell(PT.find_node_from_path(tree, p)) for p in lvl_group_next]
        n_face_group_next = [PT.Zone.n_face(PT.find_node_from_path(tree, p)) for p in lvl_group_next]
      
      # Before merging, store some data under the BCs of the the current level:
      # --> OldPointList is the current PointList, shifted, before merging
      # --> CoarseIdx is the face if of coarse face, shifted, before merging
      # In both case, shift is computed from all zones belonging to the current group
      # We will need this data to update CoarseIdx after merging

      offset_cur = 0
      offset_next = 0
      for j, path in enumerate(lvl_group):
        zone = PT.find_node_from_path(tree, path)
        for bc in PT.get_children_from_predicates(zone, "ZoneBC_t/BC_t"):
          pl   = PT.get_np_value(PT.find_child_from_name(bc, "PointList"))

          bcds = PT.update_child(bc, 'MultiGridBCFaceInfo', label='BCDataSet_t')
          dd   = PT.update_child(bcds, 'DirichletData', label='BCData_t')
          PT.new_DataArray('OldPointList', pl[0] + offset_cur, parent=dd)

          if lvl < nb_lvl:
            node = PT.find_node_from_path(bc, 'MultiGridBCFaceInfo/DirichletData/CoarseIdx')
            node[1] += offset_next

        offset_cur += n_face_group_cur[j]
        if lvl < nb_lvl:
          offset_next += n_face_group_next[j]


      # Merge group
      output_path = f"{PT.utils.path_head(lvl_group[0])}/mergedZone{i}"
      maia.algo.dist.merge_zones(tree, lvl_group, comm, output_path=output_path, **kwargs)
      merged_zone = PT.find_node_from_path(tree, output_path)

      # Postreat to update CoarseIdx
      # --> Update cells for current level
      if lvl < nb_lvl:
        cell_distrib = MT.distribution_value(merged_zone, 'Cell')
        c_u_idx = PT.get_np_value(PT.find_node_from_path(merged_zone, "MultiGridCellInfo/CoarseIdx"))
        start = 0
        shift_value = 0
        for j in range(len(group)):
          end = start + n_cell_group_cur[j]
          # Restrict c_u_idx to cell_ids coming from zone n°j (in current group)
          loc_start, loc_end = local_bounds(cell_distrib, start, end)
          c_u_idx[loc_start:loc_end] += shift_value

          start = end
          shift_value += n_cell_group_next[j]

      # --> Update faces for *previous* level (since we need current and previous level merged)
      if lvl > 0:
        output_path_prev = update_path_level(output_path, lvl-1)
        merged_zone_prev = PT.find_node_from_path(tree, output_path_prev)

        bcs_cur  = PT.get_children_from_predicates(merged_zone,      'ZoneBC_t/BC_t')
        bcs_prev = PT.get_children_from_predicates(merged_zone_prev, 'ZoneBC_t/BC_t')

        cur_path = 'MultiGridBCFaceInfo/DirichletData/OldPointList'
        coarse_path = 'MultiGridBCFaceInfo/DirichletData/CoarseIdx'
        src = [PT.get_np_value(PT.find_node_from_path(bc, cur_path))  for bc in bcs_cur]
        tgt = [PT.get_np_value(PT.find_node_from_path(bc, coarse_path)) for bc in bcs_prev]

        new_coarse_id = [PT.get_np_value(PT.find_node_from_path(bc, 'PointList'))[0] for bc in bcs_cur]

        updated_coarse_id = EP.part_to_part(new_coarse_id, src, tgt, comm)
        for k,bc in enumerate(bcs_prev):
          node = PT.find_node_from_path(bc, 'MultiGridBCFaceInfo/DirichletData/CoarseIdx')
          PT.set_value(node, updated_coarse_id[k])
          # Meanwhile, cleanup prev level
          PT.rm_node_from_path(bc, 'MultiGridBCFaceInfo/PointList')
          PT.rm_node_from_path(bc, 'MultiGridBCFaceInfo/GridLocation')
          PT.rm_node_from_path(bc, 'MultiGridBCFaceInfo/DirichletData/OldPointList')

      if lvl == nb_lvl:
        # Cleanup last level
        for bc in PT.get_nodes_from_label(merged_zone, 'BC_t'):
          PT.rm_children_from_name(bc, 'MultiGridBCFaceInfo')
        

def partition_dist_tree(dist_tree:CGNSDistTree, comm:MPIComm, **kwargs) -> CGNSPartTree:
  """ Implements partition_dist_tree for multigrid trees """

  nb_lvl = n_level(dist_tree)

  data_transfer = kwargs.get('data_transfer', [])

  # Split coarser level
  dt_lvl_coarse = single_level_tree(dist_tree, nb_lvl)
  mg_pt = maia.factory.partition_dist_tree(dt_lvl_coarse, comm, **kwargs)

  # For others levels, zone_to_part and target part are recomputed
  kwargs.pop('zone_to_parts', None)
  kwargs.pop('target_part', None)

  for lvl in reversed(range(nb_lvl)):
    dt_cur_lvl = single_level_tree(dist_tree, lvl)
    pt_coarse_lvl = single_level_tree(mg_pt, lvl+1)
    
    zone_to_parts = dict()
    target_part = list()
    for zone_path in PT.predicates_to_paths(dt_cur_lvl, 'CGNSBase_t/Zone_t'):
      # Get partitioned zones for coarser level
      coarse_p_zones = tr_utils.get_partitioned_zones(pt_coarse_lvl, update_path_level(zone_path, lvl+1))
      start_rank_idx = par_utils.gather_and_shift(len(coarse_p_zones), comm)
      coarse_gnum = list()
      i_coarse_part = list()
      for ipart, coarse_p_zone in enumerate(coarse_p_zones):
        tgt_part_id = start_rank_idx[comm.rank] + ipart
        _coarse_gnum = MT.globalnumbering_value(coarse_p_zone, "Cell")
        coarse_gnum.append(_coarse_gnum)
        i_coarse_part.append(np.full(_coarse_gnum.size, tgt_part_id, dtype=np.int32))

      # Bring back coarse tgt part on curent level distzone, using MGCellInfo value
      coarse_parent = PT.get_np_value(PT.find_node_from_path(dt_cur_lvl, zone_path + "/MultiGridCellInfo/CoarseIdx"))
      wanted_cell_zone_i = EP.part_to_part(i_coarse_part, coarse_gnum, [coarse_parent], comm)[0]
      target_part.append(wanted_cell_zone_i)
      zone_to_parts[zone_path] = [1.]*len(coarse_p_zones)

    # Split curent level
    pt = maia.factory.partition_dist_tree(dt_cur_lvl, comm, zone_to_parts=zone_to_parts, target_part=target_part, **kwargs)
    
    # Split done, update MGInfo to make it local
    for zone_path in PT.predicates_to_paths(dt_cur_lvl, 'CGNSBase_t/Zone_t'):
      cur_d_zone  = PT.find_node_from_path(dt_cur_lvl, zone_path)
      cur_p_zones = tr_utils.get_partitioned_zones(pt, zone_path)
      # First we need to transfer some field, if not already done by the user
      include_dict = {}
      if data_transfer not in ['ALL', 'FIELDS'] and 'DiscreteData_t' not in data_transfer:
        include_dict['DiscreteData_t'] = ['MultiGridCellInfo/*']
      if data_transfer not in ['ALL', 'FIELDS'] and 'BCDataSet_t' not in data_transfer:
        include_dict['BCDataSet_t'] = ['*/MultiGridBCFaceInfo/DirichletData/*']
      maia.transfer.dist_zone_to_part_zones_only(cur_d_zone, cur_p_zones, comm, include_dict)
      # Now we can update data
      for cur_p_zone in cur_p_zones:
        p_zone_path = PT.utils.path_head(zone_path) + '/' + PT.get_name(cur_p_zone)
        coarse_p_zone = PT.find_node_from_path(mg_pt, update_path_level(p_zone_path, lvl+1))

        coarse_gnum = MT.globalnumbering_value(coarse_p_zone, "Cell")
        coarse_idx_n = PT.find_node_from_path(cur_p_zone, "MultiGridCellInfo/CoarseIdx")
        pcoarse_idx = (np_utils.search(coarse_gnum, PT.get_np_value(coarse_idx_n))).astype(np.int32, copy=False)
        PT.set_value(coarse_idx_n, pcoarse_idx)
      
        coarse_ngon = PT.Zone.NGonNode(coarse_p_zone)
        coarse_gnum_face = MT.globalnumbering_value(coarse_ngon, 'Element')
        for cur_p_bc in PT.get_nodes_from_label(cur_p_zone, "BC_t"):
          # Same for faces, using gnum from NGonNode
          coarse_p_bc = PT.find_node_from_path(coarse_p_zone, f"ZoneBC/{cur_p_bc[0]}")
          coarse_p_bc_pl = PT.get_np_value(PT.find_child_from_name(coarse_p_bc, 'PointList'))
          coarse_bc_gnum = coarse_gnum_face[coarse_p_bc_pl[0]-PT.Element.Range(coarse_ngon)[0]]

          coarse_idx_n = PT.find_node_from_path(cur_p_bc, "MultiGridBCFaceInfo/DirichletData/CoarseIdx")
          pcoarse_idx = (np_utils.search(coarse_bc_gnum, PT.get_np_value(coarse_idx_n))).astype(np.int32, copy=False)
          PT.set_value(coarse_idx_n, pcoarse_idx)

    # Complete MG part_tree with current part level
    for base in PT.get_all_CGNSBase_t(pt):
      PT.add_child(mg_pt, base)

  return mg_pt