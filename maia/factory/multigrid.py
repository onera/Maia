import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP
from maia.transfer import utils as tr_utils

from maia.utils import par_utils, np_utils
from maia.algo.dist import agglomeration as AGL

from maia.typing import *


def partition_dist_tree(dist_tree:CGNSDistTree, comm:MPIComm, **kwargs) -> CGNSPartTree:
  """ Implements partition_dist_tree for multigrid trees """

  nb_lvl = AGL.n_level(dist_tree)

  data_transfer = kwargs.get('data_transfer', [])

  # Split coarser level
  dt_lvl_coarse = AGL.single_level_tree(dist_tree, nb_lvl)
  mg_pt = maia.factory.partition_dist_tree(dt_lvl_coarse, comm, **kwargs)

  # For others levels, zone_to_part and target part are recomputed
  kwargs.pop('zone_to_parts', None)
  kwargs.pop('target_part', None)

  for lvl in reversed(range(nb_lvl)):
    dt_cur_lvl = AGL.single_level_tree(dist_tree, lvl)
    pt_coarse_lvl = AGL.single_level_tree(mg_pt, lvl+1)
    
    zone_to_parts = dict()
    target_part = list()
    for zone_path in PT.predicates_to_paths(dt_cur_lvl, 'CGNSBase_t/Zone_t'):
      # Get partitioned zones for coarser level
      coarse_p_zones = tr_utils.get_partitioned_zones(pt_coarse_lvl, AGL.update_path_level(zone_path, lvl+1))
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
        coarse_p_zone = PT.find_node_from_path(mg_pt, AGL.update_path_level(p_zone_path, lvl+1))

        coarse_gnum = MT.globalnumbering_value(coarse_p_zone, "Cell")
        coarse_idx_n = PT.find_node_from_path(cur_p_zone, "MultiGridCellInfo/CoarseIdx")
        pcoarse_idx = (np_utils.search(coarse_gnum, PT.get_np_value(coarse_idx_n))).astype(np.int32, copy=False)
        PT.update_node(coarse_idx_n, name='CoarseLocalIdx', value=pcoarse_idx)
      
        coarse_ngon = MT.Zone.EdgeNode(coarse_p_zone) if PT.Zone.CellDimension(coarse_p_zone) == 2 else PT.Zone.NGonNode(coarse_p_zone)
        coarse_gnum_face = MT.globalnumbering_value(coarse_ngon, 'Element')
        for cur_p_bc in PT.get_nodes_from_label(cur_p_zone, "BC_t"):
          # Same for faces, using gnum from NGonNode
          coarse_p_bc = PT.find_node_from_path(coarse_p_zone, f"ZoneBC/{cur_p_bc[0]}")
          coarse_p_bc_pl = PT.get_np_value(PT.find_child_from_name(coarse_p_bc, 'PointList'))
          coarse_bc_gnum = coarse_gnum_face[coarse_p_bc_pl[0]-PT.Element.Range(coarse_ngon)[0]]

          coarse_idx_n = PT.find_node_from_path(cur_p_bc, "MultiGridBCFaceInfo/DirichletData/CoarseIdx")
          pcoarse_idx = (np_utils.search(coarse_bc_gnum, PT.get_np_value(coarse_idx_n))).astype(np.int32, copy=False)
          PT.update_node(coarse_idx_n, name='CoarseLocalIdx', value=pcoarse_idx)

    # Complete MG part_tree with current part level
    for base in PT.get_all_CGNSBase_t(pt):
      PT.add_child(mg_pt, base)

  return mg_pt
