import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.transfer import protocols as EP

from maia.utils import s_numbering
from maia.algo.dist import agglomeration as AGL

from maia.typing import *



def convert_s_to_ngon(tree:CGNSDistTree, comm:MPIComm):
  """ Implements convert_s_to_ngon for multigrid trees """

  nb_lvl = AGL.n_level(tree)
  zone_path_to_vertex_size = {path: PT.Zone.VertexSize(PT.find_node_from_path(tree, path))
                              for path in PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')}

  maia.algo.dist.convert_s_to_ngon(tree, comm)

  for zone_path, n_vtx_s in zone_path_to_vertex_size.items():
    n_vtx_s_coarse = [s//2 + 1 for s in n_vtx_s]

    if AGL.path_to_level(zone_path) == nb_lvl:
      # Skip last level
      continue

    zone = PT.find_node_from_path(tree, zone_path)
    celldim = PT.Zone.CellDimension(zone)
    prefix = 'IJK'[:celldim]
    to_index_func = s_numbering.ij_to_index_from_loc if celldim == 2 else s_numbering.ijk_to_index_from_loc
    mg_cell_info = PT.find_child_from_name(zone, "MultiGridCellInfo")
    coarse_idx_ijk = [PT.get_np_value(PT.find_child_from_name(mg_cell_info, f"{p}CoarseIdx")) for p in prefix]
    u_c_idx = to_index_func(*coarse_idx_ijk, "CellCenter", n_vtx_s_coarse)
    PT.rm_children_from_name(mg_cell_info, "*CoarseIdx")
    PT.new_DataArray("CoarseIdx", u_c_idx, parent=mg_cell_info)

    for bc in PT.get_nodes_from_label(zone, "BC_t"):
      mg_face_info = PT.find_child_from_name(bc, "MultiGridBCFaceInfo")
      coarse_idx_ijk = [PT.get_np_value(PT.find_node_from_name(mg_face_info, f"{p}CoarseIdx")) for p in prefix]
      bc_loc = PT.get_str_value(PT.find_node_from_path(mg_face_info, "BCStructuredLocation"))
      u_c_idx = to_index_func(*coarse_idx_ijk, bc_loc, n_vtx_s_coarse)

      # Now BC and BCDS should be FaceCenter with same PointList
      assert np.array_equal(PT.get_np_value(PT.find_child_from_name(bc, 'PointList')),
                            PT.get_np_value(PT.find_child_from_name(mg_face_info, 'PointList')))
      # --> Erase everything
      ds = PT.new_BCData('DirichletData', {'CoarseIdx' : u_c_idx})
      PT.set_children(mg_face_info, [ds])

def merge_connected_zones(tree:CGNSDistTree, comm:MPIComm, **kwargs):
  """ Implements merge_connected_zones for multigrid trees """
                          
  nb_lvl = AGL.n_level(tree)
  # Use any level to get connected zones
  tree_lvl_0 = PT.new_node('CGNSTree', 'CGNSTree_t', children=[b for b in PT.get_children(tree) if PT.get_name(b).endswith('.LV0')])
  celldim = PT.Base.CellDimension(PT.find_child_from_label(tree_lvl_0, 'CGNSBase_t'))
  if celldim == 2:
    n_face_or_edge = lambda z: MT.Element.n_elt(MT.Zone.EdgeNode(z))
  else:
    n_face_or_edge = lambda z: PT.Zone.n_face(z)
  groups = PT.Tree.find_connected_zones(tree_lvl_0)

  for lvl in range(nb_lvl+1):
    for i, group in enumerate(groups):
      lvl_group = [AGL.update_path_level(path, lvl) for path in group]
      n_cell_group_cur = [PT.Zone.n_cell(PT.find_node_from_path(tree, p)) for p in lvl_group]
      n_face_group_cur = [n_face_or_edge(PT.find_node_from_path(tree, p)) for p in lvl_group]


      if lvl < nb_lvl:
        lvl_group_next    = [AGL.update_path_level(path, lvl+1) for path in group]
        n_cell_group_next = [PT.Zone.n_cell(PT.find_node_from_path(tree, p)) for p in lvl_group_next]
        n_face_group_next = [n_face_or_edge(PT.find_node_from_path(tree, p)) for p in lvl_group_next]
      
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
          loc_start, loc_end = AGL.local_bounds(cell_distrib, start, end)
          c_u_idx[loc_start:loc_end] += shift_value

          start = end
          shift_value += n_cell_group_next[j]

      # --> Update faces for *previous* level (since we need current and previous level merged)
      if lvl > 0:
        output_path_prev = AGL.update_path_level(output_path, lvl-1)
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
        

