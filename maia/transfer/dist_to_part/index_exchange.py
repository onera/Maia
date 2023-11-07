import numpy              as np
import Pypdm.Pypdm        as PDM

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils     import np_utils, par_utils, s_numbering
from maia.transfer  import utils    as te_utils

def collect_distributed_pl(dist_zone, query_list, filter_loc=None):
  """
  Search and collect all the pointList values found under the nodes
  matching one of the query of query_list
  If a 1d PR is used, it is converted to a contiguous
  pointlist using the distribution node.
  If filter_loc list is not None, select only the pointLists of given
  GridLocation.
  """
  point_lists = []
  for query in query_list:
    for node in PT.iter_children_from_predicates(dist_zone, query):
      if filter_loc is None or PT.Subset.GridLocation(node) in filter_loc:
        pl_n = PT.get_child_from_name(node, 'PointList')
        pr_n = PT.get_child_from_name(node, 'PointRange')
        if pl_n is not None:
          pl_raw = pl_n[1]
          if PT.Zone.Type(dist_zone) == 'Structured':
            loc = PT.Subset.GridLocation(node)
            idx = s_numbering.ijk_to_index_from_loc(*pl_raw, loc, PT.Zone.VertexSize(dist_zone))
            point_lists.append(idx.reshape((1,-1), order='F'))
          else:
            point_lists.append(pl_raw)
        elif pr_n is not None and PT.Zone.Type(dist_zone) == 'Unstructured':
          pr = PT.get_value(pr_n)
          distrib = PT.get_value(MT.getDistribution(node, 'Index'))
          point_lists.append(np_utils.single_dim_pr_to_pl(pr, distrib))
        # else:
          # point_lists.append(np.empty((1,0), dtype=np.int32, order='F'))
  return point_lists


def create_part_pointlists(dist_zone, p_zone, p_groups, pl_pathes, locations):
  i_pl = 0
  for pl_path in pl_pathes:
    for nodes in PT.iter_children_from_predicates(dist_zone, pl_path, ancestors=True):
      ancestors, node = nodes[:-1], nodes[-1]
      loc = PT.Subset.GridLocation(node)
      if loc in locations:
        pl_n = PT.get_child_from_name(node, 'PointList')
        pr_n = PT.get_child_from_name(node, 'PointRange')
        #Exclude nodes with no pl
        if pl_n or (pr_n and PT.Zone.Type(p_zone) == 'Unstructured'):
          beg_pl = p_groups['npZSRGroupIdx'][i_pl]
          end_pl = p_groups['npZSRGroupIdx'][i_pl+1]
          if beg_pl != end_pl:
            ancestor = p_zone
            for parent in ancestors:
              ancestor = PT.update_child(ancestor, PT.get_name(parent), PT.get_label(parent), PT.get_value(parent))
            p_node = PT.new_child(ancestor, PT.get_name(node), PT.get_label(node), PT.get_value(node))
            PT.new_GridLocation(PT.Subset.GridLocation(node), parent=p_node)
            pl_raw = p_groups['npZSRGroup'][beg_pl:end_pl]
            if PT.Zone.Type(p_zone) == 'Structured':
              pl_value = s_numbering.index_to_ijk_from_loc(pl_raw, loc, PT.Zone.VertexSize(dist_zone))
            else:
              pl_value = pl_raw.reshape((1,-1), order='F')
            PT.new_PointList('PointList', pl_value, parent=p_node)
            lntogn_ud = MT.newGlobalNumbering(parent=p_node)
            PT.new_DataArray('Index', p_groups['npZSRGroupLNToGN'][beg_pl:end_pl], parent=lntogn_ud)

          i_pl += 1

def dist_pl_to_part_pl(dist_zone, part_zones, type_paths, entity, comm):

  if entity == 'Elements':
    filter_loc = ['EdgeCenter', 'FaceCenter', 'CellCenter']
  elif entity == 'Vertex':
    filter_loc = ['Vertex']
  elif entity == 'SFace': #Only for structured meshes
    assert PT.Zone.Type(dist_zone) == 'Structured'
    filter_loc = ['IFaceCenter', 'JFaceCenter', 'KFaceCenter']
  else:
    raise ValueError("Unsupported location for PointList exchange")

  #Create distri and lngn
  if entity == 'Vertex':
    distri_partial = te_utils.get_cgns_distribution(dist_zone, 'Vertex')
    ln_to_gn_list = te_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
  elif entity == 'SFace':
    distri_partial = te_utils.get_cgns_distribution(dist_zone, 'Face')
    ln_to_gn_list = te_utils.collect_cgns_g_numbering(part_zones, 'Face')

  elif entity == 'Elements':
    elts = PT.get_children_from_label(dist_zone, 'Elements_t')
    distri_partial = te_utils.create_all_elt_distribution(elts, comm)
    ln_to_gn_list = [te_utils.create_all_elt_g_numbering(p_zone, elts) for p_zone in part_zones]
    # Use this if we have indirection from PDM
    #ln_to_gn_parent_list = [te_utils.create_all_elt_g_numbering(p_zone, elts, 'ParentElement2') for p_zone in part_zones]
    # Otherwise
    elt_to_entity_list = [] #Usefull if NGON output from elt input
    for p_zone in part_zones:
      if PT.get_child_from_label(p_zone, 'FakeElements_t') is not None:
        sorted_dist_elts = sorted(elts, key = lambda item : PT.Element.Range(item)[0])
        p_elts = [PT.get_node_from_name(p_zone, PT.get_name(elt)) for elt in sorted_dist_elts]
        sections_parent_gnum = [PT.get_node_from_path(elt, ':CGNS#LocalNumbering/ExplicitEntity')[1] for elt in p_elts if elt]
        elt_to_entity_list.append(np_utils.concatenate_np_arrays(sections_parent_gnum)[1])

  pdm_distri = par_utils.partial_to_full_distribution(distri_partial, comm)

  # Recreate query for collect_distributed_pl interface
  query_list = [type_path.split('/') for type_path in type_paths] 
  #Collect PL
  point_lists = collect_distributed_pl(dist_zone, query_list, filter_loc=filter_loc)
  d_pl_idx, d_pl = np_utils.concatenate_point_list(point_lists, pdm_gnum_dtype)

  #Exchange
  list_group_part = PDM.part_distgroup_to_partgroup(comm, pdm_distri, d_pl_idx.shape[0]-1, d_pl_idx, d_pl,
      len(ln_to_gn_list), [len(lngn) for lngn in ln_to_gn_list], ln_to_gn_list)

  # Post treatement if we have dist elt but part ngon : convert elt PL to ngon
  if entity == 'Elements' and len(elt_to_entity_list) > 0:
    for _group_part, _elt_to_entity in zip(list_group_part, elt_to_entity_list):
      id_in_section_num = _group_part['npZSRGroup']
      id_in_poly_num    = _elt_to_entity[id_in_section_num-1] + 1
      _group_part['npZSRGroup'] = id_in_poly_num
  for i_part, p_zone in enumerate(part_zones):
    create_part_pointlists(dist_zone, p_zone, list_group_part[i_part], type_paths, filter_loc)

