import numpy as np

from maia.typing    import *
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils     import np_utils, par_utils, s_numbering
from maia.transfer  import utils    as te_utils
import Pypdm.Pypdm as PDM

def collect_distributed_pl(dist_zone: CGNSDistTree, 
                           query_list: List[Any],
                           filter_loc: Optional[List[str]] = None) -> List[NDArray]:
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
      if not PT.pred.IS_SUBSET(node):
        continue # Skip nodes w/o PL (eg. full FlowSolution_t)
      if PT.Zone.Type(dist_zone) == 'Structured' and PT.get_child_from_name(node, 'PointRange') is not None:
        continue # Skip structured nodes with PointRange (only PL are collected)
      if filter_loc is not None and PT.Subset.GridLocation(node) not in filter_loc:
        continue # Skip nodes that does not match provided loc

      pl_raw = MT.Subset.distributed_pointlist(node)
      if (s:=pl_raw.shape[0]) > 1:
        assert PT.Zone.Type(dist_zone) == 'Structured'
        func = s_numbering.ij_to_index_from_loc if s == 2 else s_numbering.ijk_to_index_from_loc
        idx = func(*pl_raw, PT.Subset.GridLocation(node), PT.Zone.VertexSize(dist_zone))
        pl = idx.reshape((1,-1), order='F')
      else:
        pl = pl_raw
      point_lists.append(pl)

  return point_lists


def create_part_pointlists(dist_zone: CGNSDistTree, 
                           p_zone: CGNSPartTree,
                           p_groups: Dict[str, NDArray],
                           pl_pathes: List[str], 
                           locations: List[str]) -> None:
  i_pl = 0
  for pl_path in pl_pathes:
    for nodes in PT.iter_children_from_predicates(dist_zone, pl_path, ancestors=True):
      ancestors, node = nodes[:-1], nodes[-1]
      loc = PT.Subset.GridLocation(node) if PT.pred.IS_SUBSET(node) else PT.Container.GridLocation(node, dist_zone)
      if loc in locations:
        pl_n = PT.get_child_from_name(node, 'PointList')
        pr_n = PT.get_child_from_name(node, 'PointRange')
        #Exclude nodes with no pl
        if pl_n or (pr_n and PT.Zone.Type(p_zone) == 'Unstructured'):
          beg_pl = p_groups['npZSRGroupIdx'][i_pl]
          end_pl = p_groups['npZSRGroupIdx'][i_pl+1]
          if beg_pl != end_pl:
            ancestor:CGNSTree = p_zone
            for parent in ancestors:
              ancestor = PT.update_child(ancestor, PT.get_name(parent), PT.get_label(parent), PT.get_value(parent))
            p_node = PT.update_child(ancestor, PT.get_name(node), PT.get_label(node), PT.get_value(node))
            PT.update_child(p_node, 'GridLocation', 'GridLocation_t', value=PT.Subset.GridLocation(node))
            pl_raw = p_groups['npZSRGroup'][beg_pl:end_pl]
            if PT.Zone.Type(p_zone) == 'Structured':
              pl_value = s_numbering.index_to_ijk_from_loc(pl_raw, loc, PT.Zone.VertexSize(p_zone))
            else:
              pl_value = pl_raw.reshape((1,-1), order='F') #type:ignore[assignment] #(reuse same var)
            PT.update_child(p_node, 'PointList', 'IndexArray_t', pl_value)
            MT.new_GlobalNumbering({'Index': p_groups['npZSRGroupLNToGN'][beg_pl:end_pl]}, p_node)
            # A corner case specific to BCDataSet : we can have a partitioned BCDS/PointList even if BC/PointList
            # was empty. In this case, we must create here an PointList (empty) and GridLoc for the parent BC
            if PT.get_label(p_node) == 'BCDataSet_t' and PT.get_child_from_name(ancestor, 'PointList') is None:
              d_ancestor = PT.find_node_from_path(dist_zone, '/'.join([PT.get_name(n) for n in ancestors]))
              d_ancestor_loc = PT.Subset.GridLocation(d_ancestor)
              d_ancestor_pl = PT.get_np_value(PT.find_child_from_name(d_ancestor, 'PointList'))
              PT.new_IndexArray('PointList', np.empty((d_ancestor_pl.shape[0],0), np.int32, order='F'), parent=ancestor)
              PT.new_GridLocation(d_ancestor_loc, ancestor)
              MT.new_GlobalNumbering({'Index' : np.empty(0, pdm_gnum_dtype)}, parent=ancestor)

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
    distri_partial = MT.Zone.vtx_distribution(dist_zone)
    ln_to_gn_list = [MT.Zone.vtx_globalnumbering(p_zone) for p_zone in part_zones]
  elif entity == 'SFace':
    distri_partial = MT.Zone.face_distribution(dist_zone)
    ln_to_gn_list = [MT.Zone.face_globalnumbering(p_zone) for p_zone in part_zones]

  elif entity == 'Elements':
    elts = PT.get_children_from_label(dist_zone, 'Elements_t')
    # For 3DPoly zone w/o NFaceElements (ie with only NGON+PE), create a fake NFace
    # node to correctly compute 'all-elements' distri & numbering (see #144)
    if PT.Zone.has_ngon_elements(dist_zone) and not PT.Zone.has_nface_elements(dist_zone) and PT.Zone.CellDimension(dist_zone) == 3:
      assert len(elts) == 1, f'Unable to guess position of missing NFaceElements in zone {dist_zone}. Try to add it manually.'
      ng_size = PT.Element.Size(PT.Zone.NGonNode(dist_zone))
      elts.append(PT.new_Elements('NFaceElements', 'NFACE_n', erange=[ng_size+1, ng_size+PT.Zone.n_cell(dist_zone)]))
    distri_partial = te_utils.create_all_elt_distribution(elts, comm)
    ln_to_gn_list = [te_utils.create_all_elt_g_numbering(p_zone, elts) for p_zone in part_zones]
    # Get elt_to_entity indirection from PDM, which is needed if we have a NGON/NFace output
    elt_to_entity_list = []
    for p_zone in part_zones:
      if PT.get_child_from_label(p_zone, 'FakeElements_t') is not None:
        sorted_dist_elts = sorted(elts, key = lambda item : PT.Element.Range(item)[0])
        p_elts = [PT.get_node_from_name(p_zone, PT.get_name(elt)) for elt in sorted_dist_elts]
        sections_parent_gnum = [PT.get_node_from_path(elt, ':CGNS#LocalNumbering/Entity')[1] for elt in p_elts if elt]
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

