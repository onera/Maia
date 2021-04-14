import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

from maia      import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.sids import sids     as SIDS
import maia.sids.Internal_ext as IE
from maia.utils          import py_utils
from maia.utils.parallel import utils    as par_utils
from maia.tree_exchange  import utils    as te_utils

def collect_distributed_pl(dist_zone, type_paths, filter_loc=None):
  """
  Search and collect all the pointList values found under the given
  types pathes.
  If a 1d PR is used, it is converted to a contiguous
  pointlist using the distribution node.
  If filter_loc list is not None, select only the pointLists of given
  GridLocation.
  """
  point_lists = []
  for type_path in type_paths:
    for node in py_utils.getNodesFromTypePath(dist_zone, type_path):
      if filter_loc is None or SIDS.GridLocation(node) in filter_loc:
        pl_n = I.getNodeFromName1(node, 'PointList')
        pr_n = I.getNodeFromName1(node, 'PointRange')
        if pl_n is not None:
          point_lists.append(pl_n[1])
        elif pr_n is not None and I.getValue(pr_n).shape[0] == 1:
          pr = I.getValue(pr_n)
          distrib = IE.getDistribution(node, 'Index')
          point_lists.append(py_utils.single_dim_pr_to_pl(pr, distrib))
        # else:
          # point_lists.append(np.empty((1,0), dtype=np.int32, order='F'))
  return point_lists


def create_part_pointlists(dist_zone, p_zone, p_groups, pl_pathes, locations):
  i_pl = 0
  for pl_path in pl_pathes:
    for nodes in py_utils.getNodesWithParentsFromTypePath(dist_zone, pl_path):
      ancestors, node = nodes[:-1], nodes[-1]
      if SIDS.GridLocation(node) in locations:
        pl_n = I.getNodeFromName1(node, 'PointList')
        pr_n = I.getNodeFromName1(node, 'PointRange')
        #Exclude nodes with no pl
        if pl_n or (pr_n and I.getValue(pr_n).shape[0] == 1):
          beg_pl = p_groups['npZSRGroupIdx'][i_pl]
          end_pl = p_groups['npZSRGroupIdx'][i_pl+1]
          if beg_pl != end_pl:
            ancestor = p_zone
            for parent in ancestors:
              ancestor = I.createUniqueChild(ancestor, I.getName(parent), I.getType(parent), I.getValue(parent))
            p_node = I.createChild(ancestor, I.getName(node), I.getType(node), I.getValue(node))
            I.newGridLocation(SIDS.GridLocation(node), parent=p_node)
            I.newIndexArray('PointList', p_groups['npZSRGroup'][beg_pl:end_pl].reshape((1,-1), order='F'), parent=p_node)
            lntogn_ud = IE.newGlobalNumbering(parent=p_node)
            I.newDataArray('Index', p_groups['npZSRGroupLNToGN'][beg_pl:end_pl], parent=lntogn_ud)

          i_pl += 1

def dist_pl_to_part_pl(dist_zone, part_zones, type_paths, entity, comm):

  assert entity in ['Vertex', 'Elements']
  filter_loc = ['FaceCenter', 'CellCenter'] if entity=='Elements' else ['Vertex']

  #Create distri and lngn
  if entity == 'Vertex':
    distri_partial = te_utils.get_cgns_distribution(dist_zone, ':CGNS#Distribution/Vertex')
    pdm_distri     = par_utils.partial_to_full_distribution(distri_partial, comm)

    ln_to_gn_list = te_utils.collect_cgns_g_numbering(part_zones, ':CGNS#GlobalNumbering/Vertex')
  elif entity == 'Elements':
    elts = I.getNodesFromType1(dist_zone, 'Elements_t')
    distri_partial = te_utils.create_all_elt_distribution(elts, comm)
    pdm_distri     = par_utils.partial_to_full_distribution(distri_partial, comm)
    ln_to_gn_list = [te_utils.create_all_elt_g_numbering(p_zone, elts) for p_zone in part_zones]

  #Collect PL
  point_lists = collect_distributed_pl(dist_zone, type_paths, filter_loc=filter_loc)
  d_pl_idx, d_pl = py_utils.concatenate_point_list(point_lists, pdm_gnum_dtype)

  #Exchange
  list_group_part = PDM.part_distgroup_to_partgroup(comm, pdm_distri, d_pl_idx.shape[0]-1, d_pl_idx, d_pl,
      len(ln_to_gn_list), [len(lngn) for lngn in ln_to_gn_list], ln_to_gn_list)

  for i_part, p_zone in enumerate(part_zones):
    create_part_pointlists(dist_zone, p_zone, list_group_part[i_part], type_paths, filter_loc)


