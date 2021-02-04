import numpy              as np
import Converter.Internal as I
import Pypdm.Pypdm        as PDM

import maia.sids.sids     as SIDS
from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
from maia.utils.parallel import utils as par_utils
from maia.utils import py_utils
from maia.distribution.distribution_function import uniform_distribution
from maia.partitioning.split_U.collect_pl import collect_distributed_pl

def collect_lntogn_from_path(part_zones, path):
  return [I.getNodeFromPath(part_zone, path)[1] for part_zone in part_zones]

def create_all_elems_distri(dist_elts, comm):
  #Create distri
  elt_sections_dn   = [SIDS.ElementSize(elt) for elt in dist_elts]
  elt_sections_idx = py_utils.nb_to_offset(elt_sections_dn, dtype=np.int32)
  #Todo : create shortcut for full distri
  distri_partial = uniform_distribution(elt_sections_idx[-1], comm).astype(pdm_gnum_dtype)
  return distri_partial

def create_all_elems_lntogn(p_zone, dist_elts):
  #Create lngn -- Assert that in distree ER are increasing
  elt_sections_dn   = [SIDS.ElementSize(elt) for elt in dist_elts]
  elt_sections_idx = py_utils.nb_to_offset(elt_sections_dn, dtype=np.int32)
  p_elts = [I.getNodeFromName(p_zone, I.getName(elt)) for elt in dist_elts]
  elt_sections_pn = [SIDS.ElementSize(elt) if elt else 0 for elt in p_elts]
  offset = 0
  np_elt_ln_to_gn = np.empty(sum(elt_sections_pn), dtype=pdm_gnum_dtype)
  for i_elt, p_elt in enumerate(p_elts):
    if p_elt:
      local_ln_gn = I.getNodeFromPath(p_elt, ':CGNS#GlobalNumbering/Element')[1]
      np_elt_ln_to_gn[offset:offset+elt_sections_pn[i_elt]] = local_ln_gn + elt_sections_idx[i_elt]
      offset += elt_sections_pn[i_elt]
  return np_elt_ln_to_gn

def create_part_pointlists(dist_zone, p_zone, p_groups, pl_pathes, locations):
  i_pl = 0
  zone_suffix = '.' + '.'.join(I.getName(p_zone).split('.')[-2:])
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
            #Recreate path -- Carefull stupid name convention
            ancestor = p_zone
            for parent in ancestors:
              parent_name = I.getName(parent)
              # if I.getType(parent) in ['BC_t', 'GridConnectivity_t']:
                # parent_name = parent_name + zone_suffix
              ancestor = I.createUniqueChild(ancestor, parent_name, I.getType(parent), I.getValue(parent))
            node_name = I.getName(node)
            # if I.getType(nodes[-1]) in ['BC_t', 'GridConnectivity_t']:
              # node_name = node_name + zone_suffix
            p_node = I.createChild(ancestor, node_name, I.getType(node), I.getValue(node))
            I.newGridLocation(SIDS.GridLocation(node), parent=p_node)
            I.newIndexArray('PointList', p_groups['npZSRGroup'][beg_pl:end_pl].reshape((1,-1), order='F'), parent=p_node)
            lntogn_ud = I.createUniqueChild(p_node, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
            I.newDataArray('Index', p_groups['npZSRGroupLNToGN'][beg_pl:end_pl], parent=lntogn_ud)

          i_pl += 1

def dist_pl_to_part_pl(dist_zone, part_zones, type_paths, entity, comm):

  assert entity in ['Vertex', 'Elements']
  filter_loc = ['FaceCenter', 'CellCenter'] if entity=='Elements' else ['Vertex']

  #Create distri and lngn
  if entity == 'Vertex':
    distri_path = ':CGNS#Distribution/Vertex'
    distri_partial = I.getNodeFromPath(dist_zone, distri_path)[1].astype(pdm_gnum_dtype)
    pdm_distri     = par_utils.partial_to_full_distribution(distri_partial, comm)

    ln_to_gn_list = collect_lntogn_from_path(part_zones, ':CGNS#GlobalNumbering/Vertex')
  elif entity == 'Elements':
    elts = I.getNodesFromType1(dist_zone, 'Elements_t')
    distri_partial = create_all_elems_distri(elts, comm)
    pdm_distri     = par_utils.partial_to_full_distribution(distri_partial, comm)
    ln_to_gn_list = [create_all_elems_lntogn(p_zone, elts) for p_zone in part_zones]

  #Collect PL
  point_lists = collect_distributed_pl(dist_zone, type_paths, filter_loc=filter_loc)
  d_pl_idx, d_pl = py_utils.concatenate_point_list(point_lists, pdm_gnum_dtype)

  #Exchange
  list_group_part = PDM.part_distgroup_to_partgroup(comm, pdm_distri, d_pl_idx.shape[0]-1, d_pl_idx, d_pl,
      len(ln_to_gn_list), [len(lngn) for lngn in ln_to_gn_list], ln_to_gn_list)

  for i_part, p_zone in enumerate(part_zones):
    create_part_pointlists(dist_zone, p_zone, list_group_part[i_part], type_paths, filter_loc)


def dist_pl_to_part_pl_exc(dist_zone, part_zones, pl_path, comm):
  dist_node = I.getNodeFromPath(dist_zone, pl_path)
  cgnslocation = SIDS.GridLocation(dist_node)
  location = {'Vertex':'Vertex', 'FaceCenter':'Face', 'CellCenter':'Cell'}[cgnslocation]

  if location in ['Cell', 'Vertex']:
    distri_path = ':CGNS#Distribution/' + location
  # elif location == 'Face':
    # distri_path = 'NGonElements/:CGNS#Distribution/Distribution'

  distri_partial = I.getNodeFromPath(dist_zone, distri_path)[1].astype(pdm_gnum_dtype)
  pdm_distri     = par_utils.partial_to_full_distribution(distri_partial, comm)

  d_group = I.getNodeFromName1(dist_node, 'PointList')[1][0,:].astype(pdm_gnum_dtype)
  d_group_idx = np.array([0, len(d_group)], dtype=np.int32)

  ln_to_gn_list = collect_lntogn_from_path(part_zones, ':CGNS#GlobalNumbering/' + location)

  list_group_part = PDM.part_distgroup_to_partgroup(comm, pdm_distri, d_group_idx.shape[0]-1, d_group_idx, d_group, 
      len(ln_to_gn_list), [len(lngn) for lngn in ln_to_gn_list], ln_to_gn_list)

  print(list_group_part)
  #Now put in tree
  for i_part, p_zone in enumerate(part_zones):
    #Only 1 group for now
    this_part_data = list_group_part[i_part]
    beg_pl = this_part_data['npZSRGroupIdx'][0]
    end_pl = this_part_data['npZSRGroupIdx'][1]
    if beg_pl != end_pl:
      suffix = '.' + '.'.join(I.getName(p_zone).split('.')[-2:])
      #name = '{0}.{1}.{2}'.format(I.getName(dist_bc), *I.getName(p_zone).split('.')[-2:])
      pl_data = this_part_data['npZSRGroup'][beg_pl:end_pl]
      pl_lngn = this_part_data['npZSRGroupLNToGN'][beg_pl:end_pl]
      parent_node = I.getNodeFromPath(p_zone, '/'.join(pl_path.split('/')[:-1]) + suffix)
      I.newBCDataSet(I.getName(dist_node), I.getValue(dist_node), cgnslocation, parent=parent_node)


