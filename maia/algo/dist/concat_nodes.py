from collections import defaultdict
from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import np_utils, par_utils
from maia.algo.dist import matching_jns_tools as MJT

import numpy as np


def find_suffix(perio:PT.PeriodicValues, perio_refs:List[PT.PeriodicValues], add_opp_perio=False,
                perio_to_one_side_path_jn={}, cur_path='', donor_path=''):
  found = False
  for i,perio_ref in enumerate(perio_refs):
    if all([np.allclose(a,b,1e-5,1e-16) for a,b in zip(perio, perio_ref)]):
      if donor_path=='':
        suffix = f'.P{i}'
      elif donor_path not in perio_to_one_side_path_jn[i]:
        suffix = f'.P{i}'
        perio_to_one_side_path_jn[i].append(cur_path)
      else:
        suffix = f'.P{i+1}'
      found = True
      break
  if not found:
    perio_refs.append(perio)
    suffix = f'.P{len(perio_refs)-1}'
    perio_to_one_side_path_jn[len(perio_refs)-1] = [cur_path]
    if add_opp_perio:
      perio_to_one_side_path_jn[len(perio_refs)] = []
      perio_refs.append(-perio)
  return suffix
  

def concatenate_subset_nodes(nodes: List[CGNSTree],
                             comm: MPIComm,
                             output_name: str = 'ConcatenatedNode',
                             additional_data_queries: List[str] = [],
                             additional_child_queries: List[str] = [],
                             master: Optional[CGNSTree] = None) -> CGNSTree:
  """
  Concatenate some subset nodes (ie nodes having a PointList) into a single one.
  Subset nodes to be merged shall describe the same entity (eg a BC that have been split in two parts)

  Name of the concatenated node can be specified using output_name argument
  Concatenated array are PointList, PointListDonor, any DataArray_t + additional queries requested with
  additional_data_queries argument.
  Also copy the GridLocation node + all the nodes found using additional_child_queries
  Query are understood starting from nodes[0], unless if an other node is given using master argument
  Note that datas are only concatenated : duplicated, if any, are not removed.
  """

  if master is None:
    master = nodes[0]
  node = PT.new_node(output_name, PT.get_label(master), PT.get_value(master))

  data_queries = additional_data_queries + ['PointList', 'DataArray_t']
  for data_query in data_queries:
    #Use master node to understand queries and collect nodes
    for childs in PT.iter_children_from_predicates(master, data_query, ancestors=True):
      path =  '/'.join([PT.get_name(n) for n in childs])
      data_to_merge = [PT.get_np_value(PT.find_node_from_path(node, path)) for node in nodes]
      _, data_merged = np_utils.concatenate_np_arrays(data_to_merge)

      #Recreate structure (still using master infos) and add merged array
      parent = node
      for child in childs[:-1]:
        parent = PT.update_child(parent, PT.get_name(child), PT.get_label(child), PT.get_value(child))
      child = childs[-1]
      PT.new_child(parent, PT.get_name(child), PT.get_label(child), data_merged)

  #Copy child nodes by using node intersection to find common nodes
  empty_subset_nodes = [PT.new_node() for n in nodes]
  for child_query in ['GridLocation_t'] + additional_child_queries:

    for orig_n, fake_n in zip(nodes, empty_subset_nodes):
      for childs in PT.iter_children_from_predicates(orig_n, child_query, ancestors=True):
        parent = fake_n
        for child in childs[:-1]:
          parent = PT.update_child(parent, PT.get_name(child), PT.get_label(child), PT.get_value(child))
        child = childs[-1]
        PT.new_child(parent, PT.get_name(child), PT.get_label(child), PT.get_value(child), children=PT.get_children(child))

    intersected_node = PT.intersection(*empty_subset_nodes)

    for childs in PT.iter_children_from_predicates(intersected_node, child_query, ancestors=True):
      parent = node
      for child in childs[:-1]:
        parent = PT.update_child(parent, PT.get_name(child), PT.get_label(child), PT.get_value(child))
      child = childs[-1]
      PT.new_child(parent, PT.get_name(child), PT.get_label(child), PT.get_value(child), children=PT.get_children(child))

  val = PT.find_child_from_name(node, 'PointList')[1]
  assert val is not None
  newsize = val.shape[1]
  distri = par_utils.dn_to_distribution(newsize, comm)
  MT.new_Distribution({'Index' : distri}, node)

  for bcds_n in PT.get_children_from_label(node, 'BCDataSet_t'):
    bcds_pl_n = PT.get_child_from_name(bcds_n, 'PointList')
    if bcds_pl_n is not None:
      bcds_pl = PT.get_np_value(bcds_pl_n)[0]
      bcds_distrib = par_utils.dn_to_distribution(bcds_pl.size, comm)
      MT.new_Distribution({'Index':bcds_distrib}, parent=bcds_n)

  return node


def concatenate_bc_nodes(bc_nodes: List[CGNSTree],
                         comm: MPIComm,
                         output_name: str = 'ConcatenatedNode',
                         additional_data_queries: List[str] = [],
                         additional_child_queries: List[str] = [],
                         master: Optional[CGNSTree] = None) -> CGNSTree:
  """
  API for concatenate_subset_nodes which concatenate BCDS/PointList and BCDS/BCData/DataArray by default.
  """
  bcds_point_list = "BCDataSet_t/PointList"
  bcd_data_array  = "BCDataSet_t/BCData_t/DataArray_t"
  bcds_grid_loc   = "BCDataSet_t/GridLocation_t"
  # handling of scalar/tables combination (sets scalars to constant tables)
  for bc in bc_nodes :
    for bcds in PT.iter_children_from_label(bc, 'BCDataSet_t'):
      subset = PT.Container.SubsetNode(bcds, bc)
      subset_size_l = MT.Subset.dn_elem(subset)
      subset_size_g = MT.Subset.n_elem(subset)
      for data_array in PT.iter_children_from_predicates(bcds, 'BCData_t/DataArray_t'):
        da_value = PT.get_np_value(data_array)
        if (1 < subset_size_g) and (da_value.size == 1):
          PT.set_value(data_array, np.full(subset_size_l, da_value[0], da_value.dtype))
  bc_n = concatenate_subset_nodes(bc_nodes, comm, output_name=output_name,
                                  additional_data_queries=additional_data_queries+[bcds_point_list, bcd_data_array],
                                  additional_child_queries=additional_child_queries+[bcds_grid_loc],
                                  master=master)
  return bc_n


def concatenate_jns(tree: CGNSDistTree, comm: MPIComm) -> None:
  """
  Parse the GridConnectivity_t of a tree and concatenate the GCs related to a same zone:
  if we have two jns A and B from zone1 to zone2 and two jns C and D from zone2 to zone1,
  produce A' from zone1 to zone2 and B' from zone2 to zone1
  Periodic jns are merged if their Periodic node are the same
  Manage no match joins too
  """
  loc_suffix = {'Vertex' : '@Vtx', 'FaceCenter' : '@Face', 'CellCenter' : '@Cell'}
  key_index = lambda d,k: list(d.keys()).index(k)

  MJT.find_joins_donor_name(tree, comm)
  
  
  match_perio_refs:List[PT.PeriodicValues]   = []
  nomatch_perio_refs:List[PT.PeriodicValues] = []
  
  perio_to_one_side_path_jn:Dict[int, List[str]] = {}
  
  for base, zone in PT.iter_children_from_predicates(tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    
    zone_path = '/'.join([PT.get_name(node) for node in [base, zone]])
    
    #Do a get here because tree is modified
    for zgc in PT.get_children_from_label(zone, 'ZoneGridConnectivity_t'):
    
      match_jns_to_merge:Dict[str, Dict]   = {key: defaultdict(list) for key in loc_suffix}
      nomatch_jns_to_merge:Dict[str, Dict] = {key: defaultdict(list) for key in loc_suffix}
      nomatch_jns_to_keep:Dict[str, Dict]  = {key: defaultdict(list) for key in loc_suffix}
      other_gcs:Dict[str, List]            = {key: list() for key in loc_suffix}
      
      for jn in PT.get_children_from_label(zgc, 'GridConnectivity_t'):
        donor_path = PT.GridConnectivity.ZoneDonorPath(jn, PT.get_name(base))
        location = PT.Subset.GridLocation(jn)
        if location.endswith('FaceCenter'):
          location = 'FaceCenter' # Map I,J,K FaceCenter to FaceCenter
        type = PT.GridConnectivity.Type(jn)
        is_perio_gc = PT.GridConnectivity.isperiodic(jn)
        cur_jn_path = '/'.join([zone_path]+[PT.get_name(node) for node in [zgc, jn]])
        if type=="Abutting1to1":
          opp_jn_path = MJT.get_jn_donor_path(tree, cur_jn_path)
          key = min(cur_jn_path, opp_jn_path)
        else:
          key = cur_jn_path
        
        intra_gc = donor_path == zone_path
        if (intra_gc) and (not is_perio_gc) and (type == "Abutting"):
          nomatch_jns_to_keep[location][donor_path].append((key,jn))
          continue # Skip this join
  
        #Manage periodic -- merge only if periodic values are identical
        if is_perio_gc:
          perio = PT.GridConnectivity.periodic_values(jn)
          if type=="Abutting1to1":
            suffix = find_suffix(perio, match_perio_refs, add_opp_perio=intra_gc,
                                 perio_to_one_side_path_jn=perio_to_one_side_path_jn,
                                 cur_path=cur_jn_path, donor_path=opp_jn_path)
          elif type=="Abutting":
            suffix = find_suffix(perio, nomatch_perio_refs)
        #Manage intrazone -- prevent merge of two sides into one
        elif intra_gc:
          id = 0 if cur_jn_path < opp_jn_path else 1
          suffix = f'.I{id}'
        else:
          suffix = ''
        donor_path = donor_path + suffix
  
        # Set opposite name here -- it will be transfered on merged node
        if suffix == '.I0':
          opp_suffix = '.I1'
        elif suffix == '.I1':
          opp_suffix = '.I0'
        elif (suffix.startswith(".P")):
          if type=="Abutting1to1":
            suff_int = int(suffix[2:])
            if suff_int%2 == 0:
              opp_suffix = f'.P{suff_int+1}'
            else:
              opp_suffix = f'.P{suff_int-1}'
          elif type=="Abutting":
            opp_suffix = find_suffix(-perio, nomatch_perio_refs)
        else:
          opp_suffix = suffix
  
        if type=="Abutting1to1":
          match_jns_to_merge[location][donor_path].append((key,jn))
          index = key_index(match_jns_to_merge[location], donor_path)
          if suffix == "" or (suffix.startswith(".P") and not intra_gc):
            opp_jn = PT.find_node_from_path(tree, opp_jn_path)
            PT.update_child(opp_jn, "GridConnectivityDonorName", value=f'mergedGCMatch{index}{loc_suffix[location]}{suffix}')
          else:
            cur_path = f"{zone_path}{opp_suffix}"
            try:
              index_opp = key_index(match_jns_to_merge[location], cur_path)
              index = min(index, index_opp)
            except ValueError:
              pass
            if suffix.startswith(".I") or (suffix.startswith(".P") and intra_gc):
                gc_d_n_node = PT.find_child_from_name(jn, "GridConnectivityDonorName")
                PT.set_value(gc_d_n_node, f'mergedGCMatch{index}{loc_suffix[location]}{opp_suffix}')
        elif type=="Abutting":
          nomatch_jns_to_merge[location][donor_path].append((key,jn))
        else:
          other_gcs[location].append((key,jn))
      
      PT.rm_children_from_label(zgc, 'GridConnectivity_t')
      
      for location, lother_gcs in other_gcs.items():
        gc_counter = 0
        for _, other_gc in lother_gcs:
          type = PT.GridConnectivity.Type(other_gc)
          PT.set_name(other_gc, f'{type}{gc_counter}{loc_suffix[location]}')
          gc_counter += 1
          PT.add_child(zgc, other_gc)
      
      for location, lnomatch_gcs in nomatch_jns_to_keep.items():
        gc_counter = 0
        for _, nomatch_gcs in lnomatch_gcs.items():
          for _, nomatch_gc in nomatch_gcs:
            PT.set_name(nomatch_gc, f'GCNoMatch{gc_counter}{loc_suffix[location]}')
            gc_counter += 1
            PT.add_child(zgc, nomatch_gc)
      
      for location, ljns_to_merge in list(match_jns_to_merge.items())+list(nomatch_jns_to_merge.items()):
        if len(ljns_to_merge)==0: continue
        first_jn = ljns_to_merge[list(ljns_to_merge.keys())[0]][0][1]
        type = PT.GridConnectivity.Type(first_jn)
        prefix = "mergedGCMatch" if type == 'Abutting1to1' else "mergedGCNoMatch"
        for donor_path, jns in ljns_to_merge.items():
          #We need to merge jn and opposite jn in same order so sort according to ordinal key
          sorted_jns = [elem[1] for elem in sorted(jns)]
          first_jn = sorted_jns[0]
          intra_gc = ".".join(donor_path.split(".")[:-1]) == zone_path
          if type == "Abutting1to1":
            index = key_index(match_jns_to_merge[location], donor_path)
            gc_d_n = PT.get_str_value(PT.find_child_from_name(first_jn, 'GridConnectivityDonorName'))
            has_opp_suffix_i = gc_d_n.endswith('.I0') or gc_d_n.endswith('.I1')
            has_opp_suffix_p = len(gc_d_n.split('.P')) > 1 and gc_d_n.split('.P')[-1].isdigit()
            if has_opp_suffix_i:
              opp_suffix = gc_d_n[-3:]
              if opp_suffix == '.I0': suffix = '.I1'
              elif opp_suffix == '.I1': suffix = '.I0'
              index_opp = key_index(match_jns_to_merge[location], f"{zone_path}{opp_suffix}")
              index = min(index, index_opp)
            elif has_opp_suffix_p and intra_gc:
              opp_suffix = gc_d_n[gc_d_n.rfind('.P'):]
              opp_suffix_index = int(opp_suffix[2:])
              opp_perio_node = match_perio_refs[opp_suffix_index]
              suffix = find_suffix(-opp_perio_node, match_perio_refs)
              try:
                index_opp = key_index(match_jns_to_merge[location], f"{zone_path}{opp_suffix}")
                index = min(index, index_opp)
              except ValueError:
                pass
            elif (not intra_gc) and PT.GridConnectivity.isperiodic(first_jn):
              perio = PT.GridConnectivity.periodic_values(first_jn)
              suffix = find_suffix(perio, match_perio_refs)
            else:
              suffix = ""
          else:
            index = key_index(nomatch_jns_to_merge[location], donor_path)
            if PT.GridConnectivity.isperiodic(first_jn):
              perio = PT.GridConnectivity.periodic_values(first_jn)
              suffix = find_suffix(perio, match_perio_refs)
            else:
              suffix = ""
          
          merged_name = f'{prefix}{index}{loc_suffix[location]}{suffix}'
          additional_child_queries = ['GridConnectivityType_t', 'GridConnectivityProperty_t', 'Descriptor_t']
          additional_data_queries  = []
          if type=="Abutting1to1":
            additional_data_queries = ['PointListDonor']
          merged = concatenate_subset_nodes(sorted_jns, comm, output_name=merged_name,
                                            additional_data_queries=additional_data_queries,
                                            additional_child_queries=additional_child_queries)
          PT.add_child(zgc, merged)


def concatenate_subsets_from_families(dist_tree: CGNSDistTree,
                                      comm: MPIComm,
                                      families: Union[Literal['*'], List[str]] = '*') -> None:
  """ For each family, gather the related BC nodes into a single BC.

  If the shorcut ``'*'`` is used for ``families`` argument,
  all the detected FamilyName values in the tree will be used.

  BCDataSet are concatenated as well, and common metadata (such as FamilyName, Descriptors, etc.)
  are preserved. Tree is modified inplace and initial BCs are removed after concatenation.

  Note:

    - This function add some nodes in resulting BCs to preserve pre-concatenate tree info.
      Do not delete them if, for any reason, you want to retrieve initial tree
      (see :func:`~maia.algo.dist.deconcatenate_subsets_from_families`)
    - If ``dist_tree`` has ZoneSubRegion nodes with BCRegionName related to a concatenated BC,
      the BCRegionName descriptor will be replaced by the associated PointList. 

  Warning:
    For each family-grouped BCs, BCDataSet nodes must have the same tree structure

  Args:
    dist_tree (CGNSDistTree)                 : Distributed unstructured tree, starting at Zone_t level or higher.
    comm      (MPIComm)                      : MPI communicator
    families  (list of str or '*', optional) : Family names. Default to ``"*"``. 

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #concat_from_fam@start
      :end-before:  #concat_from_fam@end
      :dedent: 2

  """
  MT.check_cgns_dist_tree(dist_tree)
  for dist_zone in PT.iter_all_Zone_t(dist_tree):

    assert PT.Zone.Type(dist_zone)=="Unstructured"
    
    zone_families = {PT.get_str_value(n) for n in PT.get_nodes_from_predicates(dist_zone, 'ZoneBC_t/BC_t/FamilyName_t')}

    # > If all families, we need to discover them first
    if families=='*':
        _families = sorted(zone_families)
    else:
        _families = [f for f in families if f in zone_families]
      
    # > Merge bc nodes from a same family
    zone_bc_n = PT.find_node_from_label(dist_zone, "ZoneBC_t")
    for family in _families:

      # > Predicates to find family BCs
      is_subset_container = PT.pred.label_is('ZoneBC_t')
      is_subset = PT.pred.label_is('BC_t') & PT.pred.belongs_to_family(family)

      # > Go through family BCs gathering informations
      bc_nodes = list() ; bc_names = list() ; bc_ordin = list() 
      for i_bc, bc_n in enumerate(PT.get_nodes_from_predicates(dist_zone, [is_subset_container, is_subset])):
        bc_pl  = PT.get_np_value(PT.Subset.getPatch(bc_n))[0]
        bcds_n = PT.new_BCDataSet(":maia#concatenate", parent=bc_n)
        PT.new_BCData('DirichletData',
                      fields={'OriginalBCId':np.full(bc_pl.size, i_bc, dtype=np.int32)},
                      parent=bcds_n)
        ord_n = PT.get_child_from_label(bc_n, 'Ordinal_t')

        # > Manage ZSR with BCRegionName
        bc_name = PT.get_name(bc_n)
        is_zsr_rel_to_bc = PT.pred.label_is('ZoneSubRegion_t') & PT.pred.has_child_of_name('BCRegionName') \
                         & PT.pred.NodePredicate(lambda n : PT.get_value(PT.find_child_from_name(n, 'BCRegionName'))==bc_name)
        for zsr_bc_n in PT.get_children_from_predicate(dist_zone, is_zsr_rel_to_bc):
          pl_n = PT.find_child_from_name(bc_n, 'PointList')
          PT.new_IndexArray(value=PT.get_value(pl_n), parent=zsr_bc_n)
          PT.new_GridLocation(PT.Subset.GridLocation(bc_n), zsr_bc_n)
          PT.rm_children_from_name(zsr_bc_n, 'BCRegionName')

        bc_nodes.append(bc_n)
        bc_names.append(bc_name)
        if ord_n is not None:
          bc_ordin.append(str(PT.get_np_value(ord_n)[0]))

        for path in PT.predicates_to_paths(bc_n, 'BCDataSet_t/BCData_t'):
          bcd_path = PT.utils.path_head(path, 2)
          bcd_n = PT.find_node_from_path(bc_n, bcd_path)
          if PT.get_child_from_name(bcd_n, 'OriginalBCId') is None:
            array = PT.get_np_value(PT.find_child_from_label(bcd_n, 'DataArray_t'))
            PT.new_DataArray('OriginalBCId', np.full(array.size, i_bc, dtype=np.int32), parent=bcd_n)

        PT.rm_child(zone_bc_n, bc_n)

      if len(bc_ordin)!=0:
        assert len(bc_ordin)==len(bc_nodes)

      bc_n = concatenate_bc_nodes(bc_nodes, comm, output_name=family,
                                  additional_child_queries=['FamilyName_t'])
      PT.new_Descriptor('BCNames', '\n'.join(bc_names), parent=bc_n)
      if len(bc_ordin)!=0:
        PT.new_Descriptor('BCOrdinal', '\n'.join(bc_ordin), parent=bc_n)
      PT.add_child(zone_bc_n, bc_n)

is_concat = PT.pred.has_child_of_name(':maia#concatenate')

def deconcatenate_subsets_from_families(dist_tree: CGNSDistTree,
                                        comm: MPIComm,
                                        families: Union[Literal['*'], List[str]] = '*') -> None:
  """ For each given family, deconcatenate the related BC gathered with
  the concatenation service.

  If the shorcut ``'*'`` is used for ``families`` argument,
  all the detected FamilyName values in the tree will be used.

  BCDataSet are deconcatenated as well, and metadata (such as FamilyName, Descriptors, etc.)
  is preserved on generated BCs. Tree is modified inplace.
  
  Warning:
    Each family from ``families`` argument must lead to unique BC, including the custom node
    added by :func:`~maia.algo.dist.concatenate_subsets_from_families`.

  Args:
    dist_tree (CGNSDistTree)                 : Distributed unstructured tree, starting at Zone_t level or higher.
    comm      (MPIComm)                      : MPI communicator
    families  (list of str or '*', optional) : Family names. Default to ``"*"``. 

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #deconcatenate_from_fam@start
      :end-before:  #deconcatenate_from_fam@end
      :dedent: 2

  """
  MT.check_cgns_dist_tree(dist_tree)
  for dist_zone in PT.iter_all_Zone_t(dist_tree):

    assert PT.Zone.Type(dist_zone)=="Unstructured"

    zone_families = {PT.get_str_value(n) for n in PT.get_nodes_from_predicates(dist_zone, 'ZoneBC_t/BC_t/FamilyName_t')}
    # > If all families, we need to discover them first
    if families=='*':
        _families = sorted(zone_families)
    else:
        _families = [f for f in families if f in zone_families]

    # > Merge bc nodes from a same family
    zone_bc_n = PT.find_child_from_label(dist_zone, "ZoneBC_t")
    for family in _families:

      # > Predicates to find family BCs
      bc_nodes = PT.get_nodes_from_predicates(dist_zone, ['ZoneBC_t', PT.pred.label_is('BC_t') & PT.pred.belongs_to_family(family)])
      if len(bc_nodes)>1:
        raise ValueError(f"Family {family} leads to multiple BCs.")
      concat_bc_n = bc_nodes[0]

      # > For now only BCs are managed
      if PT.get_label(concat_bc_n)!='BC_t':
        raise NotImplementedError(f"Deconcatenation only works for BC_t nodes for now (predicate leads to {PT.get_label(concat_bc_n)} node)")

      # > Get concatenated BC node informations
      concat_bc_type = PT.get_str_value(concat_bc_n)
      concat_bc_loc  = PT.Subset.GridLocation(concat_bc_n)
      concat_bc_fam_n = PT.get_child_from_label(concat_bc_n, 'FamilyName_t')

      concat_bc_pl_n = PT.find_child_from_name(concat_bc_n, 'PointList')
      concat_bc_pl   = PT.get_np_value(concat_bc_pl_n)[0]

      concat_bc_id_n = PT.find_node_from_path(concat_bc_n, ':maia#concatenate/DirichletData/OriginalBCId')
      concat_bc_id   = PT.get_value(concat_bc_id_n)
      PT.rm_children_from_name(concat_bc_n, ':maia#concatenate')

      orig_bc_names = PT.get_str_value(PT.find_node_from_name_and_label(concat_bc_n, 'BCNames', 'Descriptor_t')).split('\n')
      orig_bc_ordin_n = PT.get_node_from_name_and_label(concat_bc_n, 'BCOrdinal', 'Descriptor_t')
      if orig_bc_ordin_n is not None:
        orig_bc_ordin = np.array(PT.get_str_value(orig_bc_ordin_n).split('\n'), dtype=np.int32)

      PT.rm_child(zone_bc_n, concat_bc_n)

      for bc_id, bc_name in enumerate(orig_bc_names):

        bc_pl_ids = np.where(concat_bc_id==bc_id)[0]
        bc_pl = concat_bc_pl[bc_pl_ids]
        bc_distrib = par_utils.dn_to_distribution(bc_pl.size, comm)

        bc_n = PT.new_BC(bc_name, concat_bc_type,
                         point_list=bc_pl.reshape((1,-1), order='F'),
                         loc=concat_bc_loc,
                         parent=zone_bc_n)
        if concat_bc_fam_n is not None:
          PT.new_FamilyName(PT.get_str_value(concat_bc_fam_n), parent=bc_n)
        MT.new_Distribution({'Index':bc_distrib}, parent=bc_n)
        if orig_bc_ordin_n is not None:
          PT.new_node('Ordinal', 'Ordinal_t', orig_bc_ordin[bc_id], parent=bc_n)

        # > Deconcatenate related BCDataSet children
        for nodes in PT.iter_children_from_predicates(concat_bc_n, 'BCDataSet_t/BCData_t', ancestors=True):
          bcds_n = nodes[0]
          bcd_n  = nodes[1]

          bcds_type  = PT.get_str_value(bcds_n)
          bcds_loc_n = PT.get_child_from_label(bcds_n, 'GridLocation_t')
          bcds_loc   = PT.Container.GridLocation(bcds_n, bc_n) if bcds_loc_n is not None else None

          bcds_pl_n = PT.get_child_from_name(bcds_n, 'PointList')
          if bcds_pl_n is not None:
            bcds_pl = PT.get_np_value(bcds_pl_n)[0]
            bcd_bc_id_n = PT.find_child_from_name_and_label(bcd_n, 'OriginalBCId', 'DataArray_t')
            bcd_bc_id   = PT.get_np_value(bcd_bc_id_n)
            bcds_pl_ids = np.where(bcd_bc_id==bc_id)[0]
            bcds_pl = bcds_pl[bcds_pl_ids]
            bcds_distrib = par_utils.dn_to_distribution(bcds_pl.size, comm)

            bc_bcds_n = PT.new_BCDataSet(PT.get_name(bcds_n), type=bcds_type,
                                         point_list=bcds_pl.reshape((1,-1), order='F'),
                                         loc=bcds_loc, parent=bc_n)
            MT.new_Distribution({'Index':bcds_distrib}, parent=bc_bcds_n)
            fields = {PT.get_name(data_array_n):PT.get_np_value(data_array_n)[bcds_pl_ids]
              for data_array_n in PT.get_children_from_label(bcd_n, 'DataArray_t')}
            bc_bcd_n = PT.new_BCData(PT.get_name(bcd_n), fields=fields, parent=bc_bcds_n)
            PT.rm_children_from_name(bc_bcd_n, 'OriginalBCId')
          else:
            bcds_pl_ids = bc_pl_ids
            bc_bcds_n = PT.new_BCDataSet(PT.get_name(bcds_n), type=bcds_type,
                                         loc=bcds_loc, parent=bc_n)
            fields = {PT.get_name(data_array_n):PT.get_np_value(data_array_n)[bcds_pl_ids]
              for data_array_n in PT.get_children_from_label(bcd_n, 'DataArray_t')}
            bc_bcd_n = PT.new_BCData(PT.get_name(bcd_n), fields=fields, parent=bc_bcds_n)
            PT.rm_children_from_name(bc_bcd_n, 'OriginalBCId')
