import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import np_utils, par_utils
from maia.algo.apply_function_to_nodes import zones_iterator
from maia.algo.dist import matching_jns_tools as MJT

import numpy as np

def concatenate_subset_nodes(nodes, comm, output_name='ConcatenatedNode',
    additional_data_queries=[], additional_child_queries=[], master=None):
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

  data_queries = additional_data_queries + ['PointList', 'PointListDonor', 'DataArray_t']
  for data_query in data_queries:
    #Use master node to understand queries and collect nodes
    for childs in PT.iter_children_from_predicates(master, data_query, ancestors=True):
      path =  '/'.join([PT.get_name(n) for n in childs])
      data_to_merge = [PT.get_node_from_path(node, path)[1] for node in nodes]
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

  newsize = PT.get_child_from_name(node, 'PointList')[1].shape[1]
  distri = par_utils.dn_to_distribution(newsize, comm)
  MT.newDistribution({'Index' : distri}, node)

  for bcds_n in PT.get_children_from_label(node, 'BCDataSet_t'):
    bcds_pl_n = PT.get_child_from_name(bcds_n, 'PointList')
    if bcds_pl_n is not None:
      bcds_pl = PT.get_value(bcds_pl_n)[0]
      bcds_distrib = par_utils.dn_to_distribution(bcds_pl.size, comm)
      PT.maia.newDistribution({'Index':bcds_distrib}, parent=bcds_n)

  return node


def concatenate_bc_nodes(bc_nodes, comm, output_name='ConcatenatedNode',
    additional_data_queries=[], additional_child_queries=[], master=None):
  """
  API for concatenate_subset_nodes which concatenate BCDS/PointList and BCDS/BCData/DataArray by default.
  """
  bcds_point_list = "BCDataSet_t/PointList"
  bcd_data_array  = "BCDataSet_t/BCData_t/DataArray_t"
  bcds_grid_loc   = "BCDataSet_t/GridLocation_t"
  bc_n = concatenate_subset_nodes(bc_nodes, comm, output_name=output_name,
                                  additional_data_queries=additional_data_queries+[bcds_point_list, bcd_data_array],
                                  additional_child_queries=additional_child_queries+[bcds_grid_loc])
  return bc_n


def concatenate_jns(tree, comm):
  """
  Parse the GridConnectivity_t of a tree and concatenate the GCs related to a same zone:
  if we have two jns A and B from zone1 to zone2 and two jns C and D from zone2 to zone1,
  produce A' from zone1 to zone2 and B' from zone2 to zone1
  Periodic jns are merged if their Periodic node are the same
  """
  match_jns = lambda n: PT.get_label(n) == 'GridConnectivity_t' and PT.GridConnectivity.is1to1(n)

  MJT.add_joins_donor_name(tree, comm)
  for base, zone in PT.iter_children_from_predicates(tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    jns_to_merge = {'Vertex' : dict(), 'FaceCenter' : dict(), 'CellCenter' : dict()}
    perio_refs   = {'Vertex' : list(), 'FaceCenter' : list(), 'CellCenter' : list()}
    #Do a get here because tree is modified
    for zgc, jn in PT.get_children_from_predicates(zone, ['ZoneGridConnectivity_t', match_jns], ancestors=True):
      donor_path = PT.GridConnectivity.ZoneDonorPath(jn, PT.get_name(base))
      location = PT.Subset.GridLocation(jn)
      if location.endswith('FaceCenter'):
        location = 'FaceCenter' # Map I,J,K FaceCenter to FaceCenter
      perio_node = PT.get_child_from_label(jn, 'GridConnectivityProperty_t')
      is_periodic = perio_node is not None
      cur_jn_path = '/'.join([PT.get_name(node) for node in [base, zone, zgc, jn]])
      opp_jn_path = MJT.get_jn_donor_path(tree, cur_jn_path)
      key = min(cur_jn_path, opp_jn_path)

      #Manage periodic -- merge only if periodic values are identical
      if is_periodic:
        found = False
        for i,ref in enumerate(perio_refs[location]):
          if PT.is_same_tree(perio_node, ref):
            suffix = f'.P{i}'
            found = True
            break
        if not found:
          perio_refs[location].append(perio_node)
          suffix = f'.P{len(perio_refs[location])-1}'
      #Manage intrazone -- prevent merge of two sides into one
      elif donor_path == PT.get_name(base) + '/' + PT.get_name(zone):
        id = 0 if cur_jn_path < opp_jn_path else 1
        suffix = f'.I{id}'
      else:
        suffix = ''
      donor_path = donor_path + suffix

      # Set opposite name here -- it will be transfered on merged node
      if suffix == '.I0':  opp_suffix = '.I1'
      elif suffix == '.I1':  opp_suffix = '.I0'
      else:  opp_suffix = suffix
      opp_name_node = PT.get_child_from_name(jn, "GridConnectivityDonorName")
      PT.set_value(opp_name_node, opp_jn_path.split('/')[1] + '.To.' + PT.get_name(zone) + opp_suffix)

      try:
        jns_to_merge[location][donor_path].append((key,jn))
      except KeyError:
        jns_to_merge[location][donor_path] = [(key,jn)]
      PT.rm_child(zgc, jn)

    for location, ljns_to_merge in jns_to_merge.items():
      for donor_path, jns in ljns_to_merge.items():
        #We need to merge jn and opposite jn in same order so sort according to ordinal key
        sorted_jns = [elem[1] for elem in sorted(jns)]
        merged_name = PT.get_name(zone) + '.To.' + donor_path.split('/')[1]
        merged = concatenate_subset_nodes(sorted_jns, comm, output_name=merged_name,
            additional_child_queries=['GridConnectivityType_t', 'GridConnectivityProperty_t', 'Descriptor_t'])
        PT.add_child(zgc, merged)
    # Make name uniques if we have multiple GridLocation
    if sum([len(ljns_to_merge) > 0 for ljns_to_merge in jns_to_merge.values()]) > 1:
      loc_suffix = {'Vertex' : '_v', 'FaceCenter' : '_f', 'CellCenter' : '_c'}
      for jn in PT.get_children_from_label(zgc, 'GridConnectivity_t'):
        if len(PT.get_children_from_name(zgc, PT.get_name(jn))) > 1:
          PT.set_name(jn, PT.get_name(jn) + '_' + PT.Subset.GridLocation(jn)[0])
          opp_name_node = PT.get_child_from_name(jn, "GridConnectivityDonorName")
          PT.set_value(opp_name_node, PT.get_value(opp_name_node) + '_' + PT.Subset.GridLocation(jn)[0])
  # If we have multiple periodic jns or intrazone periodics, we can not guarantee that GridConnectivityDonorName is
  # good so rebuild it
  perio_found = False
  for jn in PT.iter_children_from_predicates(tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', match_jns]):
    perio_found = PT.get_child_from_label(jn, 'GridConnectivityProperty_t') is not None
    if perio_found:
      break
  if perio_found:
    MJT.add_joins_donor_name(tree, comm, force=True)


def concatenate_subset_from_families(dist_tree, comm, families='*'):
  """
  Create unique BC for each family while concatenating associated BCDataSets.
  Initial BCs will be removed after concatenation.

  Note:
    This function add some nodes in concatenated BCs to preserve pre-concatenate tree info.
    Do not delete them if, for any reason, you want to retrieve initial tree.

  Args:
    dist_tree (CGNSTree)              : Distributed unstructured tree
    comm      (MPIComm)               : MPI communicator
    families  (str or list, optional) : Family names. Default to ``"*"``. 

  Example:
    .. literalinclude:: snippets/test_algo.py
      :language: python
      :start-after: #concat_from_fam@start
      :end-before:  #concat_from_fam@end
      :dedent: 2

  """
  for dist_zone in zones_iterator(dist_tree):

    assert PT.Zone.Type(dist_zone)=="Unstructured"

    # > If all families, we need to discover them first
    if families=='*':
      families = list()
      for n in PT.get_nodes_from_label(dist_zone, 'FamilyName_t'):
        if PT.get_value(n) not in families:
          families.append(PT.get_value(n))

    # > Merge bc nodes from a same family
    zone_bc_n = PT.get_node_from_label(dist_zone, "ZoneBC_t")
    for family in families:

      # > Predicates to find family BCs
      is_subset_container = lambda n: PT.get_label(n) in ['ZoneBC_t']
      is_subset = lambda n: PT.get_label(n) in ['BC_t'] and\
                            PT.predicate.belongs_to_family(n, family, True)

      # > Go through family BCs gathering informations
      bc_nodes = list() ; bc_names = list() ; bc_ordin = list() 
      for i_bc, bc_n in enumerate(PT.get_nodes_from_predicates(dist_zone, [is_subset_container, is_subset])):
        bc_pl  = PT.Subset.getPatch(bc_n)[1][0]
        bcds_n = PT.new_BCDataSet(":maia#concatenate", parent=bc_n)
        PT.new_BCData('DirichletData',
                      fields={'OriginalBCId':np.full(bc_pl.size, i_bc)},
                      parent=bcds_n)
        ord_n = PT.get_child_from_label(bc_n, 'Ordinal_t')

        bc_nodes.append(bc_n)
        bc_names.append(PT.get_name(bc_n))
        if ord_n is not None:
          bc_ordin.append(str(PT.get_value(ord_n)[0]))

        for path in PT.predicates_to_paths(bc_n, 'BCDataSet_t/BCData_t'):
          bcd_path = PT.utils.path_head(path, 2)
          bcd_n = PT.get_node_from_path(bc_n, bcd_path)
          if PT.get_child_from_name(bcd_n, 'OriginalBCId') is None:
            array = PT.get_child_from_label(bcd_n, 'DataArray_t')[1]
            PT.new_DataArray('OriginalBCId', np.full(array.size, i_bc), parent=bcd_n)

        PT.rm_child(zone_bc_n, bc_n)

      if len(bc_ordin)!=0:
        assert len(bc_ordin)==len(bc_nodes)

      bc_n = concatenate_bc_nodes(bc_nodes, comm, output_name=family,
                                  additional_child_queries=['FamilyName_t'])
      PT.new_Descriptor('BCNames', '\n'.join(bc_names), parent=bc_n)
      if len(bc_ordin)!=0:
        PT.new_Descriptor('BCOrdinal', '\n'.join(bc_ordin), parent=bc_n)
      PT.add_child(zone_bc_n, bc_n)