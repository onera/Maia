import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.utils import np_utils, par_utils
from maia.algo.dist import matching_jns_tools as MJT

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
      if childs[-1][1].ndim == 1:
        data_merged = np_utils.concatenate_np_arrays(data_to_merge)[1]
      else:
        data_merged = np_utils.concatenate_point_list(data_to_merge)[1].reshape(1, -1, order='F')

      #Recreate structure (still using master infos) and add merged array
      parent = node
      for child in childs[:-1]:
        parent = PT.update_child(parent, PT.get_name(child), PT.get_label(child), PT.get_value(child))
      child = childs[-1]
      PT.new_child(parent, PT.get_name(child), PT.get_label(child), data_merged)

  #Copy child nodes
  for child_query in ['GridLocation_t'] + additional_child_queries:
    for child in PT.iter_children_from_predicates(master, child_query):
      PT.add_child(node, child)

  newsize = PT.get_node_from_name(node, 'PointList')[1].size
  idx_distri = PT.get_value(MT.getDistribution(master, 'Index'))
  distri = np_utils.safe_int_cast(par_utils.gather_and_shift(newsize, comm), idx_distri.dtype)
  MT.newDistribution({'Index' : distri[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]}, node)
  return node

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
      donor_path = PT.getZoneDonorPath(PT.get_name(base), jn)
      location = PT.Subset.GridLocation(jn)
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
