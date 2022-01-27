import Converter.Internal as I
from maia.sids import Internal_ext as IE
from maia.sids import pytree as PT
from maia.sids import sids

from maia.utils import py_utils
from maia.utils.parallel import utils as par_utils
from maia.transform.dist_tree import add_joins_ordinal as AJO

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
  """

  if master is None:
    master = nodes[0]
  node = I.createNode(output_name, I.getType(master), I.getValue(master))

  data_queries = additional_data_queries + ['PointList', 'PointListDonor',
                  [lambda n : I.getType(n) == 'DataArray_t' and I.getName(n) not in ['Ordinal', 'OrdinalOpp']]]
  for data_query in data_queries:
    #Use master node to understand queries and collect nodes
    for childs in IE.getNodesWithParentsByMatching(master, data_query):
      path =  '/'.join([I.getName(n) for n in childs])
      data_to_merge = [I.getNodeFromPath(node, path)[1] for node in nodes]
      if childs[-1][1].ndim == 1:
        data_merged = py_utils.concatenate_np_arrays(data_to_merge)[1]
      else:
        data_merged = py_utils.concatenate_point_list(data_to_merge)[1].reshape(1, -1, order='F')

      #Recreate structure (still using master infos) and add merged array
      parent = node
      for child in childs[:-1]:
        parent = I.createUniqueChild(parent, I.getName(child), I.getType(child), I.getValue(child))
      child = childs[-1]
      I.createUniqueChild(parent, I.getName(child), I.getType(child), data_merged)

  #Copy child nodes
  for child_query in ['GridLocation_t'] + additional_child_queries:
    for child in IE.getNodesByMatching(master, child_query):
      I._addChild(node, child)

  newsize = I.getNodeFromName(node, 'PointList')[1].size
  idx_distri = I.getValue(IE.getDistribution(master, 'Index'))
  distri = par_utils.gather_and_shift(newsize, comm).astype(idx_distri.dtype)
  IE.newDistribution({'Index' : distri[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]}, node)
  return node

def concatenate_jns(tree, comm):
  """
  Parse the GridConnectivity_t of a tree and concatenate the GCs related to a same zone:
  if we have two jns A and B from zone1 to zone2 and two jns C and D from zone2 to zone1,
  produce A' from zone1 to zone2 and B' from zone2 to zone1
  Periodic jns are merged if their Periodic node are the same
  """
  match_jns = lambda n: I.getType(n) == 'GridConnectivity_t' \
                        and I.getValue(I.getNodeFromType(n, 'GridConnectivityType_t')) == 'Abutting1to1'

  AJO.add_joins_ordinal(tree, comm)
  for base, zone in IE.getNodesWithParentsByMatching(tree, ['CGNSBase_t', 'Zone_t']):
    jns_to_merge = {'Vertex' : dict(), 'FaceCenter' : dict(), 'CellCenter' : dict()}
    perio_refs   = {'Vertex' : list(), 'FaceCenter' : list(), 'CellCenter' : list()}
    for zgc, jn in IE.getNodesWithParentsByMatching(zone, ['ZoneGridConnectivity_t', match_jns]):
      donor_path = IE.getZoneDonorPath(I.getName(base), jn)
      location = sids.GridLocation(jn)
      perio_node = I.getNodeFromType1(jn, 'GridConnectivityProperty_t')
      is_periodic = perio_node is not None
      key = min(I.getNodeFromName(jn, 'Ordinal')[1][0], I.getNodeFromName(jn, 'OrdinalOpp')[1][0])

      #Manage periodic -- merge only if periodic values are identical
      if is_periodic:
        found = False
        for i,ref in enumerate(perio_refs[location]):
          if PT.is_same_tree(perio_node, ref):
            donor_path += f'.P{i}'
            found = True
            break
        if not found:
          perio_refs[location].append(perio_node)
          donor_path += f'.P{len(perio_refs[location])-1}'
      #Manage intrazone -- prevent merge of two sides into one
      elif donor_path == I.getName(base) + '/' + I.getName(zone):
        id = 0 if I.getNodeFromName(jn, 'Ordinal')[1][0] < I.getNodeFromName(jn, 'OrdinalOpp')[1][0] else 1
        donor_path = donor_path + f'.I{id}'

      try:
        jns_to_merge[location][donor_path].append((key,jn))
      except KeyError:
        jns_to_merge[location][donor_path] = [(key,jn)]
      I._rmNode(zgc, jn)

    for location, ljns_to_merge in jns_to_merge.items():
      for donor_path, jns in ljns_to_merge.items():
        #We need to merge jn and opposite jn in same order so sort according to ordinal key
        sorted_jns = [elem[1] for elem in sorted(jns)]
        merged_name = I.getName(zone) + '.To.' + donor_path.split('/')[1]
        merged = concatenate_subset_nodes(sorted_jns, comm, output_name=merged_name,
            additional_child_queries=['GridConnectivityType_t', 'GridConnectivityProperty_t', 'Ordinal', 'OrdinalOpp'])
        I._addChild(zgc, merged)
    # Make name uniques if we have multiple GridLocation
    if any([sum(ljns_to_merge.values()) for jns_to_merge in jns_to_merge.values()]):
      loc_suffix = {'Vertex' : '_v', 'FaceCenter' : '_f', 'CellCenter' : '_c'}
      for jn in I.getNodesFromType1(zgc, 'GridConnectivity_t'):
        if len(I.getNodesFromName1(zgc, I.getName(jn))) > 1:
          I.setName(jn, I.getName(jn) + '_' + sids.GridLocation(jn)[0])
