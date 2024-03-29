import maia.pytree      as PT
import maia.pytree.maia as MT


import maia.transfer as TE
from . import data_exchange

__all__ = ['dist_zone_to_part_zones_only',
           'dist_zone_to_part_zones_all',
           'dist_tree_to_part_tree_only_labels',
           'dist_tree_to_part_tree_all',
           'dist_tree_to_part_tree_node_copy']

#Managed labels and corresponding funcs
LABELS = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCDataSet_t']
FUNCS = [data_exchange.dist_sol_to_part_sol, 
         data_exchange.dist_discdata_to_part_discdata,
         data_exchange.dist_subregion_to_part_subregion,
         data_exchange.dist_dataset_to_part_dataset]

def _dist_zone_to_part_zones(dist_zone, part_zones, comm, filter_dict):
  """
  Low level API to transfert data fields from the distributed zone to the partitioned zones.
  filter_dict must a dict containing, for each label defined in LABELS, a tuple (flag, paths):
   -  flag can be either 'I' (include) or 'E' (exclude)
   -  paths must be a (possibly empty) list of paths. Pathes must match the format expected by
      data_exchange functions defined in FUNCS
  If paths == [], all data will be transfered if flag == 'E' (= exclude nothing), and not data
  will be transfered if flag == 'I' (=include nothing)
  """
  for label, func in zip(LABELS, FUNCS):
    tag, paths = filter_dict[label]
    if tag == 'I' and paths != []:
      func(dist_zone, part_zones, comm, include=paths)
    elif tag == 'E':
      func(dist_zone, part_zones, comm, exclude=paths)

def dist_zone_to_part_zones_only(dist_zone, part_zones, comm, include_dict):
  """ Transfer the data fields specified in include_dict from a distributed zone
  to the corresponding partitioned zones.

  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #dist_zone_to_part_zones_only@start
        :end-before: #dist_zone_to_part_zones_only@end
        :dedent: 2
  """
  filter_dict = {label : ('I', include_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : includeall -> exclude nothing
  filter_dict.update({label : ('E', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _dist_zone_to_part_zones(dist_zone, part_zones, comm, filter_dict)

def dist_zone_to_part_zones_all(dist_zone, part_zones, comm, exclude_dict={}):
  """ Transfer all the data fields, excepted those specified in exclude_dict,
  from a distributed zone to the corresponding partitioned zones.

  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #dist_zone_to_part_zones_all@start
        :end-before: #dist_zone_to_part_zones_all@end
        :dedent: 2
  """
  filter_dict = {label : ('E', exclude_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : excludeall -> include nothing
  filter_dict.update({label : ('I', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _dist_zone_to_part_zones(dist_zone, part_zones, comm, filter_dict)

def dist_tree_to_part_tree_only_labels(dist_tree, part_tree, labels, comm):
  """ Transfer all the data fields of the specified labels from a distributed tree
  to the corresponding partitioned tree.

  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #dist_tree_to_part_tree_only_labels@start
        :end-before: #dist_tree_to_part_tree_only_labels@end
        :dedent: 2
  """
  assert isinstance(labels, list)
  include_dict = {label : ['*'] for label in labels}
  for d_base, d_zone in PT.get_children_from_labels(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    p_zones = TE.utils.get_partitioned_zones(part_tree, PT.get_name(d_base) + '/' + PT.get_name(d_zone))
    dist_zone_to_part_zones_only(d_zone, p_zones, comm, include_dict)

def dist_tree_to_part_tree_all(dist_tree, part_tree, comm):
  """ Transfer all the data fields from a distributed tree
  to the corresponding partitioned tree.

  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #dist_tree_to_part_tree_all@start
        :end-before: #dist_tree_to_part_tree_all@end
        :dedent: 2
  """
  dist_tree_to_part_tree_only_labels(dist_tree, part_tree, LABELS, comm)
 
#Possible improvement : dist_tree_to_part_tree only and all API with global paths

def dist_tree_to_part_tree_node_copy(dist_tree, part_tree, ud_predicate):
  """ Transfer nodes from a predicate and distributed tree
  to the corresponding partitioned tree.
  """
  for path in PT.predicates_to_paths(dist_tree, ud_predicate):
    # If path include a Zone_t node, we must loop over corresponding partitioned zones
    # so we update the correponding name to include wildcard *
    names = path.split('/')
    if len(names) >= 2:
      if PT.get_label(PT.get_node_from_path(dist_tree, f'{names[0]}/{names[1]}')) == 'Zone_t':
        names[1] += '.P*.N*'
    
    # Now copy dist_node to partitioned tree
    dist_node = PT.get_node_from_path(dist_tree, path)
    for part_node in PT.get_children_from_names(part_tree, names[:-1]):
      PT.rm_children_from_name(part_node, names[-1])
      PT.add_child(part_node, PT.deep_copy(dist_node))
