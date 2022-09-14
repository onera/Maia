import maia.pytree as PT

import maia.transfer as TE
from . import data_exchange

__all__ = ['part_zones_to_dist_zone_only',
           'part_zones_to_dist_zone_all',
           'part_tree_to_dist_tree_only_labels',
           'part_tree_to_dist_tree_all']

#Managed labels and corresponding funcs
LABELS = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCDataSet_t']
FUNCS = [data_exchange.part_sol_to_dist_sol, 
         data_exchange.part_discdata_to_dist_discdata,
         data_exchange.part_subregion_to_dist_subregion,
         data_exchange.part_dataset_to_dist_dataset]

def _part_zones_to_dist_zone(dist_zone, part_zones, comm, filter_dict):
  """
  Low level API to transfert data fields from the partitioned zones to the distributed zone.
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

def part_zones_to_dist_zone_only(dist_zone, part_zones, comm, include_dict):
  """ Transfer the data fields specified in include_dict from the partitioned zones
  to the corresponding distributed zone.
  """
  filter_dict = {label : ('I', include_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : includeall -> exclude nothing
  filter_dict.update({label : ('E', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _part_zones_to_dist_zone(dist_zone, part_zones, comm, filter_dict)

def part_zones_to_dist_zone_all(dist_zone, part_zones, comm, exclude_dict={}):
  """ Transfer all the data fields, excepted those specified in exclude_dict,
  from the partitioned zone to the corresponding distributed zone.
  """
  filter_dict = {label : ('E', exclude_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : excludeall -> include nothing
  filter_dict.update({label : ('I', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _part_zones_to_dist_zone(dist_zone, part_zones, comm, filter_dict)

def part_tree_to_dist_tree_only_labels(dist_tree, part_tree, labels, comm):
  """ Transfer all the data fields of the specified labels from a partitioned tree
  to the corresponding distributed tree.
  """
  assert isinstance(labels, list)
  include_dict = {label : ['*'] for label in labels}
  for d_base, d_zone in PT.get_children_from_labels(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    p_zones = TE.utils.get_partitioned_zones(part_tree, PT.get_name(d_base) + '/' + PT.get_name(d_zone))
    part_zones_to_dist_zone_only(d_zone, p_zones, comm, include_dict)

def part_tree_to_dist_tree_all(dist_tree, part_tree, comm):
  """ Transfer all the data fields from a partitioned tree
  to the corresponding distributed tree.
  """
  part_tree_to_dist_tree_only_labels(dist_tree, part_tree, LABELS, comm)
 
#Possible improvement : dist_tree_to_part_tree only and all API with global paths
