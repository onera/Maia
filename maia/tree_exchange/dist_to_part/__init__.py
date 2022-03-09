import Converter.Internal as I
from maia.sids import pytree as PT

import maia.tree_exchange as TE
from . import data_exchange

#Managed labels and corresponding funcs
LABELS = ['FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t', 'BCDataSet_t']
FUNCS = [data_exchange.dist_sol_to_part_sol, 
         data_exchange.dist_discdata_to_part_discdata,
         data_exchange.dist_subregion_to_part_subregion,
         data_exchange.dist_dataset_to_part_dataset]

def _dist_zone_to_part_zones(dist_zone, part_zones, comm, filter_dict):
  """
  Low level API to transfert data fields from the distributed zone to the partitioned zones.
  filter_dict must a dict containing, for each label defined in LABELS, a tuple (flag, pathes):
   -  flag can be either 'I' (include) or 'E' (exclude)
   -  pathes must be a (possibly empty) list of pathes. Pathes must match the format expected by
      data_exchange functions defined in FUNCS
  If pathes == [], all data will be transfered if flag == 'E' (= exclude nothing), and not data
  will be transfered if flag == 'I' (=include nothing)
  """
  for label, func in zip(LABELS, FUNCS):
    tag, pathes = filter_dict[label]
    if tag == 'I' and pathes != []:
      func(dist_zone, part_zones, comm, include=pathes)
    elif tag == 'E':
      func(dist_zone, part_zones, comm, exclude=pathes)

def dist_zone_to_part_zones_only(dist_zone, part_zones, comm, include_dict):
  """
  High level API to transfert data fields from the distributed zone to the partitioned zones.
  Only the the data fields defined in include_dict will be transfered : include_dict is
  a dictionnary of kind label : [pathes/to/include]. Path must match the format expected by
  data exchange functions, but for convenience we provide the shortcut label : ['*'] to include
  all the fields related to this specific label
  """
  filter_dict = {label : ('I', include_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : includeall -> exclude nothing
  filter_dict.update({label : ('E', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _dist_zone_to_part_zones(dist_zone, part_zones, comm, filter_dict)

def dist_zone_to_part_zones_all(dist_zone, part_zones, comm, exclude_dict={}):
  """
  High level API to transfert data fields from the distributed zone to the partitioned zones.
  All the data fields will be transfered, except the one defined in exclude_dict which is
  a dictionnary of kind label : [pathes/to/exclude]. Path must match the format expected by
  data exchange functions, but for convenience we provide the shortcut label : ['*'] to exclude
  all the fields related to this specific label
  """
  filter_dict = {label : ('E', exclude_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : excludeall -> include nothing
  filter_dict.update({label : ('I', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _dist_zone_to_part_zones(dist_zone, part_zones, comm, filter_dict)

def dist_tree_to_part_tree_only_labels(dist_tree, part_tree, labels, comm):
  """
  High level API to transfert all the data fields of the specified labels from a distributed tree
  to the corresponding partitionned tree.
  See LABELS for admissible values of labels
  """
  include_dict = {label : ['*'] for label in labels}
  for d_base, d_zone in PT.get_children_from_labels(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    p_zones = TE.utils.get_partitioned_zones(part_tree, I.getName(d_base) + '/' + I.getName(d_zone))
    dist_zone_to_part_zones_only(d_zone, p_zones, comm, include_dict)

def dist_tree_to_part_tree_all(dist_tree, part_tree, comm):
  """
  High level API to transfert all the data fields from a distributed tree
  to the corresponding partitionned tree
  """
  dist_tree_to_part_tree_only_labels(dist_tree, part_tree, LABELS, comm)
 
#Possible improvement : dist_tree_to_part_tree only and all API with global pathes
