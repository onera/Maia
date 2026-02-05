import maia.pytree      as PT
import maia.pytree.maia as MT

from   maia.typing        import *
from   maia.pytree.typing import Predicates

from ..import utils as tr_utils
from  .import data_exchange

__all__ = ['dist_zone_to_part_zones_only',
           'dist_zone_to_part_zones_all',
           'dist_tree_to_part_tree_only_labels',
           'dist_tree_to_part_tree_all',
           'dist_tree_to_part_tree_copy']

#Managed labels and corresponding funcs
LABELS = ['FlowSolution_t', 'DiscreteData_t', 'ArbitraryGridMotion_t', 'ZoneSubRegion_t', 'BCDataSet_t']
FUNCS = [data_exchange.dist_sol_to_part_sol, 
         data_exchange.dist_discdata_to_part_discdata,
         data_exchange.dist_gridmotion_to_part_gridmotion,
         data_exchange.dist_subregion_to_part_subregion,
         data_exchange.dist_dataset_to_part_dataset]

def _dist_zone_to_part_zones(dist_zone: CGNSDistTree,
                             part_zones: List[CGNSPartTree],
                             comm: MPIComm,
                             filter_dict: Dict[str, Tuple[Literal['I', 'E'], List[CGNSPath]]]) -> None:
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

def dist_zone_to_part_zones_only(dist_zone: CGNSDistTree,
                                 part_zones: List[CGNSPartTree],
                                 comm: MPIComm,
                                 include_dict: Dict[str, List[CGNSPath]]) -> None:
  """ Transfer the data fields specified in include_dict from a distributed zone
  to the corresponding partitioned zones.

  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #dist_zone_to_part_zones_only@start
        :end-before: #dist_zone_to_part_zones_only@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_zone)
  for part_zone in part_zones:
    MT.check_cgns_part_tree(part_zone)
  filter_dict: Dict[str, Tuple[Literal['I', 'E'], List[CGNSPath]]]
  filter_dict = {label : ('I', include_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : includeall -> exclude nothing
  filter_dict.update({label : ('E', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _dist_zone_to_part_zones(dist_zone, part_zones, comm, filter_dict)

def dist_zone_to_part_zones_all(dist_zone: CGNSDistTree,
                                part_zones: List[CGNSPartTree],
                                comm: MPIComm,
                                exclude_dict: Dict[str, List[CGNSPath]] = {}) -> None:
  """ Transfer all the data fields, excepted those specified in exclude_dict,
  from a distributed zone to the corresponding partitioned zones.

  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #dist_zone_to_part_zones_all@start
        :end-before: #dist_zone_to_part_zones_all@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_zone)
  for part_zone in part_zones:
    MT.check_cgns_part_tree(part_zone)
  filter_dict: Dict[str, Tuple[Literal['I', 'E'], List[CGNSPath]]]
  filter_dict = {label : ('E', exclude_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : excludeall -> include nothing
  filter_dict.update({label : ('I', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _dist_zone_to_part_zones(dist_zone, part_zones, comm, filter_dict)

def dist_tree_to_part_tree_only_labels(dist_tree: CGNSDistTree,
                                       part_tree: CGNSPartTree,
                                       labels: List[str],
                                       comm: MPIComm) -> None:
  """ Transfer all the data fields of the specified labels from a distributed tree
  to the corresponding partitioned tree.

  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #dist_tree_to_part_tree_only_labels@start
        :end-before: #dist_tree_to_part_tree_only_labels@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  MT.check_cgns_part_tree(part_tree)
  assert isinstance(labels, list)
  include_dict = {label : ['*'] for label in labels}
  for d_base, d_zone in PT.get_children_from_labels(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    p_zones = MT.get_partitioned_zones(part_tree, PT.get_name(d_base) + '/' + PT.get_name(d_zone))
    dist_zone_to_part_zones_only(CGNSDistTree(d_zone), p_zones, comm, include_dict)

def dist_tree_to_part_tree_all(dist_tree: CGNSDistTree,
                               part_tree: CGNSPartTree,
                               comm: MPIComm) -> None:
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

def dist_tree_to_part_tree_copy(dist_tree: CGNSDistTree,
                                part_tree: CGNSPartTree,
                                predicates: Predicates,
                                comm: MPIComm) -> None:
  """ Copy nodes matching the input predicates chain from dist_tree to part_tree

  Args:
    dist_tree (CGNSDistTree): Distributed tree
    part_tree (CGNSPartTree): Corresponding partitioned tree
    predicates (str or list): Predicates chain, starting from tree level
    comm (MPIComm)          : MPI communicator
  
  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #dist_tree_to_part_tree_copy@start
        :end-before: #dist_tree_to_part_tree_copy@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  MT.check_cgns_part_tree(part_tree)
  for path in PT.predicates_to_paths(dist_tree, predicates):
    # If path include a Zone_t node, we must loop over corresponding partitioned zones
    # so we update the correponding name to include wildcard *
    names = path.split('/')
    if len(names) >= 2:
      if PT.get_label(PT.find_node_from_path(dist_tree, PT.utils.path_head(path, 2))) == 'Zone_t':
        names[1] += '.P*.N*'
    # Same for GC_t nodes
    if len(names) >= 4:
      if PT.get_label(PT.find_node_from_path(dist_tree, PT.utils.path_head(path, 4))) in ['GridConnectivity_t', 'GridConnectivity1to1_t']:
        names[3] += '.*'

    # Now copy dist_node to partitioned tree
    dist_node = PT.find_node_from_path(dist_tree, path)
    if len(names) > 1:
      for part_node in PT.get_children_from_names(part_tree, names[:-1]):
        PT.rm_children_from_name(part_node, names[-1])
        PT.add_child(part_node, PT.deep_copy(dist_node))
    else:
      PT.rm_children_from_name(part_tree, names[-1])
      PT.add_child(part_tree, PT.deep_copy(dist_node))
