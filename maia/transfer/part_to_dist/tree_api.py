import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.typing        import *
from maia.pytree.typing import Predicates

from maia.transfer import utils as tr_utils
from . import data_exchange
from maia.factory.dist_from_part import _recover_base_iterative_data, discover_nodes_from_matching

__all__ = ['part_zones_to_dist_zone_only',
           'part_zones_to_dist_zone_all',
           'part_tree_to_dist_tree_only_labels',
           'part_tree_to_dist_tree_all',
           'part_tree_to_dist_tree_copy']

#Managed labels and corresponding funcs
LABELS = ['FlowSolution_t', 'DiscreteData_t', 'ArbitraryGridMotion_t', 'ZoneSubRegion_t', 'BCDataSet_t']
FUNCS = [data_exchange.part_sol_to_dist_sol, 
         data_exchange.part_discdata_to_dist_discdata,
         data_exchange.part_gridmotion_to_dist_gridmotion,
         data_exchange.part_subregion_to_dist_subregion,
         data_exchange.part_dataset_to_dist_dataset]

def _part_zones_to_dist_zone(dist_zone: CGNSDistTree,
                             part_zones: List[CGNSPartTree],
                             comm: MPIComm,
                             filter_dict: Dict[str, Tuple[Literal['I', 'E'], List[CGNSPath]]]) -> None:
  """
  Low level API to transfert data fields from the partitioned zones to the distributed zone.
  filter_dict must a dict containing, for each label defined in LABELS, a tuple (flag, paths):
   -  flag can be either 'I' (include) or 'E' (exclude)
   -  paths must be a (possibly empty) list of paths. Pathes must match the format expected by
      data_exchange functions defined in FUNCS
  If paths == [], all data will be transfered if flag == 'E' (= exclude nothing), and not data
  will be transfered if flag == 'I' (=include nothing)
  
  Args:
    dist_zone: Distributed zone to receive data
    part_zones: List of partitioned zones to transfer from
    comm: MPI communicator
    filter_dict: Dictionary mapping labels to (flag, paths) tuples
  """
  for label, func in zip(LABELS, FUNCS):
    tag, paths = filter_dict[label]
    if tag == 'I' and paths != []:
      func(dist_zone, part_zones, comm, include=paths)
    elif tag == 'E':
      func(dist_zone, part_zones, comm, exclude=paths)

def part_zones_to_dist_zone_only(dist_zone: CGNSDistTree,
                                 part_zones: List[CGNSPartTree],
                                 comm: MPIComm,
                                 include_dict: Dict[str, List[CGNSPath]]) -> None:
  """ Transfer the data fields specified in include_dict from the partitioned zones
  to the corresponding distributed zone.
  
  Args:
    dist_zone: Distributed zone to receive data
    part_zones: List of partitioned zones to transfer from
    comm: MPI communicator
    include_dict: Dictionary mapping labels to paths to include
  """
  MT.check_cgns_dist_tree(dist_zone)
  for part_zone in part_zones:
    MT.check_cgns_part_tree(part_zone)
  filter_dict: Dict[str, Tuple[Literal['I', 'E'], List[CGNSPath]]]
  filter_dict = {label : ('I', include_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : includeall -> exclude nothing
  filter_dict.update({label : ('E', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _part_zones_to_dist_zone(dist_zone, part_zones, comm, filter_dict)

def part_zones_to_dist_zone_all(dist_zone: CGNSDistTree,
                                part_zones: List[CGNSPartTree],
                                comm: MPIComm,
                                exclude_dict: Dict[str, List[CGNSPath]] = {}) -> None:
  """ Transfer all the data fields, excepted those specified in exclude_dict,
  from the partitioned zone to the corresponding distributed zone.
  
  Args:
    dist_zone: Distributed zone to receive data
    part_zones: List of partitioned zones to transfer from
    comm: MPI communicator
    exclude_dict: Dictionary mapping labels to paths to exclude
  """
  MT.check_cgns_dist_tree(dist_zone)
  for part_zone in part_zones:
    MT.check_cgns_part_tree(part_zone)
  filter_dict: Dict[str, Tuple[Literal['I', 'E'], List[CGNSPath]]]
  filter_dict = {label : ('E', exclude_dict.get(label, [])) for label in LABELS}
  #Manage joker ['*'] : excludeall -> include nothing
  filter_dict.update({label : ('I', []) for label in LABELS if filter_dict[label][1] == ['*']})
  _part_zones_to_dist_zone(dist_zone, part_zones, comm, filter_dict)

def part_tree_to_dist_tree_only_labels(dist_tree: CGNSDistTree,
                                       part_tree: CGNSPartTree,
                                       labels: List[str],
                                       comm: MPIComm) -> None:
  """ Transfer only the data fields matching the provided labels from the partitioned tree
  to the corresponding distributed tree.
  
  Args:
    dist_tree: Distributed tree to receive data
    part_tree: Partitioned tree to transfer from
    labels: List of labels to transfer
    comm: MPI communicator
  """
  MT.check_cgns_dist_tree(dist_tree)
  MT.check_cgns_part_tree(part_tree)
  assert isinstance(labels, list)
  include_dict = {label : ['*'] for label in labels}
  for d_base, d_zone in PT.get_children_from_labels(dist_tree, ['CGNSBase_t', 'Zone_t'], ancestors=True):
    p_zones = tr_utils.get_partitioned_zones(part_tree, PT.get_name(d_base) + '/' + PT.get_name(d_zone))
    part_zones_to_dist_zone_only(CGNSDistTree(d_zone), p_zones, comm, include_dict)

def part_tree_to_dist_tree_all(dist_tree: CGNSDistTree,
                               part_tree: CGNSPartTree,
                               comm: MPIComm) -> None:
  """ Transfer all the data fields from the partitioned tree to the corresponding distributed tree.
  
  Args:
    dist_tree (CGNSDistTree): Distributed tree to receive data
    part_tree (CGNSPartTree): Partitioned tree to transfer from
    comm      (MPIComm)     : MPI communicator
  """
  _recover_base_iterative_data(dist_tree, part_tree, comm)
  part_tree_to_dist_tree_only_labels(dist_tree, part_tree, LABELS, comm)
 
#Possible improvement : dist_tree_to_part_tree only and all API with global paths

def part_tree_to_dist_tree_copy(dist_tree: CGNSDistTree,
                                part_tree: CGNSPartTree,
                                predicates: Predicates,
                                comm: MPIComm) -> None:
  """ Copy nodes matching the input predicates chain from part_tree to dist_tree

  Args:
    dist_tree (CGNSDistTree): Distributed tree
    part_tree (CGNSPartTree): Corresponding partitioned tree
    predicates (str or list): Predicates chain, starting from tree level
    comm (MPIComm)          : MPI communicator

  Example:
      .. literalinclude:: snippets/test_transfer.py
        :start-after: #part_tree_to_dist_tree_copy@start
        :end-before: #part_tree_to_dist_tree_copy@end
        :dedent: 2
  """
  MT.check_cgns_dist_tree(dist_tree)
  MT.check_cgns_part_tree(part_tree)
  assert isinstance(predicates, (list, str))
  single_pred = '/' not in predicates if isinstance(predicates, str) else len(predicates) == 1
  if single_pred:
    discover_nodes_from_matching(dist_tree, [part_tree], predicates, comm, child_list=['*'], get_value='leaf')
    return

  leads_to_gc = lambda p: PT.get_label(PT.find_node_from_path(dist_tree, PT.utils.path_head(p,4))) \
                          in ['GridConnectivity_t', 'GridConnectivity1to1_t']

  # Capture start of predicate, because last node may not exist on dist tree
  _ud_predicate = PT.utils.path_head(predicates) if isinstance(predicates, str) else predicates[:-1]
  part_roots: Sequence[CGNSTree]
  for path in PT.predicates_to_paths(dist_tree, _ud_predicate):
    names = path.split('/')
    if len(names) == 1: # Data directly attached to a Base (e.g. Family_t nodes)
      cut = 1
      dist_root = PT.find_node_from_path(dist_tree, names[0])
      part_root = PT.get_node_from_path(part_tree, names[0])
      part_roots = [] if part_root is None else [part_root]
    else: # Deeper data
      cut = 2
      dist_root_path = PT.utils.path_head(path, 2)
      dist_root = PT.find_node_from_path(dist_tree, dist_root_path)
      if PT.get_label(dist_root) == 'Zone_t': # Deal zone (names differ on partitioned tree)
        part_roots = tr_utils.get_partitioned_zones(part_tree, dist_root_path)
        if len(names) >= 4 and leads_to_gc(path): # Data is actually below a GC : must manage jn splitting
          cut = 4
          dist_root_path = PT.utils.path_head(path, 4)
          dist_root = PT.find_node_from_path(dist_tree, dist_root_path)
          part_root_new = list() # Update part roots to start with concened GC_t
          for part_root in part_roots:
            part_root_new.extend(PT.get_nodes_from_predicates(part_root, f'{names[2]}/{names[3]}.*'))
          part_roots = part_root_new
      else: # Deal others
        part_root = PT.get_node_from_path(part_tree, dist_root_path)
        part_roots = [] if part_root is None else [part_root]
    _child_predicate = PT.utils.path_tail(predicates, cut) if isinstance(predicates, str) else predicates[cut:]
    # Remove nodes if they exist on dist tree, to force update of values
    for dist_path in PT.predicates_to_paths(dist_root, _child_predicate):
      PT.rm_node_from_path(dist_root, dist_path)

    discover_nodes_from_matching(dist_root, part_roots, _child_predicate, comm, child_list=['*'], get_value='leaf')
