import sys
from   functools import partial

from .generate_utils import generate_functions, camel_to_snake
from .predicate import match_name
from .predicate import match_str_label
from .predicate import match_label
from .predicate import match_value
from .predicate import match_name_label

from .             import walkers_api as WAPI
from .remove_nodes import rm_children_from_predicate
from .remove_nodes import keep_children_from_predicate
from .remove_nodes import rm_nodes_from_predicate

def _update_module_attributes(new_functions):
  for name, func in new_functions.items():
    setattr(module_object, name, func)

module_object = sys.modules[__name__]

# Specialization of XXX_from_predicate

base_functions = [
    WAPI.request_node_from_predicate,
    WAPI.get_node_from_predicate,
    WAPI.get_nodes_from_predicate,
    WAPI.iter_nodes_from_predicate,
    ]

for base_function in base_functions:
  generated = generate_functions(base_function, maxdepth=0, child=True)
  _update_module_attributes(generated)

# For predicates version, only do single argument easy predicate
for base_function in [WAPI.get_node_from_predicates, WAPI.iter_nodes_from_predicates, WAPI.get_nodes_from_predicates]:
  easypredicates = {
    'Name' : (match_name,  ('name',)),
    'Value': (match_value, ('value',)),
    'Label': (match_label, ('label',)),
    'NameAndLabel' : (match_name_label,  ('name', 'label',)),
  }
  generated = generate_functions(base_function, easypredicates=easypredicates, maxdepth=0, child=True)
  _update_module_attributes(generated)

# Specialization of XXX_from_predicate(s) for some CGNSLabel
base_functions = [partial(WAPI.get_nodes_from_predicate, explore='deep'),
                  partial(WAPI.iter_nodes_from_predicate, explore='deep')]
easypredicates = dict()
easylabels = ['CGNSBase_t', 'Zone_t', 'BC_t', 'Family_t']
for label in easylabels:
  easypredicates['Label'+ label] = (partial(match_str_label, label=label), tuple())

generated = {}
for base_function in base_functions:
  generated.update(generate_functions(base_function, easypredicates=easypredicates, maxdepth=0, child=False))
#Update name to avoid snake case and remove nodes_from
for label in easylabels:
  for prefix in ['get', 'iter']:
    old_key = f'{prefix}_nodes_from_label_' + camel_to_snake(label)
    func = generated.pop(old_key)
    func.__name__  = f'{prefix}_all_{label}'
    generated[func.__name__] = func
_update_module_attributes(generated)

# Specialization of remove functions
for rm_function in [rm_nodes_from_predicate, rm_children_from_predicate, keep_children_from_predicate]:
  generated = generate_functions(rm_function, maxdepth=0, child=False)
  _update_module_attributes(generated)

def get_node_from_path(root, path, ancestors=False):
  if path == '':
    return [root] if ancestors else root
  nodes = WAPI.get_nodes_from_predicates(root, path, depth=[1,1], ancestors=ancestors)
  if len(nodes) == 0 and ancestors:
    return []
  if len(nodes) == 1:
    return nodes[0]
  elif len(nodes) > 1:
    raise RuntimeError(f"Multiple nodes founds for path {path}")

def rm_node_from_path(root, path):
  from maia.pytree.path_utils import path_head, path_tail
  if not '/' in path:
    rm_children_from_name(root, path)
  else:
    parent = get_node_from_path(root, path_head(path))
    rm_nodes_from_name(parent, path_tail(path))

def get_all_subsets(root,filter_loc=None):
  """
  Search and collect all the subsets nodes found under root and the root
  itself if it is a subset
  If filter_loc list is not None, select only the subsets nodes of given
  GridLocation.
  """
  return list(iter_all_subsets(root,filter_loc))

def iter_all_subsets(root,filter_loc=None):
  """
  Search and iter on all the subsets nodes found under root
  If filter_loc list is not None, select only the subsets nodes of given
  GridLocation.
  """
  import maia.pytree as PT
  eligible_subset_paths = ['CGNSBase_t/Zone_t/ZoneBC_t/BC_t',
                           'CGNSBase_t/Zone_t/ZoneBC_t/BC_t/BCDataSet_t',
                           'CGNSBase_t/Zone_t/ZoneSubRegion_t',
                           'CGNSBase_t/Zone_t/DiscreteData_t',
                           'CGNSBase_t/Zone_t/FlowSolution_t',
                           'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity_t']

  root_label = PT.get_label(root)
  if root_label != 'CGNSTree_t':
    eligible_subset_paths = [path for path in eligible_subset_paths if root_label in path]

  subset_paths = []
  for path in eligible_subset_paths:
    path_split = path.split(root_label+'/')
    if len(path_split)>1:
      subset_paths.append(path_split[1])
    else:
      if filter_loc is None or PT.Subset.GridLocation(root) in filter_loc:
        pl_n = PT.get_child_from_name(root, 'PointList')
        pr_n = PT.get_child_from_name(root, 'PointRange')
        if (pl_n is not None) or (pr_n is not None):
          yield root

  for path in subset_paths:
    for subset_n in PT.iter_children_from_predicates(root, path):
      if filter_loc is None or PT.Subset.GridLocation(subset_n) in filter_loc:
        pl_n = PT.get_child_from_name(subset_n, 'PointList')
        pr_n = PT.get_child_from_name(subset_n, 'PointRange')
        if (pl_n is not None) or (pr_n is not None):
          yield subset_n


# Specialization of legacy functions

#Generation for Node(s)Walker(s) based funcs
base_functions = [
    WAPI.requestNodeFromPredicate,
    WAPI.getNodeFromPredicate,
    WAPI.getNodesFromPredicate,
    WAPI.iterNodesFromPredicate,
    ]

for base_function in base_functions:
  #Todo : raise DeprecationWarning
  easypredicates = {
    'Name' : (match_name,  ('name',)),
    'Value': (match_value, ('value',)),
    'Label': (match_label, ('label',)),
    'Type' : (match_label, ('label',)),
    'NameAndType'  : (match_name_label,  ('name', 'label',)),
    'NameAndLabel' : (match_name_label,  ('name', 'label',)),
  }
  generated = generate_functions(base_function, maxdepth=3, child=True, easypredicates=easypredicates)
  _update_module_attributes(generated)
for base_function in [WAPI.getNodesFromPredicates, WAPI.iterNodesFromPredicates]:
  generated = generate_functions(base_function, easypredicates={}, maxdepth=3, child=True)
  _update_module_attributes(generated)


