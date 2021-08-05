from functools import partial

from maia.sids.cgns_keywords import Label as CGL
import maia.sids.cgns_keywords as CGK

from .generate_utils import *
from .compare import CGNSNodeFromPredicateNotFoundError

from .remove_nodes import rmChildrenFromPredicate, keepChildrenFromPredicate, rmNodesFromPredicate
from .walkers_api  import requestNodeFromPredicate, getNodeFromPredicate, \
                          getNodesFromPredicate,  iterNodesFromPredicate, \
                          getNodesFromPredicates, iterNodesFromPredicates

# RM nodes
def create_rm_children(predicate, nargs):
  def _rm_children_from(root, *args):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return rmChildrenFromPredicate(root, partial(predicate, **pkwargs))
  return _rm_children_from

generate_rmkeep_functions(rmChildrenFromPredicate, create_rm_children, allfuncs,
  "Remove all direct child CGNS nodes")

def create_keep_children(predicate, nargs):
  def _keep_children_from(root, *args):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return keepChildrenFromPredicate(root, partial(predicate, **pkwargs))
  return _keep_children_from

generate_rmkeep_functions(keepChildrenFromPredicate, create_keep_children, allfuncs,
  "Keep all direct child CGNS nodes")


#JC GENERATION FOR NodeWalker

base_functions = [
    requestNodeFromPredicate,
    getNodeFromPredicate,
    getNodesFromPredicate,
    iterNodesFromPredicate,
    getNodesFromPredicates,
    iterNodesFromPredicates,
    ]

for base_function in base_functions:
  generate_functions(base_function)

#Generation for rm functions
generate_functions(rmNodesFromPredicate)




#Generation for cgns names
def create_get_child_name(predicate, nargs, args):
  def _get_child_name(root, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return getNodeFromPredicate(root, partial(predicate, **pkwargs), **kwargs)
  return _get_child_name

# for cgns_type in filter(lambda i : i not in ['Null', 'UserDefined'] and not i.startswith('max'), CGK.PointSetType.__members__):
#   create_functions_name(create_get_child_name, cgns_type)

for name in filter(lambda i : not i.startswith('__') and not i.endswith('__'), dir(CGK.Name)):
  create_functions_name(create_get_child_name, name)


#Generation for families

def create_get_from_family(label, family_label):
  def _get_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        return node
    raise ValueError(f"Unable to find {label} from family name : {family_name}")
  return _get_from_family


def create_get_all_from_family(label, family_label):
  def _get_all_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    nodes = []
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        nodes.append(node)
    return nodes
  return _get_all_from_family

def create_iter_all_from_family(label, family_label):
  def _iter_all_from_family(parent, family_name):
    if isinstance(family_name, str):
      family_name = [family_name]
    nodes = []
    for node in getNodesFromLabel(parent, label):
      family_name_node = requestNodeFromLabel(node, family_label)
      if family_name_node and I.getValue(family_name_node) in family_name:
        yield node
  return _iter_all_from_family


for family_label in ['Family_t', 'AdditionalFamily_t']:
  for label in ['Zone_t', 'BC_t', 'ZoneSubRegion_t', 'GridConnectivity_t', 'GridConnectivity1to1_t', 'OversetHoles_t']:
    name = "ToUpdate"  #TODO : update name
    funcname = f"get{label[:-2]}From{family_label[:-2]}"
    func = create_get_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Return a CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)

    funcname = f"getAll{label[:-2]}From{family_label[:-2]}"
    func = create_get_all_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Return a list of all CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)

    funcname = f"getAll{label[:-2]}From{family_label[:-2]}"
    func = create_iter_all_from_family(label, family_label)
    func.__name__ = funcname
    func.__doc__  = """Iterates on CGNS node from {0} with name {1}.""".format(family_label, name)
    setattr(module_object, funcname, func)
    setattr(module_object, PYU.camel_to_snake(funcname), func)



#Generation for labels  
# --------------------------------------------------------------------------
#
#   get_{label}, iter{label}
#   get_all_{label}, iter_all_{label}
#   get_{label}{depth}, iter{label}{depth}
#   get_all_{label}{depth}, iter_all_{label}{depth}
#
# --------------------------------------------------------------------------
def create_get_child(predicate, nargs, args): #Duplicated
  """
    Alias for getNodesFrom... generator
  """
  def _get_node_from(root, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    try:
      return getNodeFromPredicate(root, partial(predicate, **pkwargs), **kwargs)
    except CGNSNodeFromPredicateNotFoundError as e:
      print(f"For predicate : pkwargs = {pkwargs}", file=sys.stderr)
      raise e
  return _get_node_from

def create_get_all_children(predicate, nargs, args):
  def _get_all_children_from(root, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return getNodesFromPredicate(root, partial(predicate, **pkwargs), **kwargs)
  return _get_all_children_from

def create_iter_all_children(predicate, nargs, args):
  def _iter_all_children_from(root, **kwargs):
    pkwargs = dict([(narg, arg,) for narg, arg in zip(nargs, args)])
    return iterNodesFromPredicate(root, partial(predicate, **pkwargs), **kwargs)
  return _iter_all_children_from

label_with_specific_depth = ['CGNSBase_t',
  'BaseIterativeData_t', 'Zone_t', 'Family_t',
  'Elements_t',
  'FlowSolution_t',
  'GridCoordinates_t',
  'ZoneBC_t', 'BC_t',
  'ZoneGridConnectivity_t', 'GridConnectivity_t', 'GridConnectivity1to1_t', 'OversetHoles_t',
  'ZoneIterativeData_t',
  'ZoneSubRegion_t',
]
for label in filter(lambda i : i not in ['CGNSTree_t'], CGL.__members__):
  suffix = label[:-2]
  suffix = suffix.replace('CGNS', '')
  snake_name = PYU.camel_to_snake(suffix)

  # Generate get{Base, Zone, ..}
  func = create_get_child(match_label, ('label',), (label,))
  funcname = f"get{suffix}"
  func.__name__ = funcname
  func.__doc__  = """get the first CGNS node from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, search='bfs'))
  # Generate get_{base, get_zone, ...}
  func = create_get_child(match_label, ('label',), (label,))
  funcname = f"get_{snake_name}"
  func.__name__ = funcname
  func.__doc__  = """get the first CGNS node from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, search='bfs'))

  pargs = {'search':'bfs', 'explore':'shallow'} if label in label_with_specific_depth else {'search':'dfs'}
  # Generate getAll{Base, Zone, ...}
  pargs['caching'] = True
  func = create_get_all_children(match_label, ('label',), (label,))
  funcname = f"getAll{suffix}"
  func.__name__ = funcname
  func.__doc__  = """get all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))
  # Generate get_all_{base, zone, ...}
  func = create_get_all_children(match_label, ('label',), (label,))
  funcname = f"get_all_{snake_name}"
  func.__name__ = funcname
  func.__doc__  = """get all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))

  # Generate iterAll{Base, Zone, ...}
  pargs['caching'] = False
  func = create_iter_all_children(match_label, ('label',), (label,))
  funcname = f"getAll{suffix}"
  func.__name__ = funcname
  func.__doc__  = """Get all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))
  # Generate iter_all_{base, zone, ...}
  func = create_iter_all_children(match_label, ('label',), (label,))
  funcname = f"iter_all_{snake_name}"
  func.__name__ = funcname
  func.__doc__  = """Iterate on all CGNS nodes from CGNS label {0}.""".format(label)
  setattr(module_object, funcname, partial(func, **pargs))

  for depth in range(1,MAXDEPTH+1):
    suffix = f"{suffix}_" if suffix[-1] in [str(i) for i in range(1,MAXDEPTH+1)] else suffix
    snake_name = PYU.camel_to_snake(suffix)

    # Generate get{Base, Zone, ...}{depth}
    func = create_get_child(match_label, ('label',), (label,))
    funcname = f"get{suffix}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Get the first CGNS node from CGNS label {0} with depth={1}.""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='bfs', depth=depth))
    # Generate get_{base, zone, ...}{depth}
    func = create_get_child(match_label, ('label',), (label,))
    funcname = f"get_{snake_name}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Get the first CGNS node from CGNS label {0} with depth={1}.""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='bfs', depth=depth))

    # Generate getAll{Base, Zone, ...}{depth}
    func = create_get_all_children(match_label, ('label',), (label,))
    funcname = f"getAll{suffix}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Get all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth, caching=True))
    # Generate get_all_{base, zone, ...}{depth}
    func = create_get_all_children(match_label, ('label',), (label,))
    funcname = f"get_all_{snake_name}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Get all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth, caching=True))

    # Generate iterAll{Base, Zone, ...}{depth}
    func = create_iter_all_children(match_label, ('label',), (label,))
    funcname = f"iterAll{suffix}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Iterate on all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth, caching=False))
    # Generate get_all_{base, zone, ...}{depth}
    func = create_iter_all_children(match_label, ('label',), (label,))
    funcname = f"iter_all_{snake_name}{depth}"
    func.__name__ = funcname
    func.__doc__  = """Iterate on all CGNS nodes from CGNS label {0} with depth={1}""".format(label, depth)
    setattr(module_object, funcname, partial(func, search='dfs', depth=depth, caching=False))


