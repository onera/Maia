from functools import partial

from maia.sids.cgns_keywords import Label as CGL
import maia.sids.cgns_keywords as CGK

from .generate_utils import *
from .compare import CGNSNodeFromPredicateNotFoundError

from .remove_nodes import rmChildrenFromPredicate, keepChildrenFromPredicate, rmNodesFromPredicate
from .walkers_api  import requestNodeFromPredicate, getNodeFromPredicate, \
                          getNodesFromPredicate,  iterNodesFromPredicate, \
                          getNodesFromPredicates, iterNodesFromPredicates

#Generation for Node(s)Walker(s) based funcs
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

#Generation for remove functions
generate_functions(rmNodesFromPredicate)
generate_functions(rmChildrenFromPredicate,   maxdepth=0)
generate_functions(keepChildrenFromPredicate, maxdepth=0)

from .predicate import match_name, match_label

#Generation for CGNSName -- eg getNodesFromNameCoordinateX. Only getNodes without level version
easypredicates = dict()
for cgnsname in dir(CGK.Name):
  if not cgnsname.startswith('__') and not cgnsname.endswith('__'):
    easypredicates['Name' + cgnsname] = (partial(match_name, name=cgnsname), tuple())
generate_functions(getNodesFromPredicate, easypredicates=easypredicates, maxdepth=0)

#Generation for CGNSLabel -- eg getNodesFromLabelZoneSubRegion. Only getNodes without level version
easypredicates = dict()
for cgnslabel in CGL.__members__:
  easypredicates['Label' + cgnslabel[:-2]] = (partial(match_label, label=cgnslabel), tuple())
generate_functions(getNodesFromPredicate, easypredicates=easypredicates, maxdepth=0)


# Todo : review this part

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

