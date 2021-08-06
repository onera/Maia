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

