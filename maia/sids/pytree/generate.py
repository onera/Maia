import sys
from   functools import partial

from   maia.sids.cgns_keywords import Label as CGL
import maia.sids.cgns_keywords              as CGK

import maia.utils.py_utils as PYU

from .generate_utils import generate_functions
from .predicate      import match_name, match_label

from .remove_nodes import rmChildrenFromPredicate, keepChildrenFromPredicate, rmNodesFromPredicate
from .walkers_api  import requestNodeFromPredicate, getNodeFromPredicate, \
                          getNodesFromPredicate,  iterNodesFromPredicate, \
                          getNodesFromPredicates, iterNodesFromPredicates

def _update_module_attributes(new_functions, enable_snake=True):
  for name, func in new_functions.items():
    setattr(module_object, name, func)
    #Todo : We could update adapt camel_to_snake to avoid conversion of CGNSLabel
    if enable_snake:
      setattr(module_object, PYU.camel_to_snake(name), func)

module_object = sys.modules[__name__]

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
  generated = generate_functions(base_function)
  _update_module_attributes(generated)


#Generation for remove functions
generated = generate_functions(rmNodesFromPredicate)
_update_module_attributes(generated)
generated = generate_functions(rmChildrenFromPredicate,   maxdepth=0)
_update_module_attributes(generated)
generated = generate_functions(keepChildrenFromPredicate, maxdepth=0)
_update_module_attributes(generated)


#Generation for CGNSName -- eg getNodesFromNameCoordinateX. Only getNodes without level version
easypredicates = dict()
for cgnsname in dir(CGK.Name):
  if not cgnsname.startswith('__') and not cgnsname.endswith('__'):
    easypredicates['Name' + cgnsname] = (partial(match_name, name=cgnsname), tuple())
generated = generate_functions(getNodesFromPredicate, easypredicates=easypredicates, maxdepth=0)
#Names of functions could be updated here
_update_module_attributes(generated)

#Generation for CGNSLabel -- eg getNodesFromLabelZoneSubRegion. Only getNodes without level version
easypredicates = dict()
for cgnslabel in CGL.__members__:
  easypredicates['Label' + cgnslabel[:-2]] = (partial(match_label, label=cgnslabel), tuple())
generated = generate_functions(getNodesFromPredicate, easypredicates=easypredicates, maxdepth=0)
#Names of functions could be updated here
_update_module_attributes(generated)

