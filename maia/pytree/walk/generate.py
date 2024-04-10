import sys

from maia.pytree.typing import *

from .generate_utils import generate_functions
from .predicate import match_name
from .predicate import match_label
from .predicate import match_value
from .predicate import match_name_label

from . import walkers_api as WAPI


def _update_module_attributes(new_functions):
  for name, func in new_functions.items():
    setattr(_module_object, name, func)

_module_object = sys.modules[__name__]


# Specialization of legacy functions

#Generation for Node(s)Walker(s) based funcs
_base_functions = [
    WAPI.requestNodeFromPredicate,
    WAPI.getNodeFromPredicate,
    WAPI.getNodesFromPredicate,
    WAPI.iterNodesFromPredicate,
    ]

for _base_function in _base_functions:
  #Todo : raise DeprecationWarning
  easypredicates = {
    'Name' : (match_name,  ('name',)),
    'Value': (match_value, ('value',)),
    'Label': (match_label, ('label',)),
    'Type' : (match_label, ('label',)),
    'NameAndType'  : (match_name_label,  ('name', 'label',)),
    'NameAndLabel' : (match_name_label,  ('name', 'label',)),
  }
  generated = generate_functions(_base_function, maxdepth=3, child=True, easypredicates=easypredicates)
  _update_module_attributes(generated)
for _base_function in [WAPI.getNodesFromPredicates, WAPI.iterNodesFromPredicates]:
  generated = generate_functions(_base_function, easypredicates={}, maxdepth=3, child=True)
  _update_module_attributes(generated)

