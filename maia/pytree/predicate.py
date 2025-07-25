import warnings
import fnmatch
import numpy as np

from maia.pytree.typing import *

import maia.pytree.cgns_keywords as CGK
from   maia.pytree      import node as N
from   maia.pytree      import sids as S

DEP_MSG = "This whole file is deprecated and will be removed soon." \
          " Use the new PT.pred module instead: https://numerics.gitlab-pages.onera.net/mesh/maia/dev/maia_pytree/search.html#building-predicates"

warnings.warn(DEP_MSG, DeprecationWarning, stacklevel=2)


def match_name(n:CGNSTree, name: str) -> bool:
  return fnmatch.fnmatch(n[0], name)

def match_value(n:CGNSTree, value) -> bool:
  if n[1] is None:
    return value is None
  elif value is None: #value is None and node[1] is not None
    return False
  else:
    _value = N.access._convert_value(value)
    assert _value is not None
    return np.array_equal(n[1], _value)

def match_str_label(n:CGNSTree, label:str) -> bool:
  return fnmatch.fnmatch(n[3], label)

def match_cgk_label(n:CGNSTree, label) -> bool:
  return n[3] == label.name

def match_label(n:CGNSTree, label):
  return match_cgk_label(n, label) if isinstance(label, CGK.Label) else match_str_label(n, label)

def match_name_value(n:CGNSTree, name: str, value):
  return match_name(n, name) and match_value(n, value)

def match_name_label(n:CGNSTree, name: str, label:str):
  return match_name(n, name) and match_label(n, label)

def match_value_label(n:CGNSTree, value, label:str):
  return match_value(n, value) and match_label(n, label)

def match_name_value_label(n:CGNSTree, name: str, value:str, label):
  return match_name(n, name) and match_value(n, value) and match_label(n, label)

def belongs_to_family(n:CGNSTree, target_family:str, allow_additional=False):
  """
  Return True if the node n has a FamilyName_t child whose value is target_family.
  If allow_additional is True, also return True if node n has a AdditionalFamilyName_t child
  whose value is target_family. Wildcard are accepted in target_family.
  """
  from maia.pytree import get_node_from_predicate, iter_nodes_from_predicate
  family_name_n = get_node_from_predicate(n, 'FamilyName_t', depth=[1,1])
  if family_name_n:
    assert isinstance(fam_val:=N.get_value(family_name_n), str)
    if fnmatch.fnmatch(fam_val, target_family):
      return True
  if allow_additional:
    for additional_family_n in iter_nodes_from_predicate(n, 'AdditionalFamilyName_t', depth=[1,1]):
      assert isinstance(fam_val:=N.get_value(additional_family_n), str)
      if fnmatch.fnmatch(fam_val, target_family):
        return True
  return False

def is_bc_of_loc(grid_loc):
  predicate = lambda n: N.get_label(n)=='BC_t' and S.Subset.GridLocation(n)==grid_loc
  return predicate

def is_elmt_of_type(cgns_name):
  predicate = lambda n: N.get_label(n)=='Elements_t' and S.Element.Type(n)==cgns_name
  return predicate
