import fnmatch
from functools import partial
import numpy as np

from maia.pytree.typing import *

import maia.pytree.cgns_keywords as CGK
from   maia.pytree      import node as N
from   maia.pytree      import sids as S
from   maia.pytree.node import check

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
  predicate = lambda n: N.get_label(n)=='Elements_t' and S.Element.CGNSName(n)==cgns_name
  return predicate

def auto_predicate(query):
  if isinstance(query, str):
    if check.is_valid_label(query):
      predicate = partial(match_str_label, label=query)
    else:
      predicate = partial(match_name, name=query)
  elif isinstance(query, CGK.Label):
    predicate = partial(match_cgk_label, label=query)
  elif callable(query):
    predicate = query
  elif isinstance(query, np.ndarray):
    predicate = partial(match_value, value=query)
  else:
    raise TypeError("predicate must be a string for name, a numpy for value, a CGNS Label or a callable python function.")
  return predicate

def auto_predicates(predicates):
  """
  Convert a list a "convenience" predicates to a list a true callable predicates
  The list can also be given as a '/' separated string
  """
  _predicates = []
  if isinstance(predicates, str):
    _predicates = [auto_predicate(p) for p in predicates.split('/')]
  elif isinstance(predicates, (list, tuple)):
    _predicates = []
    for p in predicates:
      if isinstance(p, dict):
        #Create a new dict with a callable predicate
        _predicates.append({**p, 'predicate' : auto_predicate(p['predicate'])})
      else:
        _predicates.append(auto_predicate(p))
  else:
    raise TypeError("predicates must be a sequence or a path as with strings separated by '/'.")
  return _predicates

