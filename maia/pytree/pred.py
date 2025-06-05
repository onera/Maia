import fnmatch
import numpy as np

from maia.pytree.typing import *

import maia.pytree.cgns_keywords as CGK
from   maia.pytree      import node as N
from   maia.pytree      import walk as W
from   maia.pytree      import sids as S

class UnaryPredicate:
  def __init__(self, func):
    self.func = func

  def __call__(self, X:CGNSTree) -> bool:
    return self.func(X)
  
  def __and__(self, other:"UnaryPredicate") -> "UnaryPredicate":
    return UnaryPredicate(lambda X : self(X) and other(X))

  def __or__(self, other:"UnaryPredicate") -> "UnaryPredicate":
    return UnaryPredicate(lambda X : self(X) or other(X))
  
  def __invert__(self) -> "UnaryPredicate":
    return UnaryPredicate(lambda X : not self(X))

def name_is(name: str) -> UnaryPredicate:
  """ Name of the node is exactly equal to the provided name """
  return UnaryPredicate(lambda n : n[0] == name)
def name_in(name_l) -> UnaryPredicate:
  """ Name of the node belongs to the provided list """
  return UnaryPredicate(lambda n : n[0] in name_l)
def name_matches(name: str) -> UnaryPredicate:
  """ Name of the node matches the provided name, for which wildcard `*` is accepted """
  return UnaryPredicate(lambda n : fnmatch.fnmatch(n[0], name))

def __value_is(n:CGNSTree, value) -> bool:
  if n[1] is None:
    return value is None
  elif value is None: #value is None and node[1] is not None
    return False
  else:
    _value = N.access._convert_value(value)
    assert _value is not None
    return np.array_equal(n[1], _value)

def value_is(value) -> UnaryPredicate:
  """ Value of the node is equal to the provided value """
  return UnaryPredicate(lambda n : __value_is(n, value))

def label_is(label) -> UnaryPredicate:
  """ Label of the node is exactly equal to the provided label """
  _label = label.name if isinstance(label, CGK.Label) else label
  return UnaryPredicate(lambda n : n[3] == _label)

def label_matches(label) -> UnaryPredicate:
  """ Label of the node matches the provided label, for which wildcard `*` is accepted """
  if isinstance(label, CGK.Label):
    return UnaryPredicate(lambda n : n[3] == label.name)
  else:
    return UnaryPredicate(lambda n : fnmatch.fnmatch(n[3], label))

def label_in(label_l) -> UnaryPredicate:
  """ Label of the node belongs to the provided list """
  _label_l = [label.name if isinstance(label, CGK.Label) else label for label in label_l]
  return UnaryPredicate(lambda n : n[3] in _label_l)

def __belongs_to_family(n:CGNSTree, target_family:str, allow_additional=False):
  family_name_n = W.get_child_from_label(n, 'FamilyName_t')
  if family_name_n:
    fam_val = N.get_str_value(family_name_n)
    if fnmatch.fnmatch(fam_val, target_family):
      return True
  if allow_additional:
    for additional_family_n in W.iter_children_from_label(n, 'AdditionalFamilyName_t'):
      fam_val = N.get_str_value(additional_family_n)
      if fnmatch.fnmatch(fam_val, target_family):
        return True
  return False

def belongs_to_family(target_family:str, allow_additional=False) -> UnaryPredicate:
  """
  Return True if the node n has a FamilyName_t child whose value is target_family.
  If allow_additional is True, also return True if node n has a AdditionalFamilyName_t child
  whose value is target_family. Wildcard are accepted in target_family.
  """
  return  UnaryPredicate(lambda n : __belongs_to_family(n, target_family, allow_additional))

def is_bc_of_loc(grid_loc):
  predicate = lambda n: N.get_label(n)=='BC_t' and S.Subset.GridLocation(n)==grid_loc
  return UnaryPredicate(predicate)

def is_elmt_of_type(cgns_name):
  predicate = lambda n: N.get_label(n)=='Elements_t' and S.Element.CGNSName(n)==cgns_name
  return UnaryPredicate(predicate)
