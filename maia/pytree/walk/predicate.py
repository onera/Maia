import fnmatch
from functools import partial
import numpy as np

import Converter.Internal as I

import maia.pytree.cgns_keywords as CGK
from   maia.pytree.compare import is_valid_label

__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3

def match_name(n, name: str):
  return fnmatch.fnmatch(n[__NAME__], name)

def match_value(n, value):
  return np.array_equal(n[__VALUE__], value)

def match_str_label(n, label):
  return n[__LABEL__] == label

def match_cgk_label(n, label):
  return n[__LABEL__] == label.name

def match_label(n, label):
  return match_cgk_label(n, label) if isinstance(label, CGK.Label) else match_str_label(n, label)

def match_name_value(n, name: str, value):
  return match_name(n, name) and match_value(n, value)

def match_name_label(n, name: str, label):
  return match_name(n, name) and match_label(n, label)

def match_value_label(n, value, label):
  return match_value(n, value) and match_label(n, label)

def match_name_value_label(n, name: str, value, label):
  return match_name(n, name) and match_value(n, value) and match_label(n, label)

def belongs_to_family(n, target_family, allow_additional=False):
  """
  Return True if the node n has a FamilyName_t child whose value is target_family.
  If allow_additional is True, also return True if node n has a AdditionalFamilyName_t child
  whose value is target_family
  """
  from .walkers_api import getNodeFromPredicate, iterNodesFromPredicate
  family_name_n = getNodeFromPredicate(n, lambda m: match_cgk_label(m, CGK.Label.FamilyName_t), depth=1)
  if family_name_n and I.getValue(family_name_n) == target_family:
    return True
  if allow_additional:
    for additional_family_n in iterNodesFromPredicate(n, lambda m: match_cgk_label(m, CGK.Label.AdditionalFamilyName_t), depth=1):
      if I.getValue(additional_family_n) == target_family:
        return True
  return False

def auto_predicate(query):
  if isinstance(query, str):
    if is_valid_label(query):
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

