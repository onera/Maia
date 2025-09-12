import numpy as np

from   maia.pytree.node import check
import maia.pytree.cgns_keywords as CGK

from maia.pytree import pred

def _has_wildcard(s):
  return isinstance(s, str) and '*' in s

def name_matches(name):
  return pred.name_matches(name) if _has_wildcard(name) else pred.name_is(name)
def label_matches(label):
  return pred.label_matches(label) if _has_wildcard(label) else pred.label_is(label)
def value_is(val):
  return pred.value_is(val)

def auto_predicate(query):
  if isinstance(query, str):
    if check.is_valid_label(query):
      predicate = label_matches(query)
    else:
      predicate = name_matches(query)
  elif isinstance(query, CGK.Label):
    predicate = pred.label_is(query)
  elif callable(query):
    predicate = query
  elif isinstance(query, np.ndarray):
    predicate = value_is(query)
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