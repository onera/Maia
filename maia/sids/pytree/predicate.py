import fnmatch
from functools import partial
import numpy as np

import maia.sids.cgns_keywords as CGK
from .compare import is_valid_label

__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3

def match_name(n, name: str):
  return fnmatch.fnmatch(n[__NAME__], name)

def match_value(n, value):
  return np.array_equal(n[__VALUE__], value)

def match_label(n, label: str):
  return n[__LABEL__] == label

def match_name_value(n, name: str, value):
  return fnmatch.fnmatch(n[__NAME__], name) and np.array_equal(n[__VALUE__], value)

def match_name_label(n, name: str, label: str):
  return n[__LABEL__] == label and fnmatch.fnmatch(n[__NAME__], name)

def match_name_value_label(n, name: str, value, label: str):
  return n[__LABEL__] == label and fnmatch.fnmatch(n[__NAME__], name) and np.array_equal(n[__VALUE__], value)

def match_value_label(n, value, label: str):
  return n[__LABEL__] == label and np.array_equal(n[__VALUE__], value)


def auto_predicate(query):
  if isinstance(query, str):
    if is_valid_label(query):
      predicate = partial(match_label, label=query)
    else:
      predicate = partial(match_name, name=query)
  elif isinstance(query, CGK.Label):
    predicate = partial(match_label, label=query.name)
  elif callable(query):
    predicate = query
  elif isinstance(query, np.ndarray):
    predicate = partial(match_value, value=predicate)
  else:
    raise TypeError("predicate must be a string for name, a numpy for value, a CGNS Label or a callable python function.")
  return predicate

