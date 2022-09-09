import numpy as np

import maia.pytree.node as N
import maia.pytree.cgns_keywords as CGK

def is_valid_name(name, check_len: bool=True) -> bool:
  """
  Return True if name is a valid Python/CGNS name
  """
  if isinstance(name, str):
    if name not in ['', '.', '..'] and not '/' in name:
      return len(name) <= 32 if check_len else True
  return False

def is_valid_value(value) -> bool:
  """
  Return True if value is a valid Python/CGNS Value
  """
  if value is None:
    return True
  if isinstance(value, np.ndarray):
    return value.flags.f_contiguous if value.ndim > 1 else True
  return False

def is_valid_children(children) -> bool:
  """
  Return True if children is a valid Python/CGNS Children
  """
  return isinstance(children, (list, tuple))

def is_valid_label(label, only_sids: bool=True) -> bool:
  """
  Return True if label is a valid Python/CGNS Label
  """
  legacy_labels     = ['"int[1+...+IndexDimension]"', '"int[IndexDimension]"', '"int"']
  additional_labels = ['DiffusionModel_t', 'Transform_t', 'InwardNormalIndex_t', 'EquationDimension_t']

  if isinstance(label, str) and (label.endswith('_t') or label in legacy_labels):
    if only_sids:
      return label in CGK.Label.__members__ or label in legacy_labels or label in additional_labels
    else:
      return True
  return False

def is_valid_node(node) -> bool:
  if isinstance(node, (tuple, list)) and len(node)==4:
    return is_valid_name(N.get_name(node))         and is_valid_value(N.get_value(node, True)) and \
           is_valid_children(N.get_children(node)) and is_valid_label(N.get_label(node))
  return False

