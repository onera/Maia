import sys
import numpy as np
import warnings

if sys.version_info.major == 3 and sys.version_info.major < 8:
  from collections.abc import Iterable  # < py38
else:
  from typing import Iterable

import maia.pytree.cgns_keywords as CGK

from . import check

CGNS_STR_SIZE = 32

def _flatten(items):
  """Yield items from any nested iterable; see https://is.gd/gE6gjc """
  for x in items:
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
      yield from _flatten(x)
    else:
      yield x

def _convert_value(value):
  """
  Convert a Python input to a compliant pyCGNS value
  """
  result = None
  if value is None:
    return result
  # value as a numpy: convert to fortran order
  if isinstance(value, np.ndarray):
    # print(f"-> value as a numpy")
    if value.flags.f_contiguous:
      result = value
    else:
      result = np.asfortranarray(value)
  # value as a single value
  elif isinstance(value, float):   # "R8"
    # print(f"-> value as float")
    result = np.array([value],'d')
  elif isinstance(value, int):     # "I4"
    # print(f"-> value as int")
    result = np.array([value],'i')
  elif isinstance(value, str):     # "C1"
    # print(f"-> value as str with {CGK.cgns_to_dtype[CGK.C1]}")
    result = np.array([c for c in value], CGK.cgns_to_dtype[CGK.C1])
  elif isinstance(value, CGK.dtypes):
    # print(f"-> value as CGK.dtypes with {np.dtype(value)}")
    result = np.array([value], np.dtype(value))
  # value as an iterable (list, tuple, set, ...)
  elif isinstance(value, Iterable):
    # print(f"-> value as Iterable : {_flatten(value)}")
    try:
      first_value = next(_flatten(value))
      if isinstance(first_value, float):                       # "R8"
        # print(f"-> first_value as float")
        result = np.array(value, dtype=np.float64, order='F')
      elif isinstance(first_value, int):                       # "I4"
        # print(f"-> first_value as int")
        result = np.array(value, dtype=np.int32, order='F')
      elif isinstance(first_value, str):                       # "C1"
        # print(f"-> first_value as with {CGK.cgns_to_dtype[CGK.C1]}")
        # WARNING: string numpy is limited to rank=2
        assert max([len(v) for v in _flatten(value)]) <= CGNS_STR_SIZE
        size = CGNS_STR_SIZE
        if isinstance(value[0], str):
          v = np.empty( (size,len(value) ), dtype='c', order='F')
          for c, i in enumerate(value):
            s = min(len(i),size)
            v[:,c] = ' '
            v[0:s,c] = i[0:s]
          result = v
        else:
          v = np.empty( (size,len(value[0]),len(value) ), dtype='c', order='F')
          v[:,:,:] = ' '
          for c in range(len(value)):
            for d in range(len(value[c])):
              s = min(len(value[c][d]),size)
              v[0:s,d,c] = value[c][d][0:s]
          result = v
      elif isinstance(first_value, CGK.dtypes):
        result = np.array(value, dtype=np.dtype(first_value), order='F')
    except StopIteration:
      # empty iterable
      result = np.array(value, dtype=np.int32, order='F')
  else:
    # print(f"-> value as unknown type")
    result = np.array([value], order='F')
  return result

def _np_to_string(array):
  #Generic:  32 / taille de la premiere liste / nombre de listes
  if array.ndim == 1:
    return array.tobytes().decode().strip()
  elif array.ndim == 2:
    return [_np_to_string(array[:,i]) for i in range(array.shape[1])]
  elif array.ndim == 3:
    return [_np_to_string(array[:,:,i]) for i in range(array.shape[2])]
  raise ValueError(f"Incorrect dimension for bytes array: {array.ndim}")

def get_name(node):
  """ Return the name of the input CGNSNode """
  return node[0]

def set_name(node, name):
  if check.is_valid_name(name, check_len=False):
    if not check.is_valid_name(name, check_len=True):
      warnings.warn("Setting a CGNS node name with a string longer than 32 char", RuntimeWarning, stacklevel=2)
    node[0] = name
  else:
    raise ValueError("Unvalid name for node")

def get_value(node, raw=False):
  """ Return the value of the input CGNSNode """
  raw_val = node[1]
  if not raw and isinstance(raw_val, np.ndarray) and raw_val.dtype.kind == 'S':
    return _np_to_string(raw_val)
  else:
    return raw_val

def set_value(node, value):
  node[1] = _convert_value(value)

def get_children(node):
  """ Return the list of children of the input CGNSNode """
  return node[2]

def add_child(node, child):
  if get_name(child) in [get_name(n) for n in get_children(node)]:
    raise RuntimeError('Can not add child : a node with the same already exists')
  node[2].append(child)

def set_children(node, children):
  children_bck = get_children(node)
  node[2] = []
  try:
    for child in children:
      add_child(node, child)
  except Exception as e:
    node[2] = children_bck
    raise e 

def get_label(node):
  """ Return the label of the input CGNSNode """
  return node[3]

def set_label(node, label):
  if check.is_valid_label(label, only_sids=False):
    if not check.is_valid_label(label, only_sids=True):
      warnings.warn("Setting a CGNS node label with a non sids label", RuntimeWarning, stacklevel=2)
    node[3] = label
  else:
    raise ValueError("Unvalid label for node")

def get_names(nodes):
  """ Return a list of name from a list of nodes """
  return [get_name(node) for node in nodes]
