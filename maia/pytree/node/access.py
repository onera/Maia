import sys
import numpy as np
import warnings

from maia.pytree.typing import *

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
  # value is None : immediate return
  if value is None:
    return result
  # value as a numpy: convert to fortran order
  if isinstance(value, np.ndarray):
    if value.flags.f_contiguous:
      result = value
    else:
      result = np.asfortranarray(value)
  # value as a single value
  elif isinstance(value, float):   # R4
    result = np.array([value],'f')
  elif isinstance(value, int):     # I4 if possible, I8 otherwise
    dtype = np.int32 if abs(value) < np.iinfo(np.int32).max else np.int64
    result = np.array([value], dtype)
  elif isinstance(value, str):     # C1
    result = np.array([c for c in value], CGK.cgns_to_dtype[CGK.C1])
  elif isinstance(value, CGK.dtypes): # Numpy scalar
    result = np.array([value], np.dtype(value))
  # value as an iterable (list, tuple, set, ...)
  elif isinstance(value, Iterable):
    # print(f"-> value as Iterable : {_flatten(value)}")
    try:
      first_value = next(_flatten(value))
      if isinstance(first_value, float):                       # R4
        result = np.array(value, dtype=np.float32, order='F')
      elif isinstance(first_value, int):                       # I4 if possible, I8 otherwise
        max_val = max([abs(x) for x in _flatten(value)])
        dtype = np.int32 if max_val < np.iinfo(np.int32).max else np.int64
        result = np.array(value, dtype=dtype, order='F')
      elif isinstance(first_value, str):                       # C1
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
      # empty iterable -> default to I4
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

def get_name(node:CGNSTree) -> str:
  """
  Return the name of a CGNSNode

  Args:
    node (CGNSTree): Input node
  Returns:
    str: Name of the node
  Example:
    >>> PT.get_name(PT.new_node(name='MyNodeName', label='Zone_t'))
    'MyNodeName'
  """
  return node[0]
def set_name(node:CGNSTree, name:str):
  """
  Set the name of a CGNSNode

  Args:
    node (CGNSTree): Input node
    name (str): Name to be set
  Warns:
    RuntimeWarning: If name is longer than 32 characters
  Raises:
    ValueError: If name is not valid
  Example:
    >>> node = PT.new_node('Node')
    >>> PT.set_name(node, 'UpdatedNodeName')
  """
  if check.is_valid_name(name, check_len=False):
    if not check.is_valid_name(name, check_len=True):
      warnings.warn("Setting a CGNS node name with a string longer than 32 char", RuntimeWarning, stacklevel=2)
    node[0] = name
  else:
    raise ValueError("Unvalid name for node")

def get_value(node:CGNSTree, raw:bool=False) -> Union[None, np.ndarray, str, List[str]]:
  """ Return the value of a CGNSNode

  If value is an array of characters, it returned as a (or a
  sequence of) string, unless raw parameter is True.
  
  Args:
    node (CGNSTree): Input node
    raw  (bool): If ``True``, always return the numpy array
  Returns:
    CGNSValue: Value of the node
  Example:
    >>> PT.get_value(PT.new_node('MyNode', value=3.14))
    array([3.14], dtype=float32)
  """
  raw_val = node[1]
  if not raw and isinstance(raw_val, np.ndarray) and raw_val.dtype.kind == 'S':
    return _np_to_string(raw_val)
  else:
    return raw_val

def get_value_type(node:CGNSTree) -> str:
  """ Return the value type of a CGNSNode 

  This two letters string identifies the datatype of the node and belongs to
  ``["MT", "B1", "C1", "I4", "U4", "I8", "U8", "R4", "R8", "X4", "X8"]``

  Args:
    node (CGNSTree): Input node
  Returns:
    str: Value type of the node
  Example:
    >>> PT.get_value_type(PT.new_node('MyNode', value=None))
    'MT'
  """
  val = get_value(node, raw=True)
  if val is None:
    return 'MT'
  return CGK.dtype_to_cgns[val.dtype]

def get_value_kind(node:CGNSTree) -> str:
  """ Return the value kind of a CGNSNode

  If node is not empty, this one letter string identifies the
  datakind of the node and belongs to
  ``["MT", "B", "C", "I", "U", "R", "X"]``
  
  Args:
    node (CGNSTree): Input node
  Returns:
    str: Value kind of the node
  Example:
    >>> PT.get_value_kind(PT.new_node('MyNode', value=3.14))
    'R'
  """
  val_type = get_value_type(node)
  if val_type != 'MT':
    val_type = val_type[0]
  return val_type

def set_value(node:CGNSTree, value:Any):
  """
  Set the value of a CGNSNode

  If value is neither ``None`` nor a f-contiguous numpy array,
  it is converted to a suitable numpy array depending on its type:

  - Literal (or sequence of) floats are converted to R4 arrays
  - Literal (or sequence of) ints are converted to I4 arrays if possible, I8 otherwise
  - Numpy scalars keep their corresponding kind
  - Strings are converted to numpy bytearrays
  - Sequence of N strings are converted to numpy (32,N) shaped bytearrays
  - Nested sequence of strings (M sequences of N strings) are converted
    to numpy (32, N, M) shaped bytearrays

  Args:
    node (CGNSTree): Input node
    value (Any): Value to be set
  Example:
    >>> node = PT.new_node('Node')
    >>> PT.set_value(node, [3,2,1])
  """
  node[1] = _convert_value(value)

def get_children(node:CGNSTree) -> List[CGNSTree]:
  """ Return the list of children of a CGNSNode

  Args:
    node (CGNSTree): Input node
  Returns:
    List[CGNSTree]: Children of the node
  Example:
    >>> len(PT.get_children(PT.new_node('MyNode')))
    0
  """
  return node[2]

def add_child(node:CGNSTree, child:CGNSTree):
  """
  Append a child node to the children list of a CGNSNode

  Args:
    node (CGNSTree): Input node
    child (CGNSTree): Child node to be add

  Raises:
    RuntimeError: If a node with same name than ``child`` already exists
  Example:
    >>> node = PT.new_node('Zone', 'Zone_t')
    >>> PT.add_child(node, PT.new_node('ZoneType', 'ZoneType_t', 'Structured'))
  """
  if child is None:
    return
  if get_name(child) in [get_name(n) for n in get_children(node)]:
    raise RuntimeError(f'Can not add child {child[0]} to node {node[0]}: a node with the same name already exists')
  node[2].append(child)

def rm_child(node:CGNSTree, child:CGNSTree):
  if child is None:
    return
  sub_nodes = get_children(node)
  for i, sub_node in enumerate(sub_nodes):
    if sub_node is child:
      break
  else:
    raise RuntimeError('Can not remove child : not found in node')
  sub_nodes.pop(i)

def set_children(node:CGNSTree, children:List[CGNSTree]):
  """
  Set the children list of a CGNSNode

  This will replace the existing children with the provided list.
  See also: :func:`add_child`

  Args:
    node (CGNSTree): Input node
    children (List[CGNSTree]): Children to be set
  Example:
    >>> node = PT.new_node('Zone', 'Zone_t')
    >>> PT.add_child(node, PT.new_node('ZoneType', 'ZoneType_t', 'Structured'))
    >>> PT.set_children(node, []) # Replace the children with the given list
    >>> len(PT.get_children(node))
    0
  """
  children_bck = get_children(node)
  node[2] = []
  try:
    for child in children:
      add_child(node, child)
  except Exception as e:
    node[2] = children_bck
    raise e 

def get_label(node:CGNSTree) -> str:
  """ Return the label of a CGNSNode

  Args:
    node (CGNSTree): Input node
  Returns:
    str: Label of the node
  Example:
    >>> PT.get_label(PT.new_node('Zone', label='Zone_t'))
    'Zone_t'
  """
  return node[3]

def set_label(node:CGNSTree, label:str):
  """
  Set the label of a CGNSNode

  Args:
    node (CGNSTree): Input node
    label (str): Label to be set

  Warns:
    RuntimeWarning: If label does not belong to SIDS label list
  Raises:
    ValueError: If label is not valid
  Example:
    >>> node = PT.new_node('BCNode')
    >>> PT.set_label(node, 'BC_t')
  """
  if check.is_valid_label(label, only_sids=False):
    if not check.is_valid_label(label, only_sids=True):
      warnings.warn("Setting a CGNS node label with a non sids label", RuntimeWarning, stacklevel=2)
    node[3] = label
  else:
    raise ValueError("Unvalid label for node")

def get_names(nodes:List[CGNSTree]) -> List[str]:
  """ Return a list of name from a list of nodes """
  return [get_name(node) for node in nodes]
