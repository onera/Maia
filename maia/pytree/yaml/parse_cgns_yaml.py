import warnings
import numpy as np
from maia.pytree.typing import *
import maia.pytree.node as N
import maia.pytree.cgns_keywords as CGK

def generate_line(node:CGNSTree, lines:List[str], ident:int=0, line_max:int=120):
  """
  Recursive function Writting a single line in yaml format for a CGNSNode : depending of line_max value,
  line will be formated as
  - "  NodeName NodeLabel DataType DataValue:" (for short lines, DataType and DataValue beeing optionnal)
  - "  NodeName NodeLabel:
         DataType : DataValue" (for long lines)
  In both case, correct level of indentation is set.
  The list of all generated lines is returned
  """

  # Get value and type
  node_value = N.get_value(node)
  if isinstance(node_value, np.ndarray) and node_value.size == 1:
    if N.get_label(node) not in ['IndexArray_t', 'DataArray_t']:
      node_value = node_value[0]
  value_type = None
  value      = None
  if isinstance(node_value, str):
    value = f"'{node_value}'"
  elif isinstance(node_value, (int, float, list, np.integer, np.float32)):
    value = str(node_value)
  elif isinstance(node_value, (tuple, set)):
    value = str(list(node_value))
  elif isinstance(node_value, np.ndarray):
    value_type = f"{CGK.dtype_to_cgns[node_value.dtype]}"
    value = f"{node_value.tolist()}"
  elif node_value is not None:
    warnings.warn(f'Unable to convert value of node {N.get_name(node)}', RuntimeWarning)

  #Short lines, with or without value_type / value
  if value:
    suffix = f" {value_type} {value}:" if value_type else f" {value}:"
  else:
    suffix = ":"
  line = f"{' '*ident}{N.get_name(node)} {N.get_label(node)}{suffix}"

  #Split long lines
  if len(line) > line_max and value_type: #Only true array can be multiline
    first_line = f"{' '*ident}{N.get_name(node)} {N.get_label(node)}:\n{' '*(ident+2)}{value_type} : "
    data_line = ''
    count = len(f"{' '*(ident+2)}{value_type} : ")
    values = value.split(',') #type:ignore #(if value_type is not None, then value is not None)
    for i, val in enumerate(values):
      data_line += f"{val}," 
      count += len(val)+1
      if count >= line_max and i != len(values)-1: #Dont split values, start a new line only after a ','
        prefix = f"\n{' '*(ident+7)}"
        data_line += prefix
        count = len(prefix)-1
    line = first_line + data_line[:-1] #Remove last comma

  lines.append(line)
  for child in N.get_children(node):
    generate_line(child, lines, ident+2, line_max)


def to_yaml(t:CGNSTree, max_line_size=120, write_root=True) -> List[str]:
  """ Convert a python CGNSTree to a yaml string.

  Args:
    t (CGNSTree): input python CGNSTree
    max_line_size (int) : Maximum line lenght using when rendering arrays. Defaults to 120.
    write_root (bool): if ``False``, top_level node is ignored
  Returns:
    list of str : yaml representation of the tree
  Example:
    >>> node = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='ROW')
    >>> lines = PT.yaml.to_yaml(node)
    >>> for l in lines:
    ...    print(l)
    Zone Zone_t I4 [[9, 4, 0]]:
      ZoneType ZoneType_t 'Unstructured':
      FamilyName FamilyName_t 'ROW':
  """
  lines:List[str] = []
  if write_root:
    generate_line(t, lines=lines, line_max=max_line_size)
  else:
    for node in N.get_children(t):
      generate_line(node, lines=lines, line_max=max_line_size)
  return lines

