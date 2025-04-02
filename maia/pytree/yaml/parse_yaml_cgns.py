import ast
import warnings
import numpy as np
import yaml

from maia.pytree.typing import *
import maia.pytree.node as N
import maia.pytree.walk as W
import maia.pytree.sids as S

import maia.pytree.cgns_keywords as CGK

TREE_CHILDREN = {'CGNSBase_t', 'CGNSLibraryVersion_t', 'UserDefinedData_t'}
BASE_CHILDREN = {'Axisymmetry_t', 'BaseIterativeData_t', 'DataClass_t', 'Descriptor_t', 'DimensionalUnits_t', 'Family_t',
                 'FlowEquationSet_t', 'ConvergenceHistory_t', 'Gravity_t', 'IntegralData_t', 'ReferenceState_t',
                 'RotatingCoordinates_t', 'SimulationType_t', 'UserDefinedData_t', 'ParticleZone_t', 'Zone_t'}
# Node start / end by space -> KO

def parse_node(node):
  name,label_value = node.split(" ", 1)
  name = name + ' ' # Space is cut by split -> read it

  next_token = label_value.strip().split(" ", 1)[0]
  try:
    while not N.check.is_valid_label(next_token, only_sids=False):
      pre, token, post = label_value.partition(next_token)
      name = name + pre + token
      label_value = post
      next_token = label_value.strip().split(" ", 1)[0]
  except ValueError:
    raise ValueError(f"Unable to parse line {node} : unrecognized label")
    
  name = name.rstrip() # Remove trailing space, especially if we did not enter while loop

  label_value = label_value.strip().split(" ", 1)
  label = label_value[0].strip()
  if len(label_value)==1:
    value = None
  else:
    svalue = label_value[1].replace(':', '')
    svalue = svalue.replace('\n', '')
    value = svalue.strip()
    if len(value) > 2 and value[:2] in CGK.cgns_types:
      cgns_type = value[:2]
      value     = value[2:].strip().replace(' ', '')
      py_value = ast.literal_eval(value)
      value = np.array(py_value, dtype=CGK.cgns_to_dtype[cgns_type], order='F')
      if value.dtype != CGK.cgns_to_dtype[cgns_type]:
        value = value.astype(CGK.cgns_to_dtype[cgns_type])
    else:
      py_value = ast.literal_eval(value)
      value = N.access._convert_value(py_value)
  return name,label,value

def extract_value(sub_nodes):
  if sub_nodes is None:
    return None

  for data_type,np_dtype in CGK.cgns_to_dtype.items():
    value = sub_nodes.pop(data_type,None)
    if value is not None:
      return np.array(value, order='F', dtype=np_dtype)

  return None

def parse_yaml_dict(yaml_dict):
  t = []
  for node,sub_nodes in yaml_dict.items():
    name,label,value = parse_node(node)

    # other way to specify the value
    other_value = extract_value(sub_nodes)
    assert (value is None) or (other_value is None) # two ways to specify a value, but only one possible at once!
    if value is None:
      value = other_value

    if sub_nodes is None:
      children = []
    else:
      children = parse_yaml_dict(sub_nodes)
    t += [[name,value,children,label]]
  return t

def to_nodes(yaml_stream) -> List[CGNSTree]:
  """ Convert a yaml stream into a list of python CGNSTree.

  This function is similar to :func:`to_node`, but allows
  to declare several root nodes at the yaml top level, which
  are parsed independently.

  Args:
    yaml_stream (str or filename): Yaml description of the nodes
  Returns:
    list of CGNSTree : python representation of each root node
  Example:
    >>> nodes = PT.yaml.to_nodes('''
    BC1 BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[1,2,3]]:
    BC2 BC_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[1,2,3]]:
    ''')
    >>> len(nodes)
    2
  """
  if yaml_stream=="":
    return []
  else:
    yaml_dict = yaml.safe_load(yaml_stream)
    return parse_yaml_dict(yaml_dict)

def to_node(yaml_stream) -> CGNSTree:
  """ Convert a yaml stream into a python CGNSTree.

  Tree is parsed recursively, but must start from a single
  root node.

  Args:
    yaml_stream (str or filename): Yaml description of the node
  Returns:
    CGNSTree : python representation of the node
  Example:
    >>> node = PT.yaml.to_node('''
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[1,2,3]]:
    ''')
    >>> PT.print_tree(node)
    BC BC_t 
    ├───GridLocation GridLocation_t "FaceCenter"
    └───PointList IndexArray_t I4 [[1 2 3]]
  """
  if yaml_stream=="":
    return None
  else:
    nodes = to_nodes(yaml_stream)
    assert len(nodes) == 1, f"Cannot convert yaml tree with {len(nodes)} to single CGNS node. Use to_nodes"
    return nodes[0]

def to_cgns_tree(yaml_stream) -> CGNSTree:
  """ Convert a yaml stream into a top level python CGNSTree.

  This function is similar to :func:`to_node` or :func:`to_nodes`,
  but it also automatically create the top level (``CGNSTree_t``, 
  ``CGNSBase_t`` and ``CGNSLibraryVersion_t``) nodes if necessary.

  This function should not be called on nodes lower than ``Zone_t``.

  Args:
    yaml_stream (str or filename): Yaml description of the tree
  Returns:
    CGNSTree : python representation of the tree
  Example:
    >>> tree = PT.yaml.to_cgns_tree('''
    Zone Zone_t:
      ZoneType ZoneType_t "Structured":
    ''')
    >>> PT.print_tree(tree)
    CGNSTree CGNSTree_t 
    ├───Base CGNSBase_t I4 [3 3]
    │   └───Zone Zone_t 
    │       └───ZoneType ZoneType_t "Structured"
    └───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
  """
  t = N.new_node('CGNSTree', 'CGNSTree_t')
  childs = to_nodes(yaml_stream)
  top_labels = [N.get_label(c) for c in childs]
  if len(childs) == 0:
    pass # yaml_stream is empty
  # Detect if yaml is already a top level CGNSTree
  elif 'CGNSTree_t' in top_labels:
    if len(top_labels) > 1:
      raise ValueError("Multiple top level CGNSTree_t nodes is not allowed")
    t = childs[0]
  elif set(top_labels) <= TREE_CHILDREN:
    N.set_children(t, childs)
  elif set(top_labels) <= BASE_CHILDREN:
    b = N.new_CGNSBase(parent=t)
    N.set_children(b, childs)
    zone_node, gc_n = W.get_child_from_labels(b, ['Zone_t', 'GridCoordinates_t'], ancestors=True)
    if zone_node is not None:
      phy_dim = 3
      if gc_n is not None:
        coords_n = W.get_children_from_predicate(gc_n, lambda n: N.get_label(n) == 'DataArray_t' and N.get_name(n) != 'CoordinateTransform')
        phy_dim = len(coords_n)
      try:
        cell_dim = S.Zone.CellDimension(zone_node)
      except:
        cell_dim = 3
      cell_dim = min(cell_dim, phy_dim)
      N.set_value(b, [cell_dim, phy_dim])
    else:
      warnings.warn(f"Can not guess CGNSBase_t dimension, use default value [3,3]", RuntimeWarning, stacklevel=2)
  else:
    raise ValueError("Unvalid nodes label, Base level or Zone level nodes are expected")
  
  if W.get_child_from_label(t, 'CGNSLibraryVersion_t') is None:
    N.add_child(t, N.new_node('CGNSLibraryVersion', 'CGNSLibraryVersion_t', value=4.2))
  return t
