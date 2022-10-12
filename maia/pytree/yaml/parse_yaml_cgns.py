import ast
import numpy as np
from ruamel.yaml import YAML

import maia.pytree.node as N
import maia.pytree.walk as W

import maia.pytree.cgns_keywords as CGK

def parse_node(node):
  name,label_value = node.split(" ", 1)
  name = name.strip()
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

def to_nodes(yaml_stream):
  if yaml_stream=="":
    return []
  else:
    yaml = YAML(typ="safe")
    yaml_dict = yaml.load(yaml_stream)
    return parse_yaml_dict(yaml_dict)

def to_node(yaml_stream):
  if yaml_stream=="":
    return None
  else:
    nodes = to_nodes(yaml_stream)
    assert len(nodes) == 1, f"Cannot convert yaml tree with {len(nodes)} to single CGNS node. Use to_nodes"
    return nodes[0]

def to_cgns_tree(yaml_stream):
  t = N.new_node('CGNSTree', 'CGNSTree_t')
  childs = to_nodes(yaml_stream)
  if len(childs) > 0 and N.get_label(childs[0]) == 'Zone_t':
    b = N.new_CGNSBase(parent=t)
    N.set_children(b, childs)
  else:
    N.set_children(t, childs)
  if W.get_child_from_label(t, 'CGNSLibraryVersion_t') is None:
    N.add_child(t, N.new_node('CGNSLibraryVersion', 'CGNSLibraryVersion_t', value=4.2))
  return t
