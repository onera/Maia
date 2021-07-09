#from ruamel import yaml
#import ruyaml as yaml
from ruamel.yaml import YAML
from ruamel.yaml import parser
import ast
import numpy as np
import Converter.Internal as I

data_types = [  "I4"  ,  "I8"  ,   "R4"   ,   "R8"   ]
np_dtypes  = [np.int32,np.int64,np.float32,np.float64]

def parse_node(node):
  name,label_value = node.split(" ", 1)
  name = name.strip()
  label_value = label_value.strip().split(" ", 1)
  label = label_value[0].strip()
  if len(label_value)==1:
    value = None
  else:
    value = label_value[1].strip()
    if len(value) > 2 and value[:2] in data_types:
      data_type_str = value[:2]
      value         = value[2:].strip()
      i_data_type   = data_types.index(data_type_str)
      data_type_np  = np_dtypes[i_data_type]
      value = np.array(ast.literal_eval(value), dtype=data_type_np)
    else:
      value = ast.literal_eval(label_value[1])
      if isinstance(value, list):
        value = np.array(value)
  return name,label,value

def extract_value(sub_nodes):
  if sub_nodes is None:
    return None

  for data_type,np_dtype in zip(data_types,np_dtypes):
    value = sub_nodes.pop(data_type,None)
    if value is not None:
      return np.array(value,dtype=np_dtype)

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

def to_nodes(yaml_str):
  if yaml_str=="":
    return []
  else:
    yaml = YAML(typ="safe")
    yaml_dict = yaml.load(yaml_str)
    return parse_yaml_dict(yaml_dict)

def to_node(yaml_str):
  if yaml_str=="":
    return None
  else:
    nodes = to_nodes(yaml_str)
    assert len(nodes) == 1, f"Cannot convert yaml tree with {len(nodes)} to single CGNS node. Use to_nodes"
    return nodes[0]

def to_cgns_tree(yaml_str):
  t = I.newCGNSTree()
  childs = to_nodes(yaml_str)
  if len(childs) > 0 and I.getType(childs[0]) == 'Zone_t':
    b = I.newCGNSBase(parent=t)
    I.addChild(b, childs)
  else:
    I.addChild(t, childs)
  return t
