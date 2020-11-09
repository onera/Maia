#from ruamel import yaml
#import ruyaml as yaml
from ruamel.yaml import YAML
from ruamel.yaml import parser
import ast
import numpy as np

def parse_node(node):
  name,label_value = node.split(" ", 1)
  name = name.strip()
  label_value = label_value.split(" ", 1)
  label = label_value[0].strip()
  if len(label_value)==1:
    value = None
  else:
    value = ast.literal_eval(label_value[1])
  return name,label,value

def extract_value(sub_nodes):
  if sub_nodes is None:
    return None

  data_types = [  "I4"  ,  "I8"  ,   "R4"   ,   "R8"   ]
  np_dtypes  = [np.int32,np.int64,np.float32,np.float64]
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

def to_pytree(yaml_str):
  try:
    yaml = YAML(typ="safe")
    yaml_dict = yaml.load(yaml_str)
    return parse_yaml_dict(yaml_dict)
  except parser.ParserError as exc:
    print(exc)
    raise exc
