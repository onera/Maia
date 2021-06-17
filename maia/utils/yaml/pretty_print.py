from ruamel.yaml import YAML
from ruamel.yaml import parser

def yaml_tree_to_string(yaml_dict,prefix="",first=True):
  res_str = ""
  for i,node in enumerate(yaml_dict):
    sub_nodes = yaml_dict[node]
    is_last = i==(len(yaml_dict)-1)
    if is_last:
      prefix_first_line_new = prefix+"└───"
      prefix_new = prefix+"    "
    else:
      prefix_first_line_new = prefix+"├───"
      prefix_new = prefix+"│   "
    if first:
      prefix_first_line_new = ""
      prefix_new = ""

    print("node = ",node)
    res_str += prefix_first_line_new + node + "\n"
    if sub_nodes is not None:
      res_str += yaml_tree_to_string(sub_nodes,prefix_new,False)
  return res_str


def pretty_tree(yaml_str):
  if yaml_str=="":
    return []
  else:
    yaml = YAML(typ="safe")
    print(yaml_str);
    yaml_dict = yaml.load(yaml_str)
    return yaml_tree_to_string(yaml_dict)

def pretty_print(yaml_str):
  print(pretty_tree(yaml_str))
