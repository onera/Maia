from maia.pytree.typing import *
from maia.pytree import walk
from .           import access as NA

UNSET = Ellipsis

def new_node(name:str='Node', label:str='UserDefined_t', value:Any=None, children:List[CGNSTree]=[], parent:CGNSTree=None) -> CGNSTree:
  """ Create a new node """
  node = ['Node', None, [], 'UserDefined_t']
  # Use update method to enable checks through the set_ functions
  update_node(node, name, label, value, children)
  if parent is not None:
    NA.add_child(parent, node)
  return node

def update_node(node:CGNSTree, name:str=UNSET, label:str=UNSET, value:Any=UNSET, children:List[CGNSTree]=UNSET):
  if name is not UNSET:
    NA.set_name(node, name)
  if label is not UNSET:
    NA.set_label(node, label)
  if value is not UNSET:
    NA.set_value(node, value)
  if children is not UNSET:
    NA.set_children(node, children)

# def create_child(parent, name, label='UserDefined_t', value=None, children=[]):
  # walk.rm_children_from_name(parent, name)
  # return new_node(name, label, value, children, parent)

def new_child(parent:CGNSTree, name:str, label:str='UserDefined_t', value:Any=None, children:List[CGNSTree]=[]) -> CGNSTree:
  return new_node(name, label, value, children, parent)

def update_child(parent:CGNSTree, name:str, label:str=UNSET, value:Any=UNSET, children:List[CGNSTree]=UNSET) -> CGNSTree:
  node = walk.get_child_from_name(parent, name)
  if node is None:
    node = new_node(name, parent=parent)
  update_node(node, ..., label, value, children)
  return node

def shallow_copy(t:CGNSTree) -> CGNSTree:
  out = [NA.get_name(t), NA.get_value(t, raw=True), [], NA.get_label(t)]
  for child in NA.get_children(t):
    out[2].append(shallow_copy(child))
  return out

def deep_copy(t:CGNSTree) -> CGNSTree:
  out = new_node(NA.get_name(t), NA.get_label(t))
  _val = NA.get_value(t, raw=True)
  if _val is not None:
    out[1] = _val.copy(order='K')
  for child in NA.get_children(t):
    out[2].append(deep_copy(child))
  return out



