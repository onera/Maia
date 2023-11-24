from maia.pytree.typing import *
from maia.pytree import walk
from .           import access as NA

UNSET = Ellipsis

def new_node(name:str='Node', label:str='UserDefined_t', value:Any=None, children:List[CGNSTree]=[], parent:CGNSTree=None) -> CGNSTree:
  """ Create a new CGNS node

  If ``parent`` is not None, this node is appended as a child to the parent node.
   
  Args:
    name (str): Name of the created node -- see :func:`set_name`
    label (str): Label of the created node -- see :func:`set_label`
    value (Any): Value of the created node -- see :func:`set_value`
    children (List[CGNSTree]): Value of the created node -- see :func:`set_children`
    parent (CGNSTree or None): Other node, where created node should be attached
  Returns:
    CGNSTree: Created node
  Example:
    >>> zone = PT.new_node('Zone', label='Zone_t') # Basic node creation
    >>> PT.new_node('ZoneType', 'ZoneType_t', "Unstructured", 
    ...             parent=zone) # Create and attach to a parent
    >>> PT.print_tree(zone)
    Zone Zone_t 
    └───ZoneType ZoneType_t "Unstructured"
  """
  node = ['Node', None, [], 'UserDefined_t']
  # Use update method to enable checks through the set_ functions
  update_node(node, name, label, value, children)
  if parent is not None:
    NA.add_child(parent, node)
  return node

def update_node(node:CGNSTree, name:str=UNSET, label:str=UNSET, value:Any=UNSET, children:List[CGNSTree]=UNSET):
  """
  update_node(node, name=UNSET, label=UNSET, value=UNSET, children=UNSET)

  Update some attribute of a CGNSNode
   
  Parameters which are provided to the function trigger the update of their
  corresponding attribute.

  Args:
    node (CGNSTree): Input node
    others : See :func:`set_name`, :func:`set_label`, :func:`set_value`, :func:`set_children`

  Example:
    >>> node = PT.new_node('Zone')
    >>> PT.update_node(node, label='Zone_t', value=[[11,10,0]])
    >>> node
    ['Zone', array([[11,10,0]], dtype=int32), [], 'Zone_t']
  """
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
  """ Create a new CGNS node as a child of an other node

  This is an alternative form of :func:`new_node`, with mandatory ``parent`` argument.

  Args:
    all: See :func:`new_node`
  Returns:
    CGNSTree: Created node
  Example:
    >>> zone = PT.new_node('Zone', label='Zone_t') # Basic node creation
    >>> PT.new_child(zone, 'ZoneType', 'ZoneType_t', "Unstructured")
    >>> PT.print_tree(zone)
    Zone Zone_t 
    └───ZoneType ZoneType_t "Unstructured"
  """
  return new_node(name, label, value, children, parent)

def update_child(parent:CGNSTree, name:str, label:str=UNSET, value:Any=UNSET, children:List[CGNSTree]=UNSET) -> CGNSTree:
  """
  update_child(parent, name, label=UNSET, value=UNSET, children=UNSET)

  Create a child, or update its attributes
  
  This is an alternative form of :func:`new_child`, but this function allow the parent node
  to already have a child of the given name: in this case, its attributes are updated.
  Otherwise, the child is created with the default values of :func:`new_child`

  Args:
    all: See :func:`new_child`
  Returns:
    CGNSTree: Created node
  Example:
    >>> zone = PT.new_node('Zone', label='Zone_t') # Basic node creation
    >>> PT.update_child(zone, 'ZoneType', 'ZoneType_t') # Child is created
    >>> PT.update_child(zone, 'ZoneType', value="Unstructured") # Child is updated
    >>> PT.print_tree(zone)
    Zone Zone_t 
    └───ZoneType ZoneType_t "Unstructured"
  """
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



