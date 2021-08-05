import numpy as np

from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
from functools import wraps

import Converter.Internal as I

from maia.sids.cgns_keywords import Label as CGL
import maia.sids.cgns_keywords as CGK

TreeNode = List[Union[str, Optional[np.ndarray], List["TreeNode"]]]


class CGNSNodeFromPredicateNotFoundError(Exception):
    """
    Attributes:
        node (List): CGNS node
        name (str): Name of the CGNS Name
    """
    def __init__(self, node: List, predicate):
        self.node = node
        self.predicate = predicate
        super().__init__()

    def __str__(self):
        return f"Unable to find the predicate '{self.predicate}' from the CGNS node '[n:{I.getName(self.node)}, ..., l:{I.getType(self.node)}]', see : \n{I.printTree(self.node)}."

class CGNSLabelNotEqualError(Exception):
    """
    Attributes:
        node (List): CGNS node
        label (str): Name of the CGNS Label
    """
    def __init__(self, node: List, label: str):
        self.node  = node
        self.label = label
        super().__init__()

    def __str__(self):
        return f"Expected a CGNS node with label '{self.label}', '[n:{I.getName(self.node)}, ..., l:{I.getType(self.node)}]' found here."

class NotImplementedForElementError(NotImplementedError):
    """
    Attributes:
        zone_node (List): CGNS Zone_t node
        element_node (List): CGNS Elements_t node
    """
    def __init__(self, zone_node: List, element_node: List):
        self.zone_node    = zone_node
        self.element_node = element_node
        super().__init__()

    def __str__(self):
        return f"Unstructured CGNS Zone_t named '{I.getName(self.zone_node)}' with CGNS Elements_t named '{SIDS.ElementCGNSName(self.element_node)}' is not yet implemented."

# --------------------------------------------------------------------------
def is_valid_name(name: str):
  """
  Return True if name is a valid Python/CGNS name
  """
  return isinstance(name, str) #and len(label) < 32

def is_valid_value(value):
  """
  Return True if label is a valid Python/CGNS Label
  """
  return isinstance(value, np.ndarray) and np.isfortran(value) if value.ndim > 1 else True

def is_valid_children(children):
  """
  Return True if label is a valid Python/CGNS Children
  """
  return isinstance(children, (list, tuple))

def is_valid_label(label):
  """
  Return True if label is a valid Python/CGNS Label
  """
  return isinstance(label, str) and ((label.endswith('_t') and label in CGK.Label.__members__) or label == '')

# --------------------------------------------------------------------------
def check_name(name: str):
  if is_valid_name(name):
    return name
  raise TypeError(f"Invalid Python/CGNS name '{name}'")

def check_value(value):
  if is_valid_value(value):
    return value
  raise TypeError(f"Invalid Python/CGNS value '{value}'")

def check_children(children):
  if is_valid_children(children):
    return children
  raise TypeError(f"Invalid Python/CGNS children '{children}'")

def check_label(label):
  if is_valid_label(label):
    return label
  raise TypeError(f"Invalid Python/CGNS label '{label}'")

# --------------------------------------------------------------------------
def is_valid_node(node):
  if not isinstance(node, list) and len(node) != 4 and \
      is_valid_name(I.getName(node))         and \
      is_valid_value(I.getValue(node))       and \
      is_valid_children(I.getChildren(node)) and \
      is_valid_label(I.getName(node)) :
    return False
  return True

# --------------------------------------------------------------------------
def check_is_label(label, n=0):
  def _check_is_label(f):
    @wraps(f)
    def wrapped_method(*args, **kwargs):
      node = args[n]
      if I.getType(node) != label:
        raise CGNSLabelNotEqualError(node, label)
      return f(*args, **kwargs)
    return wrapped_method
  return _check_is_label

# --------------------------------------------------------------------------
def is_same_name(n0: TreeNode, n1: TreeNode):
  return n0[0] == n1[0]

def is_same_label(n0: TreeNode, n1: TreeNode):
  return n0[3] == n1[3]

def is_same_value(n0: TreeNode, n1: TreeNode):
  return np.array_equal(n0[1], n1[1])
