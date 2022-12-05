import sys
if sys.version_info.major == 3 and sys.version_info.major < 8:
  from collections.abc import Iterable  # < py38
else:
  from typing import Iterable
from typing import List, Optional, NoReturn, Union, Tuple, Callable, Any
from functools import wraps
import numpy as np

import maia.pytree as PT
from maia.pytree.algo.graph import step, zip_depth_first_search
from maia.pytree.compare_arrays import equal_array_comparison

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
        return f"Unable to find the predicate '{self.predicate}' from the CGNS node '[n:{PT.get_name(self.node)}, ..., l:{PT.get_label(self.node)}]"

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
        return f"Expected a CGNS node with label '{self.label}', '[n:{PT.get_name(self.node)}, ..., l:{PT.get_label(self.node)}]' found here."

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
        return f"Unstructured CGNS Zone_t named '{PT.get_name(self.zone_node)}' with CGNS Elements_t named '{SIDS.ElementCGNSName(self.element_node)}' is not yet implemented."

# --------------------------------------------------------------------------
def check_is_label(label, n=0):
  def _check_is_label(f):
    @wraps(f)
    def wrapped_method(*args, **kwargs):
      node = args[n]
      if PT.get_label(node) != label:
        raise CGNSLabelNotEqualError(node, label)
      return f(*args, **kwargs)
    return wrapped_method
  return _check_is_label

# --------------------------------------------------------------------------
def check_in_labels(labels, n=0):
  def _check_in_labels(f):
    @wraps(f)
    def wrapped_method(*args, **kwargs):
      node = args[n]
      if PT.get_label(node) not in labels:
        raise CGNSLabelNotEqualError(node, labels)
      return f(*args, **kwargs)
    return wrapped_method
  return _check_in_labels

# --------------------------------------------------------------------------
def is_same_name(n0: TreeNode, n1: TreeNode):
  return PT.get_name(n0) == PT.get_name(n1)

def is_same_label(n0: TreeNode, n1: TreeNode):
  return PT.get_label(n0) == PT.get_label(n1)

def is_same_value_type(n0: TreeNode, n1: TreeNode, strict=True):
  if strict:
    return PT.get_value_type(n0) == PT.get_value_type(n1)
  else:
    return PT.get_value_kind(n0) == PT.get_value_kind(n1)


def is_same_value(n0: TreeNode, n1: TreeNode, abs_tol=0, type_tol=False):
  """ Compare the values of two single nodes. Node are considered equal if
  they have
  - same data type (if type_tol is True, only kind of types are considered equal eg.
    I4 & I8 have not same type, but have same type kind
  - same array len
  - same value for each element, up to the absolute tolerance abs_tol when array kind is floats
  """
  if not is_same_value_type(n0, n1, strict=not type_tol):
    return False
  if n0[1] is None:
    return True
  elif n0[1].dtype.kind == 'f':
    return np.allclose(n0[1], n1[1], rtol=0, atol=abs_tol)
  else:
    return np.array_equal(n0[1], n1[1])

def is_same_node(node1, node2, abs_tol=0, type_tol=False):
  """
  Compare two single nodes (no recursion). Node are considered equal if
  they have same name, same label, same value.
  Note that no check is performed on children
  """
  return is_same_name(node1, node2) and is_same_label(node1, node2) and is_same_value(node1, node2, abs_tol, type_tol)


class same_tree_visitor:
  def __init__(self, abs_tol, type_tol):
    self.abs_tol = abs_tol
    self.type_tol = type_tol
    self.is_same = True

  def pre(self, ns):
    if ns[0] is None or ns[1] is None or not is_same_node(ns[0], ns[1], self.abs_tol, self.type_tol):
      self.is_same = False
      return step.out
    else:
      return step.into

def value_comparison_report(nodes_stack, comp):
  """ Compare the values of two single nodes. Node are considered equal if
  they have
  - same data type (if type_tol is True, only kind of types are considered equal e.g.
    I4 & I8 have not same type, but have same type kind
  - same array len
  - same value for each element, up to the absolute tolerance abs_tol when array kind is floats
  """
  n0,n1 = nodes_stack[-1]
  v0 = PT.get_value(n0, raw=True)
  v1 = PT.get_value(n1, raw=True)
  if v0 is None and v1 is None:
    return True, '', ''
  else:
    assert v0 is not None and v1 is not None
    return comp(nodes_stack)

def _zip_path(ns):
  path = '/'
  for n0,n1 in ns:
    name0 = PT.get_name(n0)
    name1 = PT.get_name(n1)
    assert name0 == name1
    path += name0 + '/'
  return path

def diff_nodes(nodes_stack, strict_value_type, value_comp):
  n0,n1 = nodes_stack[-1]
  path = _zip_path(nodes_stack[:-1])

  is_ok = False
  warn_report = ''

  next_step = step.over
  if n0 is None:
    err_report = '> ' + path + PT.get_name(n1) + '\n'
  elif n1 is None:
    err_report = '< ' + path + PT.get_name(n0) + '\n'
  elif not is_same_name(n0, n1):
    err_report = '< ' + path + PT.get_name(n0) + '\n' \
               + '> ' + path + PT.get_name(n1) + '\n'

  else:
    next_step = step.into

    if not is_same_label(n0,n1):
      err_report = path + PT.get_name(n0) + ' -- Labels differ: ' + PT.get_label(n0) + ' <> ' + PT.get_label(n1) + '\n'
    elif not is_same_value_type(n0, n1, strict_value_type):
      err_report = path + PT.get_name(n0) + ' -- Value types differ: ' + str(PT.get_value_type(n0)) + ' <> ' + str(PT.get_value_type(n1)) + '\n'
    else:
      is_ok, err_report, warn_report = value_comparison_report(nodes_stack, value_comp)
      name = PT.get_name(n0)
      if hasattr(value_comp,'modify_name'):
        name = value_comp.modify_name(name)
      if err_report != '':
        err_report = path + name + ' -- Values differ: ' + err_report + '\n'
      if warn_report != '':
        warn_report = path + name + ' -- Values differ: ' + warn_report + '\n'

  return next_step, is_ok, err_report, warn_report


class diff_tree_visitor:
  def __init__(self, strict_value_type, value_comp):
    self.value_comp = value_comp
    self.strict_value_type = strict_value_type
    self.is_ok = True
    self.err_report = ''
    self.warn_report = ''

  def pre(self, nodes_stack):
    next_step, is_ok, err_report, warn_report = diff_nodes(nodes_stack, self.strict_value_type, self.value_comp)
    self.is_ok = self.is_ok and is_ok
    self.err_report += err_report
    self.warn_report += warn_report
    return next_step


def is_same_tree(t1, t2, abs_tol=0, type_tol=False):
  """
  Recursive comparison of two nodes. Nodes are considered equal if the pass is_same_node test
  and if the have the same childrens. Children are allowed to appear in a different order.
  """
  v = same_tree_visitor(abs_tol, type_tol)
  zip_depth_first_search([t1,t2], v)
  return v.is_same

def diff_tree(t1, t2, strict_value_type = True, comp = equal_array_comparison()):
  """
  Recursive comparison of two nodes. Nodes are considered equal if the pass is_same_node test
  and if the have the same childrens. Children are allowed to appear in a different order.
  """
  v = diff_tree_visitor(strict_value_type, comp)
  zip_depth_first_search([t1,t2], v, only_nodes=False)
  return v.is_ok, v.err_report, v.warn_report

# --------------------------------------------------------------------------
# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
def flatten_cgns(items):
  from maia.pytree.node.check import is_valid_node
  """Yield items from any nested iterable; see Reference."""
  for x in items:
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)) and not is_valid_node(x):
      yield from flatten_cgns(x)
    else:
      yield x
