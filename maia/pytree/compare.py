import sys
if sys.version_info.major == 3 and sys.version_info.major < 8:
  from collections.abc import Iterable  # < py38
else:
  from typing import Iterable
from functools import wraps
import numpy as np

from maia.pytree.typing import *

import maia.pytree as PT
from maia.pytree.graph.cgns import step, zip_depth_first_search


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
# BASIC COMPARISON

def is_same_name(n0: CGNSTree, n1: CGNSTree) -> bool:
  return PT.get_name(n0) == PT.get_name(n1)

def is_same_label(n0: CGNSTree, n1: CGNSTree) -> bool:
  return PT.get_label(n0) == PT.get_label(n1)

def is_same_value_type(n0: CGNSTree, n1: CGNSTree, strict=True) -> bool:
  if strict:
    return PT.get_value_type(n0) == PT.get_value_type(n1)
  else:
    return PT.get_value_kind(n0) == PT.get_value_kind(n1)


def is_same_value(n0: CGNSTree, n1: CGNSTree, abs_tol:float=0., type_tol=False) -> bool:
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

def is_same_node(node1:CGNSTree, node2:CGNSTree, abs_tol:float=0, type_tol=False) -> bool:
  """
  Compare two nodes

  Nodes are considered equal if they have the same name, label and value.
  Note that no check is performed on their children.

  Args:
    t1 (CGNSTree): first tree
    t2 (CGNSTree): second tree
    abs_tol (float) : absolute tolerance used for value comparison, passed to ``np.allclose`` function
    type_tol (bool): if True, allow comparaison of compatible but different types (I4/I8 or R4/R8).
      Otherwise, nodes are considered to differ.
  Returns:
    bool : True if nodes are identical
  Example:
    >>> zone1 = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='ROTOR')
    >>> zone2 = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='STATOR')
    >>> PT.is_same_node(zone1, zone2)
    True
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

def is_same_tree(t1:CGNSTree, t2:CGNSTree, abs_tol:float=0, type_tol=False) -> bool:
  """
  Compare recursively two trees

  Trees are considered equal if they recursively have the same children (order does not matters),
  in the sense of :func:`is_same_node`.

  See :func:`is_same_node` for arguments description.

  Returns:
    bool : True if trees are identical
  Example:
    >>> zone1 = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='ROTOR')
    >>> zone2 = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='STATOR')
    >>> PT.is_same_tree(zone1, zone2)
    False
  """
  v = same_tree_visitor(abs_tol, type_tol)
  zip_depth_first_search([t1,t2], v)
  return v.is_same

# --------------------------------------------------------------------------
# DIFF TREE

def _report_diff(x, ref, is_equal):
  if is_equal.all():
    return True, '', ''
  elif x.size < 10:
    return False, str(x) + ' <> ' + str(ref), ''
  else:
    n_not_eq = x.size - np.count_nonzero(is_equal)
    return False, f'{n_not_eq} values are different', ''

class EqualArray:
  """
  A callable object generating a report for diff_tree, using an exact point-to-point
  comparison.

  Example:
    >>> sol1 = PT.new_FlowSolution(fields={'Density' : [1., 1.002, 1.]})
    >>> sol2 = PT.new_FlowSolution(fields={'Density' : [1., 1.001, 1.]})
    >>> comp = PT.compare.EqualArray()
    >>> PT.diff_tree(sol1, sol2, comp=comp)
    (False,
    '/FlowSolution/Density -- Values differ: [1.    1.002 1.   ] <> [1.    1.001 1.   ]\\n',
    '')
  """
  def __call__(self, nodes_stack):
    node_x,node_ref = nodes_stack[-1]
    x   = PT.get_value(node_x, raw=True)
    ref = PT.get_value(node_ref, raw=True)
    eq = np.equal(x, ref)
    return _report_diff(x, ref, eq)

class CloseArray:
  """
  A callable object generating a report for diff_tree, using a point-to-point with
  tolerance comparison
  (see `np.isclose
  <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#numpy.isclose>`_
  documentation).

  Args:
    rtol (float) : relative tolerance
    atol (float) : absolute tolerance
  Example:
    >>> sol1 = PT.new_FlowSolution(fields={'Density' : [1., 1.002, 1.]})
    >>> sol2 = PT.new_FlowSolution(fields={'Density' : [1., 1.001, 1.]})
    >>> comp = PT.compare.CloseArray(rtol=0, atol=1E-2)
    >>> PT.diff_tree(sol1, sol2, comp=comp)
    (True, '', '')
  """
  def __init__(self, rtol=1e-05, atol=1e-08):
    self.rtol = rtol
    self.atol = atol
  def __call__(self, nodes_stack):
    node_x,node_ref = nodes_stack[-1]
    x   = PT.get_value(node_x, raw=True)
    ref = PT.get_value(node_ref, raw=True)
    close = np.isclose(x, ref, self.atol, self.rtol)
    return _report_diff(x, ref, close)


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

  next_step = step.over # do not continue comparing children for now
  if n0 is None:
    err_report = '> ' + path + PT.get_name(n1) + '\n'
  elif n1 is None:
    err_report = '< ' + path + PT.get_name(n0) + '\n'
  elif not is_same_name(n0, n1):
    err_report = '< ' + path + PT.get_name(n0) + '\n' \
               + '> ' + path + PT.get_name(n1) + '\n'

  else:
    next_step = step.into # since everything it the same up to now, continue comparing children

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

DiffReport = Tuple[bool,str,str]
CompFunction = Callable[[List[Tuple[CGNSTree,CGNSTree]]], DiffReport]

def diff_tree(t1:CGNSTree, t2:CGNSTree, strict_value_type = True, comp:CompFunction = None) -> DiffReport:
  """ Report the differences between two trees

  This function is similar to :func:`is_same_tree`, but returns a full report of differences between
  the two input trees. In addition, it is possible to provide a custom comparison function 
  for numerical arrays or to choose one in the following list:

  - :class:`maia.pytree.compare.EqualArray`: compare exactly. Two arrays are equal if all their
    elements one-to-one are equal. This is the default comparison method.
  - :class:`maia.pytree.compare.CloseArray`: compare exactly. Two arrays are equal if all their
    elements are close, up to a given tolerance.


  Args:
    t1 (CGNSTree): first tree
    t2 (CGNSTree): second tree
    strict_value_type (bool): if True, allow comparaison of compatible but different types (I4/I8 or R4/R8).
      Otherwise, nodes are considered to differ.
    comp: comparison function to check the value of nodes (see above)
  Returns:
    (bool, str, str) : Difference report. First value indicates if trees are identical, second and
    third store the differences between trees, encoded as strings (respectivly errors and warnings).
  
  Example:
    >>> zone1 = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='ROTOR')
    >>> zone2 = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='STATOR')
    >>> PT.diff_tree(zone1, zone2)
    (False, '< /Zone/FamilyName\\n', '')
  """

  """
  TODO : this functions has been hidden in docstring, because they should be in maia
  and not in maia.pytree (beause of comm).

  `maia.pytree.compare_arrays.field_comparison(tol, comm)`: compare scalar fields with a relative tolerance
  `maia.pytree.compare_arrays.tensor_field_comparison(tol, comm)`: compare tensor fields with a relative tolerance
  """
  if comp is None:
    comp = EqualArray()
  v = diff_tree_visitor(strict_value_type, comp)
  zip_depth_first_search([t1,t2], v, depth='all')
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
