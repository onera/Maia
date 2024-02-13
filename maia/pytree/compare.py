import numpy as np

from maia.pytree.typing import *

import maia.pytree as PT
from maia.pytree.graph.cgns import step, zip_depth_first_search

__all__ = ['is_same_node', 'is_same_tree', 'diff_tree']

class DiffReport(NamedTuple):
  """ Stores the output of :func:`diff_tree`
  
  Parameters:
    status (bool): ``True`` if trees are identical
    errors (str): differences between the two trees
    warnings (str) : minor differences between the two trees
  """
  status:bool
  errors:str
  warnings:str

CompFunction = Callable[[List[Tuple[CGNSTree,CGNSTree]]], DiffReport]

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

def is_same_value_shape(n0: CGNSTree, n1: CGNSTree) -> bool:
  val0 = PT.get_value(n0, raw=True)
  val1 = PT.get_value(n1, raw=True)
  shape0 = val0.shape if val0 is not None else None
  shape1 = val1.shape if val1 is not None else None
  return shape0 == shape1


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
  elif not is_same_value_shape(n0, n1):
    return False
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
    DiffReport(
      status=False,
      errors='/FlowSolution/Density -- Values differ: [1.    1.002 1.   ] <> [1.    1.001 1.   ]\\n',
      warnings=''
      )
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
    DiffReport(status=True, errors='', warnings='')
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


def str_comp(nodes_stack):
  node_x,node_ref = nodes_stack[-1]
  x   = PT.get_value(node_x, raw=True)
  ref = PT.get_value(node_ref, raw=True)
  if np.array_equal(x,ref):
    return True, '', ''
  else:
    return False, f'{PT.get_value(node_x)} <> {PT.get_value(node_ref)}', ''

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

    name = PT.get_name(n0)
    vkind = PT.get_value_type(n0)
    if not is_same_label(n0,n1):
      err_report = path + PT.get_name(n0) + ' -- Labels differ: ' + PT.get_label(n0) + ' <> ' + PT.get_label(n1) + '\n'
    elif not is_same_value_type(n0, n1, strict_value_type):
      err_report = path + PT.get_name(n0) + ' -- Value types differ: ' + str(PT.get_value_type(n0)) + ' <> ' + str(PT.get_value_type(n1)) + '\n'
    elif not is_same_value_shape(n0, n1) and vkind != 'C1': #Filter str, because we do a full print for it
      err_report = path + PT.get_name(n0) + ' -- Value shape differ: ' + str(n0[1].shape) + ' <> ' + str(n1[1].shape) + '\n'
    else:
      if vkind == 'MT':
        is_ok, err_report, warn_report = True, '', ''
      elif vkind == 'C1': # STR
        is_ok, err_report, warn_report = str_comp(nodes_stack)
      else: #Numerics -> call value_comp
        is_ok, err_report, warn_report = value_comp(nodes_stack)
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

def diff_tree(t1:CGNSTree, t2:CGNSTree, strict_value_type = True, comp:CompFunction = None) -> DiffReport:
  """ Report the differences between two trees

  This function is similar to :func:`is_same_tree`, but returns a full report of differences between
  the two input trees. In addition, it is possible to provide a custom comparison function 
  for numerical arrays or to choose one in the following list:

  - :class:`maia.pytree.compare.EqualArray`: two arrays are equal if all their
    elements are one-to-one exactly equal. This is the default comparison method.
  - :class:`maia.pytree.compare.CloseArray`: two arrays are equal if all their
    elements are one-to-one close up to a given tolerance.

  Args:
    t1 (CGNSTree): first tree
    t2 (CGNSTree): second tree
    strict_value_type (bool): if True, allow comparaison of compatible but different types (I4/I8 or R4/R8).
      Otherwise, nodes are considered to differ.
    comp: comparison function to check the value of nodes (see above)
  Returns:
    (:class:`~maia.pytree.compare.DiffReport`) : Difference report. First value indicates if trees are identical, second and
    third store the differences between trees, encoded as strings (respectivly errors and warnings).
  
  Example:
    >>> zone1 = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='ROW1')
    >>> zone2 = PT.new_Zone(type='Unstructured', size=[[9,4,0]], family='ROW2')
    >>> PT.diff_tree(zone1, zone2)
    DiffReport(status=False, errors='/Zone/FamilyName -- Values differ: ROW1 <> ROW2\\n', warnings='')
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
  return DiffReport(v.is_ok, v.err_report, v.warn_report)
