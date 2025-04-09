from maia.pytree.typing import *
from maia.pytree.meta   import begin_api_export, end_api_export

from maia.pytree.predicate import match_name, match_label, match_value, match_name_label
from maia.pytree.utils     import path_head, path_tail

from .walkers_api import get_node_from_path

# Keys to access CGNSTree values
__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3

def _rm_nodes_from_predicate_with_level__(parent, predicate, depth, level=1):
  results = []
  for ichild, child in enumerate(parent[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
    else:
      if level < depth:
        _rm_nodes_from_predicate_with_level__(child, predicate, depth, level=level+1)
  for ichild in reversed(results):
    del parent[__CHILDREN__][ichild]


def _rm_nodes_from_predicate__(parent, predicate):
  results = []
  for ichild, child in enumerate(parent[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
    else:
      _rm_nodes_from_predicate__(child, predicate)
  for ichild in reversed(results):
    del parent[__CHILDREN__][ichild]

begin_api_export()

def rm_children_from_predicate(root: CGNSTree, predicate: Callable[[CGNSTree], bool]):
  """
  Remove the children of root node satisfying Predicate function
  """
  results = []
  for ichild, child in enumerate(root[__CHILDREN__]):
    if predicate(child):
      results.append(ichild)
  for ichild in reversed(results):
    del root[__CHILDREN__][ichild]

def rm_children_from_name(root:CGNSTree, name:str):
  """Specialization of rm_children_from_predicate with embedded predicate match_name"""
  return rm_children_from_predicate(root, lambda n : match_name(n, name))
def rm_children_from_label(root:CGNSTree, label:str):
  """Specialization of rm_children_from_predicate with embedded predicate match_label"""
  return rm_children_from_predicate(root, lambda n : match_label(n, label))
def rm_children_from_value(root:CGNSTree, value):
  """Specialization of rm_children_from_predicate with embedded predicate match_value"""
  return rm_children_from_predicate(root, lambda n : match_value(n, value))
def rm_children_from_name_and_label(root:CGNSTree, name:str, label:str):
  """Specialization of rm_children_from_predicate with embedded predicate match_name_label"""
  return rm_children_from_predicate(root, lambda n : match_name_label(n, name, label))

def keep_children_from_predicate(root: CGNSTree, predicate: Callable[[CGNSTree], bool]):
  """
  Remove all the children of root node expect the ones matching the given predicate.

  Args:
      root (CGNSTree): Tree in which nodes are removed
      predicate (callable): condition to keep nodes, which must
        have the following signature: ``f(n:CGNSTree) -> bool``
  Example:
    >>> zone = PT.yaml.to_node('''
    ... Zone Zone_t:
    ...   FamilyName FamilyName_t 'ROW1':
    ...   ZoneBC ZoneBC_t:
    ...     bc1 BC_t:
    ...       FamilyName FamilyName_t 'BC1':
    ...       Index_i IndexArray_t:
    ...     bc2 BC_t:
    ...       FamilyName FamilyName_t 'BC2':
    ...       Index_ii IndexArray_t:
    ... ''')
    >>> PT.keep_children_from_label(zone, 'FamilyName_t')
    >>> PT.print_tree(zone)
    Zone Zone_t 
    └───FamilyName FamilyName_t "ROW1"

  Note:
    This function admits the following shorcuts: 

    - :func:`keep_children_from_name`, :func:`keep_children_from_label`,
      :func:`keep_children_from_value`, :func:`keep_children_from_name_and_label` (embedded predicate)
  """
  rm_children_from_predicate(root, lambda n: not predicate(n))

def keep_children_from_name(root:CGNSTree, name:str):
  """Specialization of keep_children_from_predicate with embedded predicate match_name"""
  return keep_children_from_predicate(root, lambda n : match_name(n, name))
def keep_children_from_label(root:CGNSTree, label:str):
  """Specialization of keep_children_from_predicate with embedded predicate match_label"""
  return keep_children_from_predicate(root, lambda n : match_label(n, label))
def keep_children_from_value(root:CGNSTree, value):
  """Specialization of keep_children_from_predicate with embedded predicate match_value"""
  return keep_children_from_predicate(root, lambda n : match_value(n, value))
def keep_children_from_name_and_label(root:CGNSTree, name:str, label:str):
  """Specialization of keep_children_from_predicate with embedded predicate match_name_label"""
  return keep_children_from_predicate(root, lambda n : match_name_label(n, name, label))


def rm_nodes_from_predicate(root: CGNSTree, predicate: Callable[[CGNSTree], bool], **kwargs):
  """ Remove all the nodes in the input tree matching the given predicate.

  The search can be fine-tuned with the following kwargs:

  - ``depth`` (int): Stop exploring nodes once ``depth`` is reached (0 beeing the node itself, and
    ``None`` meaning unlimited). Defaults to ``None``.

  Args:
      root (CGNSTree): Tree in which nodes are removed
      predicate (callable): condition to remove nodes, which must
        have the following signature: ``f(n:CGNSTree) -> bool``
      **kwargs: Additional options (see above)
  Example:
    >>> zone = PT.yaml.to_node('''
    ... Zone Zone_t:
    ...   FamilyName FamilyName_t 'ROW1':
    ...   ZoneBC ZoneBC_t:
    ...     bc1 BC_t:
    ...       FamilyName FamilyName_t 'BC1':
    ...       Index_i IndexArray_t:
    ...     bc2 BC_t:
    ...       FamilyName FamilyName_t 'BC2':
    ...       Index_ii IndexArray_t:
    ... ''')
    >>> PT.rm_children_from_label(zone, 'FamilyName_t')
    >>> len(PT.get_nodes_from_label(zone, 'FamilyName_t'))
    2
    >>> PT.rm_nodes_from_label(zone, 'FamilyName_t')
    >>> len(PT.get_nodes_from_label(zone, 'FamilyName_t'))
    0

  Note:
    This function admits the following shorcuts: 

    - :func:`rm_nodes_from_name`, :func:`rm_nodes_from_label`,
      :func:`rm_nodes_from_value`, :func:`rm_nodes_from_name_and_label` (embedded predicate)
    - :func:`rm_children_from_name`, :func:`rm_children_from_label`,
      :func:`rm_children_from_value`, :func:`rm_children_from_name_and_label` (embedded predicate + depth=1)
  """
  depth = kwargs.get('depth')
  if depth and not isinstance(depth, int):
    raise TypeError(f"depth must be an integer.")
  if depth and depth >= 1:
    _rm_nodes_from_predicate_with_level__(root, predicate, depth)
  else:
    _rm_nodes_from_predicate__(root, predicate)

def rm_nodes_from_name(root:CGNSTree, name:str, **kwargs):
  """Specialization of rm_nodes_from_predicate with embedded predicate match_name"""
  return rm_nodes_from_predicate(root, lambda n : match_name(n, name), **kwargs)
def rm_nodes_from_label(root:CGNSTree, label:str, **kwargs):
  """Specialization of rm_nodes_from_predicate with embedded predicate match_label"""
  return rm_nodes_from_predicate(root, lambda n : match_label(n, label), **kwargs)
def rm_nodes_from_value(root:CGNSTree, value, **kwargs):
  """Specialization of rm_nodes_from_predicate with embedded predicate match_value"""
  return rm_nodes_from_predicate(root, lambda n : match_value(n, value), **kwargs)
def rm_nodes_from_name_and_label(root:CGNSTree, name:str, label:str, **kwargs):
  """Specialization of rm_nodes_from_predicate with embedded predicate match_name_label"""
  return rm_nodes_from_predicate(root, lambda n : match_name_label(n, name, label), **kwargs)




def rm_node_from_path(root:CGNSTree, path:str):
  """ Remove the node in input tree matching the given path.

  A path is a str containing a full list of names, separated by ``'/'``, leading
  to the node to remove. Root name should not be included in path.
  Wildcards are not accepted in path.

  Args:
    root (CGNSTree): Tree in which the search is performed
    path (str): path of the node to remove
  Example:
    >>> zone = PT.new_Zone('Zone')
    >>> PT.new_FlowSolution('FS', fields={'Density' : [1.], 'Temperature' : [273.]}, parent=zone)
    >>> PT.rm_node_from_path(zone, 'FS/Density')
    >>> PT.print_tree(zone)
    Zone Zone_t 
    ├───ZoneType ZoneType_t "Null"
    └───FS FlowSolution_t 
        └───Temperature DataArray_t R4 [273.]

  See also:
    Also exists as :func:`pop_node_from_path`, which removes the node and returns it.
  """
  pop_node_from_path(root, path)

def pop_node_from_path(root:CGNSTree, path:str) -> CGNSTree:
  if not '/' in path:
    parent = root
    name = path
  else:
    parent = get_node_from_path(root, path_head(path))
    name = path_tail(path)

  node = None
  if parent is not None:
    for i, child in enumerate(parent[2]):
      if child[0] == name:
        node = parent[2].pop(i)
        break
  return node

end_api_export()