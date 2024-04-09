from maia.pytree.typing import *

# Keys to access CGNSTree values
__NAME__     = 0
__VALUE__    = 1
__CHILDREN__ = 2
__LABEL__    = 3

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

