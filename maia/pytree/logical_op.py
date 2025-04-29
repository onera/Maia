from maia.pytree.typing import *

import maia.pytree as PT

__all__ = ['union', 'intersection', 'difference']

def _add_children_to_node_from_another(node1, node2):
  for child2 in node2[2]:
    try:
      child1 = next(c for c in node1[2] if c[0] == child2[0])
      _add_children_to_node_from_another(child1, child2)
    except StopIteration:
      PT.add_child(node1, PT.shallow_copy(child2))

def _rm_not_common_children(node1, node2, comp_func):
  child1_to_del = []
  for child1 in PT.get_children(node1):
    child1_name = PT.get_name(child1)
    child2 = PT.get_child_from_name(node2, child1_name)
    if child2 is None or not comp_func(child1, child2):
      child1_to_del.append(child1_name)
  PT.rm_children_from_predicate(node1, lambda n : PT.get_name(n) in child1_to_del)

  for child1 in PT.get_children(node1):
    child2 = PT.get_child_from_name(node2, PT.get_name(child1))
    _rm_not_common_children(child1, child2, comp_func)
            
def _have_common_children(node1, node2, comp_func):
  for child1 in PT.get_children(node1):
    child2 = PT.get_child_from_name(node2, PT.get_name(child1))
    if child2 is None:
      return False
    elif not comp_func(child1,child2):
      return False
    else:
      if not _have_common_children(child1, child2, comp_func):
        return False
  return True

def _rm_common_children(node1, node2, comp_func):
  child1_to_del = []
  for child1 in PT.get_children(node1):
    child1_name = PT.get_name(child1)
    child2 = PT.get_child_from_name(node2, child1_name)
    if child2 is not None:
      if _have_common_children(child1, child2, comp_func) and comp_func(child1,child2):
        child1_to_del.append(child1_name)
  PT.rm_children_from_predicate(node1, lambda n : PT.get_name(n) in child1_to_del)

  for child1 in PT.get_children(node1):
    child2 = PT.get_child_from_name(node2, PT.get_name(child1))
    if child2 is not None:
      _rm_common_children(child1, child2, comp_func)

#begin_api_export()

def union(*trees:CGNSTree) -> CGNSTree:
  """ Create a new tree from the union of the input trees.

  Nodes existing on more than one input trees keep the value and label of their first appearance.
  Note also that output values are shared references to input trees.
  Uses :func:`deep_copy` afterwards if you want an independant copy.

  Important: 
    Input root nodes must have the same name. An exception will be raised otherwise.

  Args:
    trees (CGNSTree): Input trees
  Returns:
    CGNSTree: Tree created from union
  Example:
    >>> tree1 = PT.yaml.to_node('''
    ... Base CGNSBase_t:
    ...   Zone1 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity1to1_t "Zone3":
    ...   Zone2 Zone_t:
    ... ''')
    >>> tree2 = PT.yaml.to_node('''
    ... Base CGNSBase_t:
    ...   Zone2 Zone_t:
    ...   Zone3 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity1to1_t "Zone1":
    ... ''')
    >>> tree = PT.union(tree1, tree2)
    >>> PT.print_tree(tree)
    Base CGNSBase_t 
    ├───Zone1 Zone_t 
    │   └───ZoneGridConnectivity ZoneGridConnectivity_t 
    │       └───match GridConnectivity1to1_t "Zone3"
    ├───Zone2 Zone_t 
    └───Zone3 Zone_t 
        └───ZoneGridConnectivity ZoneGridConnectivity_t 
            └───match GridConnectivity1to1_t "Zone1"
  """
  assert len(trees) > 0
  in_names = [PT.get_name(n) for n in trees]
  if len(set(in_names)) != 1:
    raise ValueError(f"Mismatching names for input nodes : {in_names}")
  union_nodes = PT.shallow_copy(trees[0])
  for t2 in trees[1:]:
    _add_children_to_node_from_another(union_nodes, t2)
  return union_nodes

def intersection(*trees:CGNSTree,
                 comp_func:Callable[[CGNSTree, CGNSTree],bool]=PT.is_same_node) -> CGNSTree:
  """ Create a new tree from the intersection of the input trees.

  At each tree level, input nodes are considered equal if the
  binary predicate ``comp_func(n1, n2)`` returns ``True``.
  If not provided, the function :func:`is_same_node` is used.

  If the intersection is empty, result contains only the root node.
  Note also that output values are shared references to first input tree.
  Uses :func:`deep_copy` afterwards if you want an independant copy.

  Important: 
    Input root nodes must have the same name. An exception will be raised otherwise.

  Args:
    trees (CGNSTree): Input trees
    comp_func (Callable): Binary predicate used for comparison (see above)
  Returns:
    CGNSTree: Tree created from intersection
  Example:
    >>> tree1 = PT.yaml.to_cgns_tree('''
    ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
    ... Base CGNSBase_t:
    ...   Zone1 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity1to1_t "Zone3":
    ...   Zone2 Zone_t:
    ... ''')
    >>> tree2 = PT.yaml.to_cgns_tree('''
    ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
    ... Base CGNSBase_t:
    ...   Zone2 Zone_t:
    ...   Zone3 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity1to1_t "Zone1":
    ... ''')
    >>> tree = PT.intersection(tree1, tree2)
    >>> PT.print_tree(tree)
    CGNSTree CGNSTree_t 
    ├───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
    └───Base CGNSBase_t 
        └───Zone2 Zone_t 
  """
  assert len(trees) > 0
  in_names = [PT.get_name(n) for n in trees]
  if len(set(in_names)) != 1:
    raise ValueError(f"Mismatching names for input nodes : {in_names}")
  intersect_nodes = PT.shallow_copy(trees[0])
  for t2 in trees[1:]:
    _rm_not_common_children(intersect_nodes, t2, comp_func)
  return intersect_nodes

def difference(t1:CGNSTree, t2:CGNSTree,
               comp_func:Callable[[CGNSTree, CGNSTree],bool]=PT.is_same_node) -> CGNSTree:
  """ Create a new tree from the difference of the input trees.
  
  At each tree level, input nodes are considered equal if the
  binary predicate ``comp_func(n1, n2)`` returns ``True``.
  If not provided, the function :func:`is_same_node` is used.

  This operation is not symmetric. Ouput contains nodes belonging *only*
  to first input.
  Note also that output values are shared references to input tree ``t1``.
  Uses :func:`deep_copy` afterwards if you want an independant copy.

  Important: 
    Input root nodes must have the same name. An exception will be raised otherwise.

  Args:
    t1 (CGNSTree): First CGNS node
    t2 (CGNSTree): Second CGNS node
    comp_func (Callable): Binary predicate used for comparison (see above)
  Returns:
    CGNSTree: Tree created from difference
  Example:
    >>> tree1 = PT.yaml.to_cgns_tree('''
    ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
    ... Base CGNSBase_t:
    ...   Zone1 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity1to1_t "Zone3":
    ...   Zone2 Zone_t:
    ... ''')
    >>> tree2 = PT.yaml.to_cgns_tree('''
    ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
    ... Base CGNSBase_t:
    ...   Zone2 Zone_t:
    ...   Zone3 Zone_t:
    ...     ZoneGridConnectivity ZoneGridConnectivity_t:
    ...       match GridConnectivity1to1_t "Zone1":
    ... ''')
    >>> tree = PT.difference(tree1, tree2)
    >>> PT.print_tree(tree)
    CGNSTree CGNSTree_t 
    └───Base CGNSBase_t 
        └───Zone1 Zone_t 
            └───ZoneGridConnectivity ZoneGridConnectivity_t 
                └───match GridConnectivity1to1_t "Zone3"
  """
  in_names = [PT.get_name(n) for n in [t1, t2]]
  if len(set(in_names)) != 1:
    raise ValueError(f"Mismatching names for input nodes : {in_names}")
  diff_nodes = PT.shallow_copy(t1)
  _rm_common_children(diff_nodes, t2, comp_func)
  return diff_nodes

#end_api_export()