import numpy as np

import maia.pytree         as PT
import maia.pytree.compare as PTC

def _add_children_to_node_from_another(node1, node2):
    for child2 in PT.get_children(node2):
        child1 = PT.get_child_from_name(node1, PT.get_name(child2))
        if child1 is None:
            PT.add_child(node1, child2)
        else:
            _add_children_to_node_from_another(child1, child2)

def _rm_not_common_children(node1, node2, comp_func):
    child1_to_del = []
    for child1 in PT.get_children(node1):
        child1_name = PT.get_name(child1)
        child2 = PT.get_child_from_name(node2, child1_name)
        if child2 is None:
            child1_to_del.append(child1_name)
        elif not(comp_func(child1,child2)):
            child1_to_del.append(child1_name)
    for child_name in child1_to_del:
        PT.rm_node_from_path(node1, child_name)
    for child1 in PT.get_children(node1):
        child2 = PT.get_child_from_name(node2, PT.get_name(child1))
        if child2 is not None:
            _rm_not_common_children(child1, child2, comp_func)
            
def _have_common_children(node1, node2, comp_func):
    for child1 in PT.get_children(node1):
        child2 = PT.get_child_from_name(node2, PT.get_name(child1))
        if child2 is None:
            return False
        elif not(comp_func(child1,child2)):
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
    for child_name in child1_to_del:
        PT.rm_node_from_path(node1, child_name)
    for child1 in PT.get_children(node1):
        child2 = PT.get_child_from_name(node2, PT.get_name(child1))
        if child2 is not None:
            _rm_common_children(child1, child2, comp_func)

def union(node1, node2):
    """
    Return a new node union of node1 and node2
    Remark: node1 and node2 must have the same CGNS label
    Remark: if a node from node2 have the same name of a node from node1
            we keep the node from node1

    Args:
      node1 (CGNSTree): First CGNS node
      node2 (CGNSTree): Second CGNS node
    Returns:
      CGNSTree: union of nodes
    Example:
      >>> tree1 = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
      ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
      ... Base CGNSBase_t:
      ...   Zone1 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone3":
      ...   Zone2 Zone_t:
      ... ''')
      >>> tree2 = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
      ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
      ... Base CGNSBase_t:
      ...   Zone2 Zone_t:
      ...   Zone3 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone1":
      ... ''')
      >>> maia.pytree.logical_op.union(tree1, tree2)
      CGNSTree CGNSTree_t
      ├───Base CGNSBase_t
      │   ├───Zone1 Zone_t
      │   │   └───ZoneGridConnectivity ZoneGridConnectivity_t
      │   │       └───match GridConnectivity1to1_t "Zone3"
      │   ├───Zone2 Zone_t
      │   └───Zone3 Zone_t
      │       └───ZoneGridConnectivity ZoneGridConnectivity_t
      │           └───match GridConnectivity1to1_t "Zone1"
      └───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
    """
    if PT.get_label(node1) != PT.get_label(node2):
        raise TypeError(f"{PT.get_name(node1)} and {PT.get_name(node2)} have different CGNS labels ({PT.get_label(node1)} vs {PT.get_label(node2)})")
    union_nodes = PT.shallow_copy(node1)
    _add_children_to_node_from_another(union_nodes, node2)
    return union_nodes

def intersection(node1, node2, comp_func=lambda n1,n2: PTC.is_same_node(n1,n2)):
    """
    Return a new node intersection of node1 and node2
    Remark: node1 and node2 must have the same CGNS label
    Remark: if the intersection is empty, return only the CGNSTree_t node

    Args:
      node1 (CGNSTree): First CGNS node
      node2 (CGNSTree): Second CGNS node
      comp_func (Callable): Fonction to compare node1 to node2
    Returns:
      CGNSTree: intersection of nodes
    Example:
      >>> tree1 = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
      ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
      ... Base CGNSBase_t:
      ...   Zone1 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone3":
      ...   Zone2 Zone_t:
      ... ''')
      >>> tree2 = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
      ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
      ... Base CGNSBase_t:
      ...   Zone2 Zone_t:
      ...   Zone3 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone1":
      ... ''')
      >>> maia.pytree.logical_op.intersection(tree1, tree2)
      CGNSTree CGNSTree_t
      ├───Base CGNSBase_t
      │   └───Zone2 Zone_t
      └───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
    """
    if PT.get_label(node1) != PT.get_label(node2):
        raise TypeError(f"{PT.get_name(node1)} and {PT.get_name(node2)} have different CGNS labels ({PT.get_label(node1)} vs {PT.get_label(node2)})")
    intersect_nodes = PT.shallow_copy(node1)
    _rm_not_common_children(intersect_nodes, node2, comp_func)
    return intersect_nodes

def diff(node1, node2, comp_func=lambda n1,n2: PTC.is_same_node(n1,n2)):
    """
    Return a new node that correspond to node1 without node2's nodes
    Remark: node1 and node2 must have the same CGNS label
    Remark: if the intersection is empty, return only the CGNSTree_t node

    Args:
      node1 (CGNSTree): First CGNS node
      node2 (CGNSTree): Second CGNS node
      comp_func (Callable): Fonction to compare node1 to node2
    Returns:
      CGNSTree: differences between nodes
    Example:
      >>> tree1 = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
      ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
      ... Base CGNSBase_t:
      ...   Zone1 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone3":
      ...   Zone2 Zone_t:
      ... ''')
      >>> tree2 = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
      ... CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]:
      ... Base CGNSBase_t:
      ...   Zone2 Zone_t:
      ...   Zone3 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone1":
      ... ''')
      >>> maia.pytree.logical_op.intersection(tree1, tree2)
      CGNSTree CGNSTree_t
      └───Base CGNSBase_t
          └───Zone1 Zone_t
              └───ZoneGridConnectivity ZoneGridConnectivity_t
                  └───match GridConnectivity1to1_t "Zone3":
    """
    if PT.get_label(node1) != PT.get_label(node2):
        raise TypeError(f"{PT.get_name(node1)} and {PT.get_name(node2)} have different CGNS labels ({PT.get_label(node1)} vs {PT.get_label(node2)})")
    diff_nodes = PT.shallow_copy(node1)
    _rm_common_children(diff_nodes, node2, comp_func)
    return diff_nodes
