import copy as cp

import maia.pytree as PT

def _add_children_to_node_from_another(node1, node2, copy=False):
    for child2 in PT.get_children(node2):
        child1 = PT.get_child_from_name(node1, PT.get_name(child2))
        if child1 is None:
            if copy:
                PT.add_child(node1, cp.deepcopy(child2))
            else:
                PT.add_child(node1, child2)
        else:
            _add_children_to_node_from_another(child1, child2, copy=False)

def _rm_not_common_children(node1, node2):
    child1_to_del = []
    for child1 in PT.get_children(node1):
        child1_name = PT.get_name(child1)
        child2 = PT.get_child_from_name(node2, child1_name)
        if child2 is None:
            child1_to_del.append(child1_name)
    for child_name in child1_to_del:
        PT.rm_node_from_path(node1, child_name)
    for child1 in PT.get_children(node1):
        child2 = PT.get_child_from_name(node2, PT.get_name(child1))
        if child2 is not None:
            _rm_not_common_children(child1, child2)
            
def _have_common_children(node1, node2):
    for child1 in PT.get_children(node1):
        child2 = PT.get_child_from_name(node2, PT.get_name(child1))
        if child2 is None:
            return False
        else:
            if not _have_common_children(child1, child2):
                return False
    return True

def _rm_common_children(node1, node2):
    print("SB")
    child1_to_del = []
    for child1 in PT.get_children(node1):
        child1_name = PT.get_name(child1)
        child2 = PT.get_child_from_name(node2, child1_name)
        if child2 is not None:
            print(child2[0], _have_common_children(child1, child2))
            if _have_common_children(child1, child2):
                child1_to_del.append(child1_name)
    print(child1_to_del)
    for child_name in child1_to_del:
        PT.rm_node_from_path(node1, child_name)
    for child1 in PT.get_children(node1):
        print(child1[0])
        child2 = PT.get_child_from_name(node2, PT.get_name(child1))
        if child2 is not None:
            print("Not None")
            _rm_common_children(child1, child2)

def union(node1, node2, copy=False):
    """
    Return a new node union of node1 and node2
    Remark: node1 and node2 must have the same CGNS label

    Args:
      node1 (CGNSNode): First CGNS node
      node2 (CGNSNode): Second CGNS node
    Returns:
      CGNSNode: union of nodes
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
    if copy:
        union_nodes = cp.deepcopy(node1)
    else:
        union_nodes = node1
    _add_children_to_node_from_another(union_nodes, node2, False)
    return union_nodes

def intersection(node1, node2, copy=False):
    """
    Return a new node intersection of node1 and node2
    Remark: node1 and node2 must have the same CGNS label
    Remark: if the intersection is empty, return only the CGNSTree_t node

    Args:
      node1 (CGNSNode): First CGNS node
      node2 (CGNSNode): Second CGNS node
    Returns:
      CGNSNode: intersection of nodes
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
      │   ├───Zone2 Zone_t
      └───CGNSLibraryVersion CGNSLibraryVersion_t R4 [4.2]
    """
    if PT.get_label(node1) != PT.get_label(node2):
        raise TypeError(f"{PT.get_name(node1)} and {PT.get_name(node2)} have different CGNS labels ({PT.get_label(node1)} vs {PT.get_label(node2)})")
    if copy:
        intersect_nodes = cp.deepcopy(node1)
    else:
        intersect_nodes = node1
    _rm_not_common_children(intersect_nodes, node2)
    return intersect_nodes

def diff(node1, node2, copy=False):
    """
    Return a new node that correspond to node1 without node2's nodes
    Remark: node1 and node2 must have the same CGNS label
    Remark: if the intersection is empty, return only the CGNSTree_t node

    Args:
      node1 (CGNSNode): First CGNS node
      node2 (CGNSNode): Second CGNS node
    Returns:
      CGNSNode: differences between nodes
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
    if copy:
        diff_nodes = cp.deepcopy(node1)
    else:
        diff_nodes = node1
    _rm_common_children(diff_nodes, node2)
    return diff_nodes
