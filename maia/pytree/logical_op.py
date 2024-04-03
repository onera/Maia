import copy

import maia.pytree as PT

def _add_children_to_node_from_another(node1, node2, copy=False):
    for child2 in PT.get_children(node2):
        child1 = PT.get_child_from_name(node1, PT.get_name(child2))
        if child1 is None:
            if copy:
                PT.add_child(node1, copy.deepcopy(child2))
            else:
                PT.add_child(node1, child2)
        else:
            _add_children_to_node_from_another(child1, child2)

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
      ... Base CGNSBase_t:
      ...   Zone1 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone3":
      ...   Zone2 Zone_t:
      ... ''')
      >>> tree2 = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
      ... Base CGNSBase_t:
      ...   Zone2 Zone_t:
      ...   Zone3 Zone_t:
      ...     ZoneGridConnectivity ZoneGridConnectivity_t:
      ...       match GridConnectivity1to1_t "Zone1":
      ... ''')
      >>> PT.union(tree1, tree2)
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
        union_nodes = copy.deepcopy(node1)
    else:
        union_nodes = node1
    _add_children_to_node_from_another(union_nodes, node2, False)
    return union_nodes
    
