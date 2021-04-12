import Converter.Internal as I

from maia.utils import py_utils

def discover_nodes_of_kind(dist_node, part_nodes, kind_path, comm,
    child_list=[], get_value="ancestors", merge_rule=lambda path:path, skip_rule=lambda node:False):
  """
  Recreate a distributed structure (basically without data) in dist_node merging all the
  path found in (locally known) part_nodes.
  Usefull eg to globally reput on a dist_zone some BC created on specific part_zones.
  Nodes already present in dist_node will not be added.
  dist_node and part_nodes are the starting point of the search to which kind_path is related
  Additional options:
    child_list is a list of node names or types related to leaf nodes that will be copied into dist_node
    get_value (string) indicates which nodes of the path must repport their value to the dist node
      (none = nothing, all = all nodes of kind_path, anything else = all but last nodes of kind_path)
    merge_rule accepts a function whose argument is the leaf node path. This function can map the path to an
      other, eg to merge splitted node related to a same dist node
    skip_rule accepts a bool function whose argument is a node. If this function returns True, the node
      will not be added in dist_tree.
  Todo : could be optimised using a distributed hash table -> see BM
  """
  collected_part_nodes = dict()
  for part_node in part_nodes:
    for nodes in py_utils.getNodesWithParentsFromTypePath(part_node, kind_path):
      # Option to skip some nodes
      if skip_rule(nodes[-1]):
        continue
      # Apply merge rule to map splitted nodes (eg jn) to the same dist node
      leaf_path = merge_rule('/'.join([I.getName(node) for node in nodes]))
      # Avoid data duplication to minimize exchange
      if I.getNodeFromPath(dist_node, leaf_path) is None and leaf_path not in collected_part_nodes:
        ancestors, leaf = nodes[:-1], nodes[-1]
        if get_value == "none":
          value_list = len(nodes)*[None]
        elif get_value == "all":
          value_list = [I.getValue(node) for node in nodes]
        else:
          value_list = [I.getValue(node) for node in ancestors] + [None]
        childs = list()
        for query in child_list:
          getNodes1 = I.getNodesFromType1 if query[-2:] == '_t' else I.getNodesFromName1
          childs.extend(getNodes1(leaf, query))
        collected_part_nodes[leaf_path] = (value_list, childs)

  for rank_node_path in comm.allgather(collected_part_nodes):
    for node_path, (value_list, childs) in rank_node_path.items():
      if I.getNodeFromPath(dist_node, node_path) is None:
        nodes_name = node_path.split('/')
        ancestor = dist_node
        for name, kind, value in zip(nodes_name, kind_path.split('/'), value_list):
          ancestor = I.createUniqueChild(ancestor, name, kind, value)
        # At the end of this loop, ancestor is in fact the leaf node
        for child in childs:
          I._addChild(ancestor, child)

