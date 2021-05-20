import Converter.Internal as I
import maia.sids.cgns_keywords as CGK
import maia.sids.Internal_ext  as IE


def discover_nodes_from_matching(dist_node, part_nodes, queries, comm,
                                 child_list=[], get_value="ancestors",
                                 merge_rule=lambda path:path):
  """
  Recreate a distributed structure (basically without data) in dist_node merging all the
  path found in (locally known) part_nodes.
  Usefull eg to globally reput on a dist_zone some BC created on specific part_zones.
  Nodes already present in dist_node will not be added.
  dist_node and part_nodes are the starting point of the search to which queries is related
  Additional options:
    child_list is a list of node names or types related to leaf nodes that will be copied into dist_node
    get_value (string) indicates which nodes of the path must repport their value to the dist node
      (none = nothing, all = all nodes of queries, anything else = all but last nodes of queries)
    merge_rule accepts a function whose argument is the leaf node path. This function can map the path to an
      other, eg to merge splitted node related to a same dist node
  Todo : could be optimised using a distributed hash table -> see BM
  """
  collected_part_nodes = dict()
  for part_node in part_nodes:
    for nodes in IE.getNodesWithParentsByMatching(part_node, queries):
      # Apply merge rule to map splitted nodes (eg jn) to the same dist node
      leaf_path = merge_rule('/'.join([I.getName(node) for node in nodes]))
      # Avoid data duplication to minimize exchange
      if I.getNodeFromPath(dist_node, leaf_path) is None and leaf_path not in collected_part_nodes:
        # Label
        labels = [I.getType(node) for node in nodes]
        # Values
        ancestors, leaf = nodes[:-1], nodes[-1]
        if get_value == "none":
          values = len(nodes)*[None]
        elif get_value == "all":
          values = [I.getValue(node) for node in nodes]
        else:
          values = [I.getValue(node) for node in ancestors] + [None]
        # Children
        childs = list()
        for query in child_list:
          node = IE.getNodesDispatch1(leaf, query)
          childs.extend(node)
        collected_part_nodes[leaf_path] = (labels, values, childs)

  for rank_node_path in comm.allgather(collected_part_nodes):
    for node_path, (labels, values, childs) in rank_node_path.items():
      if I.getNodeFromPath(dist_node, node_path) is None:
        nodes_name = node_path.split('/')
        ancestor = dist_node
        for name, label, value in zip(nodes_name, labels, values):
          ancestor = I.createUniqueChild(ancestor, name, label, value)
        # At the end of this loop, ancestor is in fact the leaf node
        for child in childs:
          I._addChild(ancestor, child)

