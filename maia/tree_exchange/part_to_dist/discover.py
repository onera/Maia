import Converter.Internal as I

from maia.utils import py_utils

def discover_partitioned_zones(part_tree, comm):
  """
  Recreate the list of distributed zone paths (baseName/zoneName) from
  the partitioned tress.
  """
  part_pathes = []
  dist_pathes = []
  for part_base, part_zone in py_utils.getNodesWithParentsFromTypePath(part_tree, 'CGNSBase_t/Zone_t'):
    part_path = I.getName(part_base) + '/' + '.'.join(I.getName(part_zone).split('.')[:-2])
    if not part_path in part_pathes:
      part_pathes.append(part_path)
  for rank_part_pathes in comm.allgather(part_pathes):
    for rank_dist_path in rank_part_pathes:
      if not rank_dist_path in dist_pathes:
        dist_pathes.append(rank_dist_path)

  return dist_pathes

def discover_nodes_of_kind(dist_node, part_nodes, kind_path, comm,
    child_list=[], allow_multiple=False, skip_rule=lambda node:False):
  """
  Recreate a distributed structure (basically without data) in dist_node merging all the
  path found in (locally known) part_nodes.
  Usefull eg to globally reput on a dist_zone some BC created on specific part_zones.
  Nodes already present in dist_node will not be added.
  dist_node and part_nodes are the starting point of the search to which kind_path is related
  Additional options:
    child_list is a list of node names or types related to leaf nodes that will be copied into dist_node
    allow_multiple, when setted true, consider leaf node of name name.N as related to same dist leaf
    skip_rule accepts a bool function whose argument is a node. If this function returns True, the node
      will not be added in dist_tree.
  Todo : could be optimised using a distributed hash table -> see BM
  """
  collected_part_nodes = dict()
  for part_node in part_nodes:
    for nodes in py_utils.getNodesWithParentsFromTypePath(part_node, kind_path):
      leaf_path = '/'.join([I.getName(node) for node in nodes])
      # Options to map splitted nodes (eg jn) to the same dist node
      if allow_multiple:
        leaf_path = '.'.join(leaf_path.split('.')[:-1])
      # Option to skip some nodes
      if skip_rule(nodes[-1]):
        continue
      # Avoid data duplication to minimize exchange
      if I.getNodeFromPath(dist_node, leaf_path) is None and leaf_path not in collected_part_nodes:
        ancestors, leaf = nodes[:-1], nodes[-1]
        type_list  = [I.getType(node)  for node in nodes]
        value_list = [I.getValue(node) for node in nodes]
        childs = list()
        for query in child_list:
          getNodes1 = I.getNodesFromType1 if query[-2:] == '_t' else I.getNodesFromName1
          childs.extend(getNodes1(leaf, query))
        collected_part_nodes[leaf_path] = (type_list, value_list, childs)

  for rank_node_path in comm.allgather(collected_part_nodes):
    for node_path, (type_list, value_list, childs) in rank_node_path.items():
      if I.getNodeFromPath(dist_node, node_path) is None:
        nodes_name = node_path.split('/')
        ancestor = dist_node
        for name, kind, value in zip(nodes_name, type_list, value_list):
          ancestor = I.createUniqueChild(ancestor, name, kind, value)
        # At the end of this loop, ancestor is in fact the leaf node
        for child in childs:
          I._addChild(ancestor, child)

