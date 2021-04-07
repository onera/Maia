import numpy      as np

import Converter.Internal as I
import Pypdm.Pypdm as PDM

from maia.utils import py_utils
import maia.tree_exchange.utils as te_utils
from maia.tree_exchange.part_to_dist import data_exchange  as PTB
from maia.tree_exchange.part_to_dist import index_exchange as IPTB

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


def match_jn_from_ordinals(dist_tree):
  """
  Retrieve for each original GridConnectivity_t node the donor zone and the opposite
  pointlist in the tree. This assume that index distribution was identical for two related
  gc nodes.
  For now, donorname does not include basename because of a solver failure.
  """
  ordinal_to_data = dict() #Will store zone name & PL for each GC
  gc_t_path = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity_t'

  for base,zone,zgc,gc in py_utils.getNodesWithParentsFromTypePath(dist_tree, gc_t_path):
    ordinal = I.getNodeFromName1(gc, 'Ordinal')[1][0]
    #ordinal_to_zname[ordinal] = I.getName(base) + '/' + I.getName(zone)
    ordinal_to_data[ordinal] = (I.getName(zone), I.getNodeFromName1(gc, 'PointList')[1])

  for base,zone,zgc,gc in py_utils.getNodesWithParentsFromTypePath(dist_tree, gc_t_path):
    ordinal_opp = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]
    donor_name, donor_pl = ordinal_to_data[ordinal_opp]
    I.setValue(gc, donor_name)
    I.newPointList('PointListDonor', donor_pl, parent=gc)

def disttree_from_parttree(part_tree, comm):
  """
  Regenerate a distributed tree from partitioned trees.
  Partitioned trees must include all GlobalNumbering data.
  For now only NGon/NFace trees are supported
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  # > Discover partitioned zones to build dist_tree structure
  dist_zone_pathes = PTB.discover_partitioned_zones(part_tree, comm)

  dist_tree = I.newCGNSTree()
  for dist_zone_path in dist_zone_pathes:
    base_name, zone_name = dist_zone_path.split('/')

    # > Create base and zone in dist tree
    dist_base = I.createUniqueChild(dist_tree, base_name, 'CGNSBase_t', [3,3])
    dist_zone = I.newZone(zone_name, zsize=[[0,0,0]], ztype='Unstructured', parent=dist_base)
    distri_ud = I.createUniqueChild(dist_zone, ':CGNS#Distribution', 'UserDefinedData_t')

    part_zones = te_utils.get_partitioned_zones(part_tree, dist_zone_path)

    # > Create vertex distribution and exchange vertex coordinates
    vtx_lngn_list = te_utils.collect_cgns_g_numbering(part_zones, ':CGNS#GlobalNumbering/Vertex')
    pdm_ptb = PDM.PartToBlock(comm, vtx_lngn_list, pWeight=None, partN=len(vtx_lngn_list),
                              t_distrib=0, t_post=1, t_stride=0)
    vtx_distri = pdm_ptb.getDistributionCopy()
    I.newDataArray('Vertex', vtx_distri[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
    dist_zone[1][0][0] = vtx_distri[2]
    d_grid_co = I.newGridCoordinates('GridCoordinates', parent=dist_zone)
    for coord in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
      I.newDataArray(coord, parent=d_grid_co)
    PTB.part_coords_to_dist_coords(dist_zone, part_zones, comm)

    # > Create ngon and nface connectivities
    IPTB.part_ngon_to_dist_ngon(dist_zone, part_zones, 'NGonElements', comm)
    IPTB.part_nface_to_dist_nface(dist_zone, part_zones, 'NFaceElements', 'NGonElements', comm)

    # > Shift nface element_range and create all cell distri
    n_face_tot  = I.getNodeFromPath(dist_zone, 'NGonElements/ElementRange')[1][1]
    nface_range = I.getNodeFromPath(dist_zone, 'NFaceElements/ElementRange')[1]
    nface_range += n_face_tot

    cell_distri = I.getNodeFromPath(dist_zone, 'NFaceElements/:CGNS#Distribution/Element')[1]
    I.newDataArray('Cell', cell_distri, parent=distri_ud)
    dist_zone[1][0][1] = cell_distri[2]

    # > BND and JNS
    bc_t_path = 'ZoneBC_t/BC_t'
    gc_t_path = 'ZoneGridConnectivity_t/GridConnectivity_t'

    # > Discover
    discover_nodes_of_kind(dist_zone, part_zones, bc_t_path, comm,
        child_list=['FamilyName_t', 'GridLocation_t'])
    discover_nodes_of_kind(dist_zone, part_zones, gc_t_path, comm,
        child_list=['GridLocation_t', 'GridConnectivityProperty_t', 'Ordinal', 'OrdinalOpp'],
        allow_multiple=True,
        skip_rule = lambda node: I.getNodeFromPath(node, ':CGNS#GlobalNumbering') is None)

    # > Index exchange
    for d_zbc, d_bc in py_utils.getNodesWithParentsFromTypePath(dist_zone, 'ZoneBC_t/BC_t'):
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, I.getName(d_zbc) + '/' + I.getName(d_bc), comm)
    for d_zgc, d_gc in py_utils.getNodesWithParentsFromTypePath(dist_zone, gc_t_path):
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, I.getName(d_zgc) + '/' + I.getName(d_gc), comm, True)

    # > Flow Solution and Discrete Data
    PTB.part_sol_to_dist_sol(dist_zone, part_zones, comm)
    
    # > Todo : BCDataSet

  match_jn_from_ordinals(dist_tree)

  return dist_tree

