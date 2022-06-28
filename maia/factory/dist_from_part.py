import numpy      as np
import Pypdm.Pypdm as PDM

import Converter.Internal as I
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.algo.dist             import matching_jns_tools as MJT
from maia.transfer              import utils              as tr_utils
from maia.transfer.part_to_dist import data_exchange      as PTB
from maia.transfer.part_to_dist import index_exchange     as IPTB
from maia.utils                 import py_utils

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
    get_value is a list of nodes of the path whose values must be repported to the dist node
      get_value can be a list of bool or one of the shortcuts 'all', 'none', 'ancestors' (=all but
      last), 'leaf' (only last)
    merge_rule accepts a function whose argument is the leaf node path. This function can map the path to an
      other, eg to merge splitted node related to a same dist node
  Todo : could be optimised using a distributed hash table -> see BM
  """
  collected_part_nodes = dict()
  for part_node in part_nodes:
    for nodes in PT.iter_children_from_predicates(part_node, queries, ancestors=True):
      # Apply merge rule to map splitted nodes (eg jn) to the same dist node
      leaf_path = merge_rule('/'.join([I.getName(node) for node in nodes]))
      # Avoid data duplication to minimize exchange
      if I.getNodeFromPath(dist_node, leaf_path) is None and leaf_path not in collected_part_nodes:
        # Label
        labels = [I.getType(node) for node in nodes]

        # Values
        if isinstance(get_value, str):
          get_value = py_utils.str_to_bools(len(nodes), get_value)
        if isinstance(get_value, (tuple, list)):
          values = [I.getValue(node) if value else None for node, value in zip(nodes, get_value)]

        # Children
        leaf = nodes[-1]
        childs = list()
        for query in child_list:
          # Convert to a list of size 1 to use get_children_from_predicates, who works on a predicate-like list 
          childs.extend(PT.get_children_from_predicates(leaf, [query]))
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

def recover_dist_tree(part_tree, comm):
  """
  Regenerate a distributed tree from partitioned trees.
  Partitioned trees must include all GlobalNumbering data.
  For now only NGon/NFace trees are supported
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  dist_tree = I.newCGNSTree()
  # > Discover partitioned zones to build dist_tree structure
  discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t', comm, child_list=['Family_t'])
  discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t/Zone_t', comm,\
                               child_list = ['ZoneType_t'],
                               merge_rule=lambda zpath : MT.conv.get_part_prefix(zpath))

  for dist_zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    dist_zone = I.getNodeFromPath(dist_tree, dist_zone_path)

    distri_ud = MT.newDistribution(parent=dist_zone)

    part_zones = tr_utils.get_partitioned_zones(part_tree, dist_zone_path)

    # > Create vertex distribution and exchange vertex coordinates
    vtx_lngn_list = tr_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
    pdm_ptb = PDM.PartToBlock(comm, vtx_lngn_list, pWeight=None, partN=len(vtx_lngn_list),
                              t_distrib=0, t_post=1)
    vtx_distri_pdm = pdm_ptb.getDistributionCopy()
    I.newDataArray('Vertex', vtx_distri_pdm[[i_rank, i_rank+1, n_rank]], parent=distri_ud)
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

    cell_distri = I.getVal(MT.getDistribution(I.getNodeFromName(dist_zone, 'NFaceElements'), 'Element'))
    vtx_distri  = I.getVal(MT.getDistribution(dist_zone, 'Vertex'))
    I.newDataArray('Cell', cell_distri, parent=distri_ud)

    I.setValue(dist_zone, np.array([[vtx_distri[2], cell_distri[2], 0]], dtype=np.int32))

    # > BND and JNS
    bc_t_path = 'ZoneBC_t/BC_t'
    gc_t_path = ['ZoneGridConnectivity_t', lambda n: I.getType(n) == 'GridConnectivity_t' and not MT.conv.is_intra_gc(I.getName(n))]

    # > Discover (skip GC created by partitioning)
    discover_nodes_from_matching(dist_zone, part_zones, bc_t_path, comm,
          child_list=['FamilyName_t', 'GridLocation_t'], get_value='all')
    # discover_nodes_from_matching(dist_zone, part_zones, gc_t_path, comm,
    #       child_list=['GridLocation_t', 'GridConnectivityProperty_t', 'GridConnectivityDonorName'],
    #       merge_rule= lambda path: MT.conv.get_split_prefix(path),
    #       skip_rule = lambda node: MT.conv.is_intra_gc(I.getName(node)))
    discover_nodes_from_matching(dist_zone, part_zones, gc_t_path, comm,
          child_list=['GridLocation_t', 'GridConnectivityProperty_t', 'GridConnectivityDonorName'],
          merge_rule= lambda path: MT.conv.get_split_prefix(path), get_value='leaf')
    #After GC discovery, cleanup donor name suffix
    for jn in PT.iter_children_from_predicates(dist_zone, gc_t_path):
      val = I.getValue(jn)
      I.setValue(jn, MT.conv.get_part_prefix(val))

    # > Index exchange
    for d_zbc, d_bc in PT.iter_children_from_predicates(dist_zone, bc_t_path, ancestors=True):
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, I.getName(d_zbc) + '/' + I.getName(d_bc), comm)
    for d_zgc, d_gc in PT.iter_children_from_predicates(dist_zone, gc_t_path, ancestors=True):
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, I.getName(d_zgc) + '/' + I.getName(d_gc), comm, True)

    # > Flow Solution and Discrete Data
    PTB.part_sol_to_dist_sol(dist_zone, part_zones, comm)

    # > Todo : BCDataSet

  MJT.copy_donor_subset(dist_tree)

  return dist_tree
