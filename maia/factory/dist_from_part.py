from mpi4py import MPI
import numpy      as np
import operator
import Pypdm.Pypdm as PDM

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.algo.dist             import matching_jns_tools as MJT
from maia.algo.dist.s_to_u      import guess_bnd_normal_index
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
      leaf_path = merge_rule('/'.join([PT.get_name(node) for node in nodes]))
      # Avoid data duplication to minimize exchange
      if PT.get_node_from_path(dist_node, leaf_path) is None and leaf_path not in collected_part_nodes:
        # Label
        labels = [PT.get_label(node) for node in nodes]

        # Values
        if isinstance(get_value, str):
          get_value = py_utils.str_to_bools(len(nodes), get_value)
        if isinstance(get_value, (tuple, list)):
          values = [PT.get_value(node) if value else None for node, value in zip(nodes, get_value)]

        # Children
        leaf = nodes[-1]
        childs = list()
        for query in child_list:
          # Convert to a list of size 1 to use get_children_from_predicates, who works on a predicate-like list 
          childs.extend(PT.get_children_from_predicates(leaf, [query]))
        collected_part_nodes[leaf_path] = (labels, values, childs)

  for rank_node_path in comm.allgather(collected_part_nodes):
    for node_path, (labels, values, childs) in rank_node_path.items():
      if PT.get_node_from_path(dist_node, node_path) is None:
        nodes_name = node_path.split('/')
        ancestor = dist_node
        for name, label, value in zip(nodes_name, labels, values):
          ancestor = PT.update_child(ancestor, name, label, value)
        # At the end of this loop, ancestor is in fact the leaf node
        for child in childs:
          PT.add_child(ancestor, child)

def get_parts_per_blocks(part_tree, comm):
  """
  From the partitioned trees, retrieve the paths of the distributed blocks
  and return a dictionnary associating each path to the list of the corresponding
  partitioned zones
  """
  dist_doms = PT.new_CGNSTree()
  discover_nodes_from_matching(dist_doms, [part_tree], 'CGNSBase_t/Zone_t', comm,
                                    merge_rule=lambda zpath : MT.conv.get_part_prefix(zpath))
  parts_per_dom = dict()
  for zone_path in PT.predicates_to_paths(dist_doms, 'CGNSBase_t/Zone_t'):
    parts_per_dom[zone_path] = tr_utils.get_partitioned_zones(part_tree, zone_path)
  return parts_per_dom

def _recover_dist_block_size(part_zones, comm):
  """ From a list of partitioned zones (coming from same initial block),
  retrieve the size of the initial block """
  intra1to1 = lambda n: PT.get_label(n) == 'GridConnectivity1to1_t' and MT.conv.is_intra_gc(PT.get_name(n))

  # Collect zone size and pr+opposite zone thought partitioning jns
  zones_to_size = {}
  zones_to_join = {}
  for part_zone in part_zones:
    zone_name = PT.get_name(part_zone)
    zones_to_size[zone_name] = PT.Zone.CellSize(part_zone)
    zones_to_join[zone_name] = []
    for intra_jn in PT.iter_children_from_predicates(part_zone, ['ZoneGridConnectivity_t', intra1to1]):
      zones_to_join[zone_name].append((PT.Subset.getPatch(intra_jn)[1], PT.get_value(intra_jn)))

  # Gather and flatten dicts
  zones_to_size_g = {}
  zones_to_join_g = {}
  for zones_to_size_rank in comm.allgather(zones_to_size):
    zones_to_size_g.update(zones_to_size_rank)
  for zones_to_join_rank in comm.allgather(zones_to_join):
    zones_to_join_g.update(zones_to_join_rank)

  # Choose any starting point
  first = next(iter(zones_to_size_g))
  d_zone_dims = np.zeros((3,3), np.int32, order='F')
  d_zone_dims[:,1] += zones_to_size_g[first] #Cell size
  for axis in range(3):
    for oper in [operator.ne, operator.eq]: #Go front (vtx != 1), then back (vtx == 1)
      # Reset
      keep_going = True
      current = first
      while keep_going:
        # Iterate jns and select one to continue in same axis/direction
        for (pr,opposite) in zones_to_join_g[current]:
          if guess_bnd_normal_index(pr, 'Vertex') == axis and oper(pr[axis,0], 1):
            current = opposite
            d_zone_dims[axis,1] += zones_to_size_g[current][axis]
            break
        else: #If loop did not break -> we reached the end of block
          keep_going = False
  d_zone_dims[:,0] = d_zone_dims[:,1] + 1 # Update vertices
  return d_zone_dims

def _recover_elements(dist_zone, part_zones, comm):
  # > Get the list of part elements
  fake_zone = PT.new_node('Zone', 'Zone_t') #This is just to store the elements
  discover_nodes_from_matching(fake_zone, part_zones, 'Elements_t', comm, get_value='leaf')
  elt_names = [PT.get_name(elt) for elt in PT.get_children(fake_zone)]
  elt_kinds = [PT.Element.CGNSName(elt) for elt in PT.get_children(fake_zone)]
  has_ngon  = 'NGON_n'  in elt_kinds
  has_nface = 'NFACE_n' in elt_kinds

  # Deal NGon/NFace
  if has_ngon:
    assert all([kind in ['NGON_n', 'NFACE_n'] for kind in elt_kinds])
    ngon_name = elt_names[elt_kinds.index('NGON_n')]
    IPTB.part_ngon_to_dist_ngon(dist_zone, part_zones, ngon_name, comm)
    if has_nface:
      nface_name = elt_names[elt_kinds.index('NFACE_n')]
      IPTB.part_nface_to_dist_nface(dist_zone, part_zones, nface_name, ngon_name, comm)
      # > Shift nface element_range and create all cell distri
      n_face_tot  = PT.get_node_from_path(dist_zone, 'NGonElements/ElementRange')[1][1]
      nface_range = PT.get_node_from_path(dist_zone, 'NFaceElements/ElementRange')[1]
      nface_range += n_face_tot

  # Deal standard elements
  else:
    for elt_name in elt_names:
      IPTB.part_elt_to_dist_elt(dist_zone, part_zones, elt_name, comm)

    elt_nodes = PT.get_children_from_label(dist_zone, 'Elements_t') #True elements
    # > Get shift per dim
    n_elt_per_dim  = [0,0,0,0]
    for elt in elt_nodes:
      n_elt_per_dim[PT.Element.Dimension(elt)] += PT.Element.Size(elt)

    elt_order = [PT.Zone.elt_ordering_by_dim(part_zone) for part_zone in part_zones]
    n_increase = comm.allreduce(elt_order.count(1),  MPI.SUM)
    n_decrease = comm.allreduce(elt_order.count(-1), MPI.SUM)
    assert n_increase * n_decrease == 0
    
    if n_increase > 0:
      for elt in elt_nodes:
        dim_shift = sum(n_elt_per_dim[:PT.Element.Dimension(elt)])
        ER = PT.get_child_from_name(elt, 'ElementRange')
        ER[1] += dim_shift

    else:
      for elt in elt_nodes:
        dim_shift = sum(n_elt_per_dim[PT.Element.Dimension(elt)+1:])
        ER = PT.get_child_from_name(elt, 'ElementRange')
        ER[1] += dim_shift

def recover_dist_tree(part_tree, comm):
  """ Regenerate a distributed tree from a partitioned tree.
  
  The partitioned tree should have been created using Maia, or
  must at least contains GlobalNumbering nodes as defined by Maia
  (see :ref:`part_tree`).

  The following nodes are managed : GridCoordinates, Elements, ZoneBC, ZoneGridConnectivity
  FlowSolution, DiscreteData and ZoneSubRegion.

  Args:
    part_tree (CGNSTree) : Partitioned CGNS Tree
    comm       (MPIComm) : MPI communicator
  Returns:
    CGNSTree: distributed cgns tree

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #recover_dist_tree@start
        :end-before: #recover_dist_tree@end
        :dedent: 2
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  dist_tree = PT.new_CGNSTree()
  # > Discover partitioned zones to build dist_tree structure
  discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t', comm, child_list=['Family_t'])
  discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t/Zone_t', comm,\
                               child_list = ['ZoneType_t', 'FamilyName_t', 'AdditionalFamilyName_t'],
                               merge_rule=lambda zpath : MT.conv.get_part_prefix(zpath))

  for dist_zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    dist_zone = PT.get_node_from_path(dist_tree, dist_zone_path)

    part_zones = tr_utils.get_partitioned_zones(part_tree, dist_zone_path)

    # Create zone distributions
    vtx_lngn_list  = tr_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
    cell_lngn_list = tr_utils.collect_cgns_g_numbering(part_zones, 'Cell')
    vtx_distri  = PTB._lngn_to_distri(vtx_lngn_list, comm)
    cell_distri = PTB._lngn_to_distri(cell_lngn_list, comm)

    MT.newDistribution({'Vertex' : vtx_distri, 'Cell' : cell_distri}, parent=dist_zone)
    if PT.Zone.Type(dist_zone) == "Unstructured":
      d_zone_dims = np.array([[vtx_distri[2], cell_distri[2], 0]], dtype=np.int32)
    elif PT.Zone.Type(dist_zone) == "Structured":
      d_zone_dims = _recover_dist_block_size(part_zones, comm)
      face_lngn_list = tr_utils.collect_cgns_g_numbering(part_zones, 'Face')
      face_distri = PTB._lngn_to_distri(face_lngn_list, comm)
      MT.newDistribution({'Face' : face_distri}, parent=dist_zone)
    PT.set_value(dist_zone, d_zone_dims)

    # > Create vertex distribution and exchange vertex coordinates
    d_grid_co = PT.new_GridCoordinates('GridCoordinates', parent=dist_zone)
    for coord in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
      PT.new_DataArray(coord, value=None, parent=d_grid_co)
    PTB.part_coords_to_dist_coords(dist_zone, part_zones, comm)

    # > Create elements
    _recover_elements(dist_zone, part_zones, comm)

    # > BND and JNS
    bc_t_path = 'ZoneBC_t/BC_t'
    gc_t_path = ['ZoneGridConnectivity_t', lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] and not MT.conv.is_intra_gc(PT.get_name(n))]

    # > Discover (skip GC created by partitioning)
    discover_nodes_from_matching(dist_zone, part_zones, bc_t_path, comm,
          child_list=['FamilyName_t', 'GridLocation_t'], get_value='all')
    discover_nodes_from_matching(dist_zone, part_zones, gc_t_path, comm,
          child_list=['GridLocation_t', 'GridConnectivityProperty_t', 'GridConnectivityDonorName', 'Transform'],
          merge_rule= lambda path: MT.conv.get_split_prefix(path), get_value='leaf')
    #After GC discovery, cleanup donor name suffix
    for jn in PT.iter_children_from_predicates(dist_zone, gc_t_path):
      val = PT.get_value(jn)
      PT.set_value(jn, MT.conv.get_part_prefix(val))
      gc_donor_name = PT.get_child_from_name(jn, 'GridConnectivityDonorName')
      PT.set_value(gc_donor_name, MT.conv.get_split_prefix(PT.get_value(gc_donor_name)))

    # > Index exchange (BCs and GCs)
    for bc_path in PT.predicates_to_paths(dist_zone, bc_t_path):
      if PT.Zone.Type(dist_zone) == 'Unstructured':
        IPTB.part_pl_to_dist_pl(dist_zone, part_zones, bc_path, comm)
      elif PT.Zone.Type(dist_zone) == 'Structured':
        IPTB.part_pr_to_dist_pr(dist_zone, part_zones, bc_path, comm)
    for gc_path in PT.predicates_to_paths(dist_zone, gc_t_path):
      if PT.Zone.Type(dist_zone) == 'Unstructured':
        IPTB.part_pl_to_dist_pl(dist_zone, part_zones, gc_path, comm, True)
      elif PT.Zone.Type(dist_zone) == 'Structured':
        IPTB.part_pr_to_dist_pr(dist_zone, part_zones, gc_path, comm, True)

    # > Flow Solution and Discrete Data
    PTB.part_sol_to_dist_sol(dist_zone, part_zones, comm)
    PTB.part_discdata_to_dist_discdata(dist_zone, part_zones, comm)
    PTB.part_subregion_to_dist_subregion(dist_zone, part_zones, comm)

    # > Todo : BCDataSet

  MJT.copy_donor_subset(dist_tree)

  return dist_tree

