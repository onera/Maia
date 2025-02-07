from mpi4py import MPI
import numpy      as np
import operator

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.algo.dist             import matching_jns_tools as MJT
from maia.transfer              import utils              as tr_utils
from maia.transfer.part_to_dist import data_exchange      as PTB
from maia.transfer.part_to_dist import index_exchange     as IPTB
from maia.transfer.part_to_dist import tree_api           as part_to_dist
from maia.utils                 import py_utils, par_utils
from maia                       import npy_pdm_gnum_dtype as pdm_dtype

from maia.pytree.graph.algo import step
class UDDCollector:
  """ A visitor for depth_first_search that collect the paths of UserDefinedData nodes """
  def __init__(self):
      self.ud_paths = list()
  def pre(self, nodes):
    last = nodes[-1]
    if PT.get_label(last) == 'UserDefinedData_t' and PT.get_name(last) not in [':CGNS#GlobalNumbering', ':CGNS#LocalNumbering']:
      path = "/".join([PT.get_name(n) for n in nodes])
      # Remove maia naming conventions, since paths should be given on disttree
      for i, node in enumerate(nodes):
        if PT.get_label(node) == 'Zone_t':
          path = PT.utils.update_path_elt(path,i, lambda s: MT.conv.get_part_prefix(s))
        elif PT.get_label(node) in ['GridConnectivity_t', 'GridConnectivity1to1_t']:
          path = PT.utils.update_path_elt(path,i, lambda s: MT.conv.get_split_prefix(s))
      self.ud_paths.append(PT.utils.path_tail(path, 1))
      return step.over # Stop exploring this level after search

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
          # If values are not needed, use PT.UNSET and not None, otherwise PT.update_child may erase
          # existing value
          values = [PT.get_value(node) if value else PT.UNSET for node, value in zip(nodes, get_value)]

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

def _get_joins_dist_tree(parts_per_dom, comm):
  """
  """
  is_face_intra_gc = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
                               and PT.Subset.GridLocation(n) == 'FaceCenter' \
                               and not MT.conv.is_intra_gc(PT.get_name(n))
  has_face_intra_gc = \
      lambda z: PT.get_node_from_predicates(z,  ['ZoneGridConnectivity_t', is_face_intra_gc]) is not None

  dist_tree = PT.new_CGNSTree()
  for dist_zone_path, part_zones in parts_per_dom.items():
    dist_base_name, dist_zone_name = dist_zone_path.split('/')
    dist_base = PT.update_child(dist_tree, dist_base_name, 'CGNSBase_t')
    dist_zone = PT.update_child(dist_base, dist_zone_name, 'Zone_t')

    PT.new_child(dist_zone, 'ZoneType', 'ZoneType_t', 'Unstructured')
    PT.maia.newDistribution(parent=dist_zone) # Needed to call relevant function nace_to_pe later
    # Elements are needed only if there are some FaceCenter jns
    if par_utils.any_true(part_zones, has_face_intra_gc, comm):
      _recover_elements(dist_zone, part_zones, comm)
    _recover_GC(dist_zone, part_zones, comm)

  return dist_tree

def get_joins_dist_tree(part_tree, comm):
  """ Recreate a dist tree containing only original jns from
  the partitioned tree (with PL). Only for U blocks !"""
  parts_per_dom = get_parts_per_blocks(part_tree, comm)
  return _get_joins_dist_tree(parts_per_dom, comm)

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
      donor_path = PT.get_value(intra_jn)
      donor_zone = donor_path if not '/' in donor_path else donor_path.split('/')[1]
      light_jn = PT.new_GridConnectivity1to1(donor_name=donor_zone,
                                             point_range=PT.Subset.getPatch(intra_jn)[1])
      zones_to_join[zone_name].append(light_jn)

  # Gather and flatten dicts
  zones_to_size_g = {}
  zones_to_join_g = {}
  for zones_to_size_rank in comm.allgather(zones_to_size):
    zones_to_size_g.update(zones_to_size_rank)
  for zones_to_join_rank in comm.allgather(zones_to_join):
    zones_to_join_g.update(zones_to_join_rank)

  # Choose any starting point
  first = next(iter(zones_to_size_g))
  idx_dim = len(zones_to_size_g[first])
  d_zone_dims = np.zeros((idx_dim,3), pdm_dtype, order='F')
  d_zone_dims[:,1] += zones_to_size_g[first] #Cell size
  for axis in range(3):
    for oper in [operator.ne, operator.eq]: #Go front (vtx != 1), then back (vtx == 1)
      # Reset
      keep_going = True
      current = first
      while keep_going:
        # Iterate jns and select one to continue in same axis/direction
        for jn in zones_to_join_g[current]:
          pr = PT.get_child_from_name(jn, 'PointRange')[1]
          if PT.Subset.normal_axis(jn) == axis and oper(pr[axis,0], 1):
            current = PT.get_value(jn)
            d_zone_dims[axis,1] += zones_to_size_g[current][axis]
            break
        else: #If loop did not break -> we reached the end of block
          keep_going = False
  d_zone_dims[:,0] = d_zone_dims[:,1] + 1 # Update vertices
  return d_zone_dims

def _recover_elements(dist_zone, part_zones, comm):
  # > Get the list of part elements
  fake_zone = PT.shallow_copy(dist_zone) #This is just to store the elements
  discover_nodes_from_matching(fake_zone, part_zones, 'Elements_t', comm, get_value='leaf')
  discover_nodes_from_matching(fake_zone, part_zones, 'Elements_t/ParentElements', comm)
  elt_names = [PT.get_name(elt)         for elt in PT.get_children_from_label(fake_zone, 'Elements_t')]
  elt_kinds = [PT.Element.CGNSName(elt) for elt in PT.get_children_from_label(fake_zone, 'Elements_t')]
  has_ngon  = 'NGON_n'  in elt_kinds
  has_nface = 'NFACE_n' in elt_kinds
  has_edge  = 'BAR_2'   in elt_kinds
  has_pe    = PT.get_child_from_predicates(fake_zone, 'Elements_t/ParentElements') is not None

  is_poly = has_ngon
  if not is_poly and has_edge: # Maybe 2D Poly with Bar + ParentElements
    is_bar = lambda n : PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'BAR_2'
    is_poly = PT.get_child_from_predicates(fake_zone, [is_bar, 'ParentElements']) is not None

  # Deal Edge/NGon & NGon/NFace
  if is_poly:
    cell_dim = PT.Zone.CellDimension(fake_zone)
    assert all([kind in ['NGON_n', 'NFACE_n', 'BAR_2'] for kind in elt_kinds])
    if cell_dim == 2:
      n_edge_tot = 0
      if has_edge: #2D with Edge + NGON or Edge only or NGON only
        assert all([PT.Zone.CellDimension(zone) == 2 for zone in part_zones])
        edge_name = elt_names[elt_kinds.index('BAR_2')]
        edge_elts = [PT.get_child_from_name(part_zone, edge_name) for part_zone in part_zones]
        # For EdgeElements, we call part_ngon_to_dist_ngon which manages ParentElements node
        # We need to create ElementStartOffset array to do that
        for edge_elt in edge_elts:
          PT.new_DataArray('ElementStartOffset', 2*np.arange(PT.Element.Size(edge_elt)+1, dtype=np.int32), parent=edge_elt)
        IPTB.part_ngon_to_dist_ngon(dist_zone, part_zones, edge_name, comm)
        for edge_elt in edge_elts:
          PT.rm_children_from_name(edge_elt, 'ElementStartOffset') # Cleanup
        dist_edge_elt = PT.get_child_from_name(dist_zone, edge_name)
        dist_edge_elt[1][0] = 3
        PT.rm_node_from_path(dist_edge_elt, 'ElementStartOffset')
        PT.rm_node_from_path(dist_edge_elt, ':CGNS#Distribution/ElementConnectivity')
        n_edge_tot = PT.Element.Range(dist_edge_elt)[1]
      if has_ngon:
        # Now treat true 2D NGON node
        ngon_name = elt_names[elt_kinds.index('NGON_n')]
        IPTB.part_ngon_to_dist_ngon(dist_zone, part_zones, ngon_name, comm)
        # > Shift ngon element_range and create all cell distri
        ngon_range = PT.get_node_from_path(dist_zone, f'{ngon_name}/ElementRange')[1]
        ngon_range += n_edge_tot

    elif cell_dim == 3: #3D with NGON + NFACE or NGON only
      from maia.algo                  import pe_to_nface, nface_to_pe
      from maia.factory.partitioning  import part_bound_orient as PBO
      
      _part_zones = [PT.shallow_copy(zone) for zone in part_zones] # Since we add/remove nodes, do a shallow copy
      if not PBO.orientation_preserved(_part_zones, comm):
        # This is to avoid modification of input partitioned tree
        to_copy = lambda n : PT.get_name(n) in ['ElementConnectivity', 'ParentElements']
        for zone in _part_zones:
          for node in PT.get_children_from_predicates(zone, ['Elements_t', to_copy]):
            node[1] = node[1].copy()
        PBO.preserve_orientation(_part_zones, comm)
      for zone in _part_zones:
        if not has_nface:
          pe_to_nface(zone)
        PT.rm_children_from_name(PT.Zone.NGonNode(zone), 'ParentElements')

      ngon_name = elt_names[elt_kinds.index('NGON_n')]
      nface_name = elt_names[elt_kinds.index('NFACE_n')] if has_nface else 'NFaceElements'
      IPTB.part_ngon_to_dist_ngon(dist_zone, _part_zones, ngon_name, comm)
      IPTB.part_nface_to_dist_nface(dist_zone, _part_zones, nface_name, ngon_name, comm)
      # > Shift nface element_range
      n_face_tot  = PT.get_node_from_path(dist_zone, f'{ngon_name}/ElementRange')[1][1]
      nface_range = PT.get_node_from_path(dist_zone, f'{nface_name}/ElementRange')[1]
      nface_range += n_face_tot
      if has_pe:
        nface_to_pe(dist_zone, comm)
      if not has_nface:
        PT.rm_children_from_name(dist_zone, nface_name)

  # Deal standard elements
  else:
    for elt_name in elt_names:
      IPTB.part_elt_to_dist_elt(dist_zone, part_zones, elt_name, comm)

    elt_nodes = PT.get_children_from_label(dist_zone, 'Elements_t') #True elements
    # > Get shift per dim
    n_elt_per_dim  = [0,0,0,0]
    for elt in elt_nodes:
      n_elt_per_dim[PT.Element.Dimension(elt)] += PT.Element.Size(elt)

    elt_order = [PT.Zone.elt_ordering_by_dim(part_zone) for part_zone in part_zones
                 if sum([d != [0,0] for d in PT.Zone.get_elt_range_per_dim(part_zone)]) > 1]
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

def _recover_BC(dist_zone, part_zones, comm):
  bc_predicate = ['ZoneBC_t', 'BC_t']

  discover_nodes_from_matching(dist_zone, part_zones, bc_predicate, comm,
        child_list=['FamilyName_t', 'GridLocation_t', 'Ordinal_t', 'AdditionalFamilyName_t'], get_value='all')

  for bc_path in PT.predicates_to_paths(dist_zone, bc_predicate):
    if PT.Zone.Type(dist_zone) == 'Unstructured':
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, bc_path, comm)
    elif PT.Zone.Type(dist_zone) == 'Structured':
      IPTB.part_pr_to_dist_pr(dist_zone, part_zones, bc_path, comm)

def _recover_GC(dist_zone, part_zones, comm):
  is_gc       = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  is_gc_intra = lambda n: is_gc(n) and not MT.conv.is_intra_gc(PT.get_name(n))

  gc_predicate = ['ZoneGridConnectivity_t', is_gc_intra]

  discover_nodes_from_matching(dist_zone, part_zones, gc_predicate, comm,
        child_list=['GridLocation_t', 'GridConnectivityType_t', 'GridConnectivityProperty_t',
                    'GridConnectivityDonorName', 'Transform', 'FamilyName_t', 'AdditionalFamilyName_t'],
        merge_rule=lambda path: MT.conv.get_split_prefix(path), get_value='leaf')

  #After GC discovery, cleanup donor name suffix
  for jn in PT.iter_children_from_predicates(dist_zone, gc_predicate):
    val = PT.get_value(jn)
    PT.set_value(jn, MT.conv.get_part_prefix(val))
    if PT.GridConnectivity.is1to1(jn):
      gc_donor_name = PT.get_child_from_name(jn, 'GridConnectivityDonorName')
      PT.set_value(gc_donor_name, MT.conv.get_split_prefix(PT.get_value(gc_donor_name)))

  # Index exchange
  for gc_path in PT.predicates_to_paths(dist_zone, gc_predicate):
    if PT.Zone.Type(dist_zone) == 'Unstructured':
      IPTB.part_pl_to_dist_pl(dist_zone, part_zones, gc_path, comm, True)
    elif PT.Zone.Type(dist_zone) == 'Structured':
      zgc_name, gc_name = gc_path.split('/')
      part_gcs = [PT.get_nodes_from_predicates(part, [zgc_name, gc_name+'*']) for part in part_zones]
      part_gcs = py_utils.to_flat_list(part_gcs)
      if par_utils.exists_everywhere(part_gcs, 'PointRange', comm):
        IPTB.part_pr_to_dist_pr(dist_zone, part_zones, gc_path, comm, True)
      elif par_utils.exists_everywhere(part_gcs, 'PointList', comm):
        IPTB.part_pl_to_dist_pl(dist_zone, part_zones, gc_path, comm, True)

def _recover_base_iterative_data(dist_tree, part_tree, comm):
  # > Add BaseIterativeData by hand, because we need to manage the names
  for dist_base in PT.get_all_CGNSBase_t(dist_tree):
    part_base = PT.get_child_from_name(part_tree, PT.get_name(dist_base))
    # part_base may not exist on some ranks; we do something only if BaseIterativeData_t
    # exists on all rank knowing the base
    p_it_data_loc = part_base is None or PT.get_child_from_label(part_base, 'BaseIterativeData_t') is not None
    
    if comm.allreduce(p_it_data_loc, MPI.LAND):
      # We remove the initial node for BaseIterativeData in the dist_tree in order to
      # ensure having all data from the part_trees when restarting an unsteady case
      PT.rm_children_from_label(dist_base, 'BaseIterativeData_t')
      # If base does not exists, we dont have data -> work on ranks having data
      subcomm = comm.Split(part_base is None)
      if part_base is not None:
        p_it_data = PT.get_child_from_label(part_base, 'BaseIterativeData_t')
        p_z_pointers = PT.get_child_from_name(p_it_data, 'ZonePointers')
        d_it_data = PT.deep_copy(p_it_data)
        if p_z_pointers is not None:
          part_zp = subcomm.allgather(PT.get_value(p_z_pointers))
          dist_zp = []
          for i in range(len(part_zp[0])):
            znames = [part_zp[ip][i] for ip in range(subcomm.Get_size())]
            znames = [MT.conv.get_part_prefix(elt) for item in znames for elt in item if elt != ''] # Remove suffix + flatten
            dist_zp.append( sorted(set(znames)) )
          PT.update_child(d_it_data, 'ZonePointers', 'DataArray_t', value=dist_zp)
          PT.update_child(d_it_data, 'NumberOfZones', 'DataArray_t', value=[len(k) for k in dist_zp])
      else:
        d_it_data = None

      if not subcomm.Get_size() == comm.Get_size(): #Some ranks have not data, we need to broadcast
        root = comm.allreduce(comm.rank if part_base is not None else -1, MPI.MAX) # A rank knowing part base
        d_it_data = comm.bcast(d_it_data, root=root)
      PT.add_child(dist_base, d_it_data)

def recover_dist_tree(part_tree, comm, data_transfer=[]):
  """ Regenerate a distributed tree from a partitioned tree.

  The partitioned tree should have been created using Maia, or
  must at least contains GlobalNumbering nodes as defined by Maia
  (see :ref:`part_tree`).

  Important:
    Similarly to :func:`partition_dist_tree`, this function reports only geometric information
    (such as boundary conditions, zone subregion, etc.) on the created dist_tree;
    data fields are **not** transfered
    automatically. Use :attr:`data_transfer` keyword argument
    or see :ref:`Transfer module<user_man_transfer>`. 
  
  Args:
    part_tree (CGNSTree) : Partitioned CGNS Tree
    comm       (MPIComm) : MPI communicator
    data_transfer (list of str): Labels of data nodes to transfer during operation
      (see :attr:`data_transfer`)
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
  
  # Check that dist zone name will not clash with Family_t nodes
  for dbase in PT.iter_all_CGNSBase_t(dist_tree):
    basename = PT.get_name(dbase)
    pbase = PT.get_child_from_name(part_tree, basename)
    family_names = set(PT.get_name(n) for n in PT.iter_children_from_label(dbase, 'Family_t'))
    zone_names   = set(MT.conv.get_part_prefix(PT.get_name(z)) for z in PT.iter_all_Zone_t(pbase)) if pbase is not None else set()
    if not comm.allreduce(family_names.isdisjoint(zone_names), MPI.LAND):
      all_zone_names = comm.allreduce(zone_names, op=lambda s1,s2 : s1 | s2)
      msg = f"Two children of the same CGNSBase_t node can not have the same name. " \
            f"Clash detected between Zone and Family {family_names & all_zone_names} under parent '{basename}'."
      raise RuntimeError(msg)
    
  discover_nodes_from_matching(dist_tree, [part_tree], 'CGNSBase_t/Zone_t', comm,\
                               child_list = ['ZoneType_t', 'FamilyName_t', 'AdditionalFamilyName_t'],
                               merge_rule=lambda zpath : MT.conv.get_part_prefix(zpath))

  _recover_base_iterative_data(dist_tree, part_tree, comm)

  for dist_zone_path in PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t'):
    dist_zone = PT.get_node_from_path(dist_tree, dist_zone_path)

    part_zones = tr_utils.get_partitioned_zones(part_tree, dist_zone_path)

    discover_nodes_from_matching(dist_zone, part_zones, "ZoneIterativeData_t/*",
                                 comm, get_value="all")

    # Create zone distributions
    vtx_lngn_list  = tr_utils.collect_cgns_g_numbering(part_zones, 'Vertex')
    cell_lngn_list = tr_utils.collect_cgns_g_numbering(part_zones, 'Cell')
    vtx_distri  = par_utils.distribution_from_gnum(vtx_lngn_list, comm)
    cell_distri = par_utils.distribution_from_gnum(cell_lngn_list, comm)

    MT.newDistribution({'Vertex' : vtx_distri, 'Cell' : cell_distri}, parent=dist_zone)
    if PT.Zone.Type(dist_zone) == "Unstructured":
      d_zone_dims = np.array([[vtx_distri[2], cell_distri[2], 0]], dtype=pdm_dtype)
    elif PT.Zone.Type(dist_zone) == "Structured":
      d_zone_dims = _recover_dist_block_size(part_zones, comm)
      if d_zone_dims.shape[0] == 3:
        face_lngn_list = tr_utils.collect_cgns_g_numbering(part_zones, 'Face')
        face_distri = par_utils.distribution_from_gnum(face_lngn_list, comm)
        MT.newDistribution({'Face' : face_distri}, parent=dist_zone)
    PT.set_value(dist_zone, d_zone_dims)

    # > Create vertex distribution and exchange vertex coordinates
    coords_name , transform_n = (None, None)
    owner = -1
    if len(part_zones) > 0:
      coords_name = PT.Zone.coordinates(part_zones[0])._fields
      transform_n = PT.get_node_from_predicates(part_zones[0], 'GridCoordinates_t/CoordinateTransform')
      owner = comm.Get_rank()

    root = comm.allreduce(owner, MPI.MAX) # Find a rank knowing partitioned data for this zone
    coords_name, transform_n = comm.bcast((coords_name, transform_n), root)

    d_grid_co = PT.new_GridCoordinates('GridCoordinates', parent=dist_zone)
    for coord in coords_name:
      PT.new_DataArray(coord, value=None, parent=d_grid_co)
    PT.add_child(d_grid_co, transform_n)
    PTB.part_coords_to_dist_coords(dist_zone, part_zones, comm)

    # > Create elements
    _recover_elements(dist_zone, part_zones, comm)

    # > BND and JNS
    _recover_BC(dist_zone, part_zones, comm)
    _recover_GC(dist_zone, part_zones, comm)
    
    # To mimic partitioning behaviour, we create here the geometric support of containers
    # (such as ZoneSubRegion) without transfering fields
    filter = {'FlowSolution_t'         : ('I', ['*/']),
              'DiscreteData_t'         : ('I', ['*/']),
              'ZoneSubRegion_t'        : ('I', ['*/']),
              'BCDataSet_t'            : ('I', ['ZoneBC_t/*/*/']),
              'ArbitraryGridMotion_t'  : ('I', [])}

    part_to_dist._part_zones_to_dist_zone(dist_zone, part_zones, comm, filter)
    is_empty_cont = lambda n : PT.get_label(n) in ['FlowSolution_t', 'DiscreteData_t', 'BCDataSet_t'] \
                           and PT.maia.getDistribution(n) is None
    PT.rm_children_from_predicate(dist_zone, is_empty_cont)
    for dist_bc in PT.iter_children_from_labels(dist_zone, ['ZoneBC_t', 'BC_t']):
      PT.rm_children_from_predicate(dist_bc, is_empty_cont)
      PT.rm_nodes_from_label(dist_bc, 'BCData_t', depth=2)

  MJT.copy_donor_subset(dist_tree)

  # Transfer fields
  if isinstance(data_transfer, str): # Convert to list if single string provided
    data_transfer = [data_transfer]
  # Fields
  if 'FIELDS' in data_transfer or 'ALL' in data_transfer:
    labels = part_to_dist.LABELS
  else:
    labels = [label for label in part_to_dist.LABELS if label in data_transfer]
  # UserDefinedData
  if 'UserDefinedData_t' in data_transfer or 'ALL' in data_transfer:
    PT.graph.cgns.depth_first_search(part_tree, v := UDDCollector(), depth='all')
    # Propagate paths across ranks
    ud_paths = sorted(set([path for rank_paths in comm.allgather(v.ud_paths) for path in rank_paths]))
  else:
    ud_paths = []

  if labels:
    part_to_dist.part_tree_to_dist_tree_only_labels(dist_tree, part_tree, labels, comm)
  for path in ud_paths:
    part_to_dist.part_tree_to_dist_tree_copy(dist_tree, part_tree, path, comm)

  return dist_tree

