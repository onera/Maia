import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.io          import distribution_tree
from maia.algo.dist   import redistribute
from maia.utils       import par_utils, np_utils

def distribute_pl_node(node, comm):
  """
  Distribute a standard node having a PointList (and its childs) over several processes,
  using uniform distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  dist_node = PT.deep_copy(node)
  n_elem = PT.Subset.n_elem(dist_node)
  distri = par_utils.uniform_distribution(n_elem, comm)

  #PL and PLDonor
  for array_n in PT.get_children_from_predicate(dist_node, 'IndexArray_t'):
    array_n[1] = array_n[1][:, distri[0]:distri[1]]
  #Data Arrays
  has_subset = lambda n : PT.get_child_from_name(n, 'PointList') is not None or PT.get_child_from_name(n, 'PointRange') is not None
  bcds_without_pl = lambda n : PT.get_label(n) == 'BCDataSet_t' and not has_subset(n)
  bcds_without_pl_query = [bcds_without_pl, 'BCData_t', 'DataArray_t']
  for array_path in ['DataArray_t', 'BCData_t/DataArray_t', bcds_without_pl_query]:
    for array_n in PT.iter_children_from_predicates(dist_node, array_path):
      array_n[1] = array_n[1][distri[0]:distri[1]]

  #Additionnal treatement for subnodes with PL (eg bcdataset)
  has_pl = lambda n : PT.get_name(n) not in ['PointList', 'PointRange'] and has_subset(n)
  for child in [node for node in PT.get_children(dist_node) if has_pl(node)]:
    dist_child = distribute_pl_node(child, comm)
    child[2] = dist_child[2]

  MT.newDistribution({'Index' : distri}, dist_node)

  return dist_node

def distribute_data_node(node, comm):
  """
  Distribute a standard node having arrays supported by allCells or allVertices over several processes,
  using uniform distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  is_data_array = lambda n: PT.get_label(n) == 'DataArray_t'
  assert PT.get_node_from_name(node, 'PointList') is None
  dist_node = PT.new_node(PT.get_name(node), PT.get_label(node), PT.get_value(node))

  for array in PT.iter_children_from_predicate(node, is_data_array):
    distri = par_utils.uniform_distribution(array[1].size, comm)
    PT.new_DataArray(PT.get_name(array),
                     (array[1].reshape(-1, order='F')[distri[0] : distri[1]]).copy(),
                     parent=dist_node) 
  for child in PT.iter_children_from_predicate(node, lambda n: not is_data_array(n)):
    PT.add_child(dist_node, PT.deep_copy(child))

  return dist_node

def distribute_element_node(node, comm):
  """
  Distribute a standard element node over several processes, using uniform distribution.
  Mainly useful for unit tests. Node must be know by each process.
  """
  assert PT.get_label(node) == 'Elements_t'
  dist_node = PT.deep_copy(node)

  n_elem = PT.Element.Size(node)
  distri = par_utils.uniform_distribution(n_elem, comm)
  MT.newDistribution({'Element' : distri}, dist_node)

  ec = PT.get_child_from_name(dist_node, 'ElementConnectivity')
  if PT.Element.CGNSName(node) in ['NGON_n', 'NFACE_n', 'MIXED']:
    eso = PT.get_child_from_name(dist_node, 'ElementStartOffset')
    distri_ec = eso[1][[distri[0], distri[1], -1]]
    ec[1] = ec[1][distri_ec[0] : distri_ec[1]]
    eso[1] = eso[1][distri[0]:distri[1]+1]

    MT.newDistribution({'ElementConnectivity' : np_utils.safe_int_cast(distri_ec, distri.dtype)}, dist_node)
  else:
    n_vtx = PT.Element.NVtx(node)
    ec[1] = ec[1][n_vtx*distri[0] : n_vtx*distri[1]]
    MT.newDistribution({'ElementConnectivity' : n_vtx*distri}, dist_node)
  
  pe = PT.get_child_from_name(dist_node, 'ParentElements')
  if pe is not None:
    pe[1] = (pe[1][distri[0] : distri[1]]).copy(order='F') #Copy is needed to have contiguous memory
  
  return dist_node

def _distribute_tree(tree, comm):
  """
  Distribute a standard cgns tree over several processes, using uniform distribution.
  Mainly useful for unit tests. Tree must be know by each process.
  """
  # Do a copy to capture all original nodes
  dist_tree = PT.deep_copy(tree)
  for zone in PT.iter_all_Zone_t(dist_tree):
    # > Cell & Vertex distribution
    n_vtx  = PT.Zone.n_vtx(zone)
    n_cell = PT.Zone.n_cell(zone)
    zone_distri = {'Vertex' : par_utils.uniform_distribution(n_vtx , comm),
                   'Cell'   : par_utils.uniform_distribution(n_cell, comm)}
    if PT.Zone.Type(zone) == 'Structured':
      zone_distri['Face'] = par_utils.uniform_distribution(PT.Zone.n_face(zone), comm)

    MT.newDistribution(zone_distri, zone)

    # > Coords
    grid_coords = PT.get_children_from_label(zone, 'GridCoordinates_t')
    for grid_coord in grid_coords:
      PT.rm_child(zone, grid_coord)
      PT.add_child(zone, distribute_data_node(grid_coord, comm))

    # > Elements
    elts = PT.get_children_from_label(zone, 'Elements_t')
    for elt in elts:
      PT.rm_child(zone, elt)
      PT.add_child(zone, distribute_element_node(elt, comm))

    # > Flow Solutions
    sols = PT.get_children_from_label(zone, 'FlowSolution_t') + PT.get_children_from_label(zone, 'DiscreteData_t')
    for sol in sols:
      PT.rm_child(zone, sol)
      if PT.get_child_from_name(sol, 'PointList') is None:
        PT.add_child(zone, distribute_data_node(sol, comm))
      else:
        PT.add_child(zone, distribute_pl_node(sol, comm))

    # > BCs
    zonebcs = PT.get_children_from_label(zone, 'ZoneBC_t')
    for zonebc in zonebcs:
      PT.rm_child(zone, zonebc)
      dist_zonebc = PT.new_child(zone, PT.get_name(zonebc), 'ZoneBC_t')
      for bc in PT.iter_children_from_label(zonebc, 'BC_t'):
        PT.add_child(dist_zonebc, distribute_pl_node(bc, comm))

    # > GCs
    zonegcs = PT.get_children_from_label(zone, 'ZoneGridConnectivity_t')
    for zonegc in zonegcs:
      PT.rm_child(zone, zonegc)
      dist_zonegc = PT.new_child(zone, PT.get_name(zonegc), 'ZoneGridConnectivity_t')
      for gc in PT.get_children_from_label(zonegc, 'GridConnectivity_t') + PT.get_children_from_label(zonegc, 'GridConnectivity1to1_t'):
        PT.add_child(dist_zonegc, distribute_pl_node(gc, comm))

    # > ZoneSubRegion
    zone_subregions = PT.get_children_from_label(zone, 'ZoneSubRegion_t')
    for zone_subregion in zone_subregions:
      # Trick if related to an other node -> add pl
      matching_region_path = PT.getSubregionExtent(zone_subregion, zone)
      if matching_region_path != PT.get_name(zone_subregion):
        PT.add_child(zone_subregion, PT.get_node_from_path(zone, matching_region_path + '/PointList'))
        PT.add_child(zone_subregion, PT.get_node_from_path(zone, matching_region_path + '/PointRange'))
      dist_zone_subregion = distribute_pl_node(zone_subregion, comm)
      if matching_region_path != PT.get_name(zone_subregion):
        PT.rm_children_from_name(dist_zone_subregion, 'PointList')
        PT.rm_children_from_name(dist_zone_subregion, 'PointRange')
        PT.rm_child(dist_zone_subregion, MT.getDistribution(dist_zone_subregion))

      PT.rm_child(zone, zone_subregion)
      PT.add_child(zone, dist_zone_subregion)

  return dist_tree

def _broadcast_full_to_dist(tree, comm, owner):
  """
  Create a distributed tree from a full tree holded by only one proc.
  """

  da_container = ['GridCoordinates_t', 'Elements_t', 'FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t',
      'BC_t', 'BCDataSet_t', 'BCData_t', 'GridConnectivity_t', 'GridConnectivity1to1_t']

  if comm.Get_rank() == owner:
    is_da_container = lambda n: PT.get_label(n) in da_container
    is_data_array   = lambda n: PT.get_label(n) == 'DataArray_t' and not PT.get_name(n).endswith('#Size')

    # Prepare disttree for owning rank : add #Size node to easily compute distribution and flatten S data
    dist_tree     = PT.deep_copy(tree)
    for zone in PT.iter_all_Zone_t(dist_tree):
      for container in PT.iter_nodes_from_predicate(zone, is_da_container, explore='deep'):
        for node in PT.get_children_from_predicate(container, 'DataArray_t'):
          if PT.get_name(node) != 'ParentElements':
            node[1] = node[1].reshape((-1), order='F')
          PT.new_node(PT.get_name(node)+'#Size', 'DataArray_t', node[1].shape, parent=container)
        for node in PT.get_children_from_predicate(container, 'IndexArray_t'):
          PT.new_node(PT.get_name(node)+'#Size', 'DataArray_t', node[1].shape, parent=container)

    # Prepare disttree for other rank: data are empty arrays. #Size node already added
    send_size_tree = PT.shallow_copy(dist_tree)
    for zone in PT.iter_all_Zone_t(send_size_tree):
      for container in PT.iter_nodes_from_predicate(zone, is_da_container, explore='deep'):
        for node in PT.get_children_from_predicate(container, is_data_array):
          # Be carefull with PE
          if PT.get_name(node) == 'ParentElements':
            PT.set_value(node, np.empty((0,2), dtype=node[1].dtype, order='F'))
          else:
            PT.set_value(node, np.empty(0, dtype=node[1].dtype))
        for node in PT.get_children_from_predicate(container, 'IndexArray_t'):
          index_dimension = PT.get_child_from_name(container, PT.get_name(node)+'#Size')[1][0]
          PT.set_value(node, np.empty((index_dimension,0), dtype=node[1].dtype, order='F'))
  else:
    send_size_tree = None

  recv_size_tree = comm.bcast(send_size_tree, root=owner)

  # Fix ElementStartOffset depending on receiving rank (0 or cnt#size)
  if comm.Get_rank() != owner:
    dist_tree = recv_size_tree
    for zone in PT.get_all_Zone_t(dist_tree):
      for elt in PT.get_children_from_label(zone, 'Elements_t'):
        eso_n = PT.get_child_from_name(elt, 'ElementStartOffset')
        if eso_n is not None:
          ec_size = PT.get_child_from_name(elt, 'ElementConnectivity#Size')[1]
          eso_n[1] = (comm.Get_rank() > owner) * np.array(ec_size, dtype=eso_n[1].dtype)

  # Create Distribution nodes from Size nodes
  distribution_tree.add_distribution_info(dist_tree, comm, f'gather.{owner}')
  PT.rm_nodes_from_name(dist_tree, '*#Size')

  return dist_tree

def full_to_dist_tree(tree, comm, owner=None):
  """ Generate a distributed tree from a standard (full) CGNS Tree.

  Input tree can be defined on a single process (using ``owner = rank_id``),
  or a copy can be known by all the processes (using ``owner=None``).

  In both cases, output distributed tree will be equilibrated over all the processes.

  Args:
    tree       (CGNSTree) : Full (not distributed) tree.
    comm        (MPIComm) : MPI communicator
    owner (int, optional) : MPI rank holding the input tree. Defaults to None.
  Returns:
    CGNSTree: distributed cgns tree

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #full_to_dist_tree@start
        :end-before: #full_to_dist_tree@end
        :dedent: 2
  """

  if owner is not None:
    dist_tree = _broadcast_full_to_dist(tree, comm, owner)
    redistribute.redistribute_tree(dist_tree, 'uniform', comm)
    return dist_tree
  else:
    return _distribute_tree(tree, comm)

