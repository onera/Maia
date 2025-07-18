import numpy as np

from maia.typing        import *
from maia.pytree.typing import Predicates

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.io          import distribution_tree
from maia.algo.dist   import redistribute
from maia.utils       import par_utils, np_utils
from typing           import overload

def distribute_pl_node(node:CGNSTree, comm:MPIComm) -> CGNSTree:
  """
  Distribute a standard node having a PointList (and its childs) over several processes,
  using uniform distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  dist_node = PT.deep_copy(node)
  n_elem = PT.Subset.n_elem(dist_node)
  distri = par_utils.uniform_distribution(n_elem, comm)
  #PL and PLDonor
  for array_n in PT.get_children_from_predicate(dist_node, 'IndexArray_t'):
    array = PT.get_np_value(array_n)
    PT.set_value(array_n, array[:, distri[0]:distri[1]])
  # Standard Data Arrays
  for array_n in PT.iter_children_from_label(dist_node, 'DataArray_t'):
    array = PT.get_np_value(array_n)
    PT.set_value(array_n, array[distri[0]:distri[1]])
  # BCData_t arrays case : can be scalar or vector
  bcds_without_pl = PT.pred.label_is('BCDataSet_t') & ~PT.pred.IS_SUBSET
  bcds_without_pl_query:Predicates = [bcds_without_pl, 'BCData_t', 'DataArray_t']
  global_arrays_list = []
  for query in ['BCData_t/DataArray_t', bcds_without_pl_query]:
    for array_path in PT.predicates_to_paths(dist_node, query):
      array_n = PT.find_node_from_path(dist_node, array_path)
      array = PT.get_np_value(array_n)
      if array.size != 1:
        PT.set_value(array_n, array[distri[0]:distri[1]])
      else:
        global_arrays_list.append(array_path)

  #Additionnal treatement for subnodes with PL (eg bcdataset)
  has_pl = ~PT.pred.name_in(['PointList', 'PointRange']) & PT.pred.IS_SUBSET
  for child in [node for node in PT.get_children(dist_node) if has_pl(node)]:
    dist_child = distribute_pl_node(child, comm)
    PT.set_children(child, PT.get_children(dist_child))

  distri_n = MT.new_Distribution({'Index' : distri}, dist_node)
  if len(global_arrays_list) > 0:
    PT.new_Descriptor('BCDataGlobal', '\n'.join(global_arrays_list), parent=distri_n)

  return dist_node

def distribute_data_node(node:CGNSTree, comm:MPIComm) -> CGNSTree:
  """
  Distribute a standard node having arrays supported by allCells or allVertices over several processes,
  using uniform distribution. Mainly useful for unit tests. Node must be know by each process.
  """
  assert PT.get_node_from_name(node, 'PointList') is None
  dist_node = PT.new_node(PT.get_name(node), PT.get_label(node), PT.get_value(node))

  for child in PT.get_children(node):
    if PT.get_label(child) == 'DataArray_t':
      val = PT.get_np_value(child)
      distri = par_utils.uniform_distribution(val.size, comm)
      PT.new_DataArray(PT.get_name(child),
                      (val.reshape(-1, order='F')[distri[0] : distri[1]]).copy(),
                      parent=dist_node) 
    else:
      PT.add_child(dist_node, PT.deep_copy(child))

  return dist_node

def distribute_element_node(node:CGNSTree, comm:MPIComm) -> CGNSTree:
  """
  Distribute a standard element node over several processes, using uniform distribution.
  Mainly useful for unit tests. Node must be know by each process.
  """
  assert PT.get_label(node) == 'Elements_t'
  dist_node = PT.deep_copy(node)

  n_elem = PT.Element.Size(node)
  distri = par_utils.uniform_distribution(n_elem, comm)
  MT.new_Distribution({'Element' : distri}, dist_node)

  ec_n = PT.find_child_from_name(dist_node, 'ElementConnectivity')
  ec = PT.get_np_value(ec_n)
  if PT.Element.Type(node) in ['NGON_n', 'NFACE_n', 'MIXED']:
    eso_n = PT.find_child_from_name(dist_node, 'ElementStartOffset')
    eso = PT.get_np_value(eso_n)
    distri_ec = eso[[distri[0], distri[1], -1]]
    PT.set_value(ec_n, ec[distri_ec[0] : distri_ec[1]])
    PT.set_value(eso_n, eso[distri[0]:distri[1]+1])
    MT.new_Distribution({'ElementConnectivity' : np_utils.safe_int_cast(distri_ec, distri.dtype)}, dist_node)
  else:
    n_vtx = PT.Element.NVtx(node)
    PT.set_value(ec_n, ec[n_vtx*distri[0] : n_vtx*distri[1]])
    MT.new_Distribution({'ElementConnectivity' : n_vtx*distri}, dist_node)
  
  pe_n = PT.get_child_from_name(dist_node, 'ParentElements')
  if pe_n is not None:
    pe = PT.get_np_value(pe_n)
    PT.set_value(pe_n, (pe[distri[0] : distri[1]]).copy(order='F')) #Copy is needed to have contiguous memory
  
  return dist_node

def _distribute_tree(tree: CGNSTree, comm: MPIComm) -> CGNSDistTree:
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

    MT.new_Distribution(zone_distri, zone)

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
      matching_region_path = PT.Subset.ZSRExtent(zone_subregion, zone)
      if matching_region_path != PT.get_name(zone_subregion):
        PT.add_child(zone_subregion, PT.get_node_from_path(zone, matching_region_path + '/PointList'))
        PT.add_child(zone_subregion, PT.get_node_from_path(zone, matching_region_path + '/PointRange'))
      dist_zone_subregion = distribute_pl_node(zone_subregion, comm)
      if matching_region_path != PT.get_name(zone_subregion):
        PT.rm_children_from_name(dist_zone_subregion, 'PointList')
        PT.rm_children_from_name(dist_zone_subregion, 'PointRange')
        PT.rm_child(dist_zone_subregion, MT.get_Distribution(dist_zone_subregion))

      PT.rm_child(zone, zone_subregion)
      PT.add_child(zone, dist_zone_subregion)

  return CGNSDistTree(dist_tree)

def _broadcast_full_to_dist(tree: Optional[CGNSTree],
                            comm: MPIComm, 
                            owner: int) -> CGNSDistTree:
  """
  Create a distributed tree from a full tree holded by only one proc.
  """

  da_container = ['GridCoordinates_t', 'Elements_t', 'FlowSolution_t', 'DiscreteData_t', 'ZoneSubRegion_t',
      'BC_t', 'BCDataSet_t', 'BCData_t', 'GridConnectivity_t', 'GridConnectivity1to1_t']

  if comm.Get_rank() == owner:
    assert tree is not None
    is_da_container = PT.pred.label_in(da_container)
    is_data_array   = PT.pred.label_is('DataArray_t') & ~PT.pred.name_matches('*#Size')

    # Prepare disttree for owning rank : add #Size node to easily compute distribution and flatten S data
    dist_tree     = PT.deep_copy(tree)
    for zone in PT.iter_all_Zone_t(dist_tree):
      for container in PT.iter_nodes_from_predicate(zone, is_da_container, explore='deep'):
        for node in PT.get_children_from_predicate(container, 'DataArray_t'):
          assert (node_val := node[1]) is not None
          if PT.get_name(node) != 'ParentElements':
            PT.set_value(node, node_val.reshape((-1), order='F'))
          scalar_ds = PT.get_label(container) == 'BCData_t' and node_val.size == 1
          if not scalar_ds:
            PT.new_node(PT.get_name(node)+'#Size', 'DataArray_t', node_val.shape, parent=container)
        for node in PT.get_children_from_predicate(container, 'IndexArray_t'):
          assert (node_val := node[1]) is not None
          PT.new_node(PT.get_name(node)+'#Size', 'DataArray_t', node_val.shape, parent=container)

    # Prepare disttree for other rank: data are empty arrays. #Size node already added
    send_size_tree = PT.shallow_copy(dist_tree)
    for zone in PT.iter_all_Zone_t(send_size_tree):
      for container in PT.iter_nodes_from_predicate(zone, is_da_container, explore='deep'):
        for node in PT.get_children_from_predicate(container, is_data_array):
          assert (node_val := node[1]) is not None
          # Be carefull with PE
          if PT.get_name(node) == 'ParentElements':
            PT.set_value(node, np.empty((0,2), dtype=node_val.dtype, order='F'))
          elif PT.get_label(container) == 'BCData_t' and node_val.size == 1:
            pass # skip scalar BCDS   
          else:
            PT.set_value(node, np.empty(0, dtype=node_val.dtype))
        for node in PT.get_children_from_predicate(container, 'IndexArray_t'):
          index_dimension_n = PT.find_child_from_name(container, PT.get_name(node)+'#Size')
          index_dimension = index_dimension_n[1][0] #type:ignore[index] #(Node is created before, should not be None)
          assert (node_val := node[1]) is not None
          PT.set_value(node, np.empty((index_dimension,0), dtype=node_val.dtype, order='F'))
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
          assert (eso_val := eso_n[1]) is not None
          ec_size = PT.find_child_from_name(elt, 'ElementConnectivity#Size')[1]
          PT.set_value(eso_n, (comm.Get_rank() > owner) * np.array(ec_size, dtype=eso_val.dtype))

  # Create Distribution nodes from Size nodes
  distribution_tree.add_distribution_info(dist_tree, comm, f'gather.{owner}')
  PT.rm_nodes_from_name(dist_tree, '*#Size')

  return CGNSDistTree(dist_tree)

@overload
def full_to_dist_tree(full_tree: CGNSTree, comm: MPIComm, owner: None) -> CGNSDistTree: ...
@overload
def full_to_dist_tree(full_tree: Optional[CGNSTree], comm: MPIComm, owner: Optional[int]) -> CGNSDistTree: ...

def full_to_dist_tree(full_tree: Optional[CGNSTree],
                      comm: MPIComm, 
                      owner: Optional[int] = None) -> CGNSDistTree:
  """ Generate a distributed tree from a standard (full) CGNS Tree.

  Input tree can be defined on a single process (using ``owner = rank_id``),
  or a copy can be known by all the processes (using ``owner=None``).

  In both cases, output distributed tree will be equilibrated over all the processes.

  Args:
    full_tree   (CGNSTree) : Full (not distributed) tree.
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
  if full_tree is not None:
    MT.check_cgns_full_tree(full_tree)
  if owner is not None:
    dist_tree = _broadcast_full_to_dist(full_tree, comm, owner)
    redistribute.redistribute_tree(dist_tree, 'uniform', comm)
    return dist_tree
  else:
    assert full_tree is not None
    return _distribute_tree(full_tree, comm)

