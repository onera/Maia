import mpi4py.MPI as MPI
import numpy      as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT
import maia.transfer.protocols as MTP

from maia.utils import par_utils


# ---------------------------------------------------------------------------------------
def redistribute_pl_node(node, distribution, comm):
  """
  Redistribute a standard node having a PointList (and its childs) over several processes,
  using a given distribution function. Mainly useful for unit tests. Node must be known by
  each process.
  """
  node_distrib = MT.getDistribution(node, 'Index')[1]
  new_distrib = distribution(node_distrib[2], comm)
  MT.newDistribution({'Index' : new_distrib}, node)

  #PL and PLDonor
  for array_n in PT.get_children_from_predicate(node, 'IndexArray_t'):
    array_n[1] = MTP.block_to_block(array_n[1][0], node_distrib, new_distrib, comm).reshape(1,-1, order='F')

  #Data Arrays
  has_subset = lambda n : PT.get_child_from_name(n, 'PointList') is not None or PT.get_child_from_name(n, 'PointRange') is not None
  bcds_without_pl = lambda n : PT.get_label(n) == 'BCDataSet_t' and not has_subset(n)
  bcds_without_pl_query = [bcds_without_pl, 'BCData_t', 'DataArray_t']
  for array_path in ['DataArray_t', 'BCData_t/DataArray_t', bcds_without_pl_query]:
    for array_n in PT.iter_children_from_predicates(node, array_path):
      array_n[1] = MTP.block_to_block(array_n[1], node_distrib, new_distrib, comm)

  #Additionnal treatement for subnodes with PL (eg bcdataset)
  has_pl = lambda n : PT.get_name(n) not in ['PointList', 'PointRange'] and has_subset(n)
  for child in [node for node in PT.get_children(node) if has_pl(node)]:
    redistribute_pl_node(child, comm)

# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def redistribute_data_node(node, distri, new_distri, comm):
  """
  Distribute a standard node having arrays supported by allCells or allVertices over several processes,
  using given distribution. Mainly useful for unit tests. Node must be known by each process.
  """
  assert PT.get_node_from_name(node, 'PointList') is None

  for array in PT.iter_children_from_label(node, 'DataArray_t'):
    array[1] = MTP.block_to_block(array[1], distri, new_distri, comm)

# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def redistribute_elements_node(node, distribution, comm):

  assert PT.get_label(node) == 'Elements_t'

  has_eso = PT.Element.CGNSName(node) in ['NGON_n', 'NFACE_n', 'MIXED']

  # Get element distribution
  elt_distrib = MT.getDistribution(node, "Element")[1]
  n_elt       = elt_distrib[2]

  # New element distribution
  new_elt_distrib = distribution(n_elt, comm)
  new_distrib     = {'Element': new_elt_distrib}

  # > ElementStartOffset
  if has_eso :
    ec_distrib     = MT.getDistribution(node, "ElementConnectivity")[1]

    eso_n = PT.get_child_from_name(node, 'ElementStartOffset')
    eso   = PT.get_value(eso_n)

    # To be consistent with initial distribution, send everything excepted last elt
    eso_wo_last = MTP.block_to_block(eso[:-1], elt_distrib, new_elt_distrib, comm)

    # Now we have to recover the last on each proc (with is the first
    # of the next proc *having data*)

    # Each rank gather its first element of eso array
    bound_elt = eso_wo_last[0] if eso_wo_last.size > 0 else -1
    all_bound_elt = np.empty(comm.Get_size()+1, dtype=eso.dtype)
    all_bound_elt_view = all_bound_elt[:-1]
    comm.Allgather(np.array([bound_elt], dtype=eso.dtype), all_bound_elt_view)
    all_bound_elt[-1] = ec_distrib[2]

    eso_gather = np.empty(eso_wo_last.size+1, eso_wo_last.dtype)
    eso_gather[:-1] = eso_wo_last
    # Now search the start of the next rank having data
    j = comm.Get_rank() + 1
    while (all_bound_elt[j] == -1):
      j += 1
    eso_gather[-1] = all_bound_elt[j]

    PT.set_value(eso_n, eso_gather)

    new_ec_distrib = np.copy(ec_distrib)
    new_ec_distrib[0] = eso_gather[0]
    new_ec_distrib[1] = eso_gather[-1]
    new_distrib['ElementConnectivity'] = new_ec_distrib

  else:
    ec_distrib     =     elt_distrib*PT.Element.NVtx(node)
    new_ec_distrib = new_elt_distrib*PT.Element.NVtx(node)

  # > Set CGNS#Distribution node in node
  MT.newDistribution(new_distrib, node)

  # > ElementConnectivity
  ec_n    = PT.get_child_from_name(node, 'ElementConnectivity')
  ec      = PT.get_value(ec_n)
  new_ec  = MTP.block_to_block(ec, ec_distrib, new_ec_distrib, comm)
  PT.set_value(ec_n, new_ec)

  # > ParentElement
  pe_n    = PT.get_child_from_name(node, 'ParentElements')
  if pe_n is not None :
    pe      = PT.get_value(pe_n)
    new_pe  = np.zeros((new_elt_distrib[1]-new_elt_distrib[0], pe.shape[1]), dtype=pe.dtype)
    for ip in range(pe.shape[1]):
      new_pe_tmp  = MTP.block_to_block(pe[:,ip], elt_distrib, new_elt_distrib, comm)
      new_pe[:,ip] = new_pe_tmp
    PT.set_value(pe_n, new_pe)

# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def redistribute_zone(zone, distribution, comm):

  # Get distribution
  vtx_distrib  = MT.getDistribution(zone, "Vertex")[1]
  cell_distrib = MT.getDistribution(zone, "Cell")[1]
  old_distrib = {'Vertex' : vtx_distrib ,
                 'Cell'   : cell_distrib}

  # New distribution
  new_vtx_distrib  = distribution(PT.Zone.n_vtx(zone) , comm)
  new_cell_distrib = distribution(PT.Zone.n_cell(zone), comm)
  new_distrib = {'Vertex' : new_vtx_distrib ,
                 'Cell'   : new_cell_distrib}
  MT.newDistribution(new_distrib, zone)
   

  # > Coords
  grid_coords = PT.get_children_from_label(zone, 'GridCoordinates_t')
  for grid_coord in grid_coords:
    redistribute_data_node(grid_coord, vtx_distrib, new_vtx_distrib, comm)

  # > Elements
  elts = PT.get_children_from_label(zone, 'Elements_t')
  for elt in elts:
    redistribute_elements_node(elt, distribution, comm)


  # > Flow Solutions
  sols = PT.get_children_from_label(zone, 'FlowSolution_t') + PT.get_children_from_label(zone, 'DiscreteData_t')
  for sol in sols:
    if PT.get_child_from_name(sol, 'PointList') is None:
      loc = PT.Subset.GridLocation(sol)
      redistribute_data_node(sol, old_distrib[loc], new_distrib[loc], comm)
    else:
      redistribute_pl_node(sol, distribution, comm)

  # > BCs
  for bc in PT.iter_children_from_predicates(zone, 'ZoneBC_t/BC_t'):
    redistribute_pl_node(bc, distribution, comm)

  # > GCs
  gc_pred = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  for gc in PT.iter_children_from_predicates(zone, ['ZoneGridConnectivity_t', gc_pred]):
    redistribute_pl_node(gc, distribution, comm)

  # > ZoneSubRegion
  zone_subregions = PT.get_children_from_label(zone, 'ZoneSubRegion_t')
  for zone_subregion in zone_subregions:
    # Trick if related to an other node -> add pl
    matching_region_path = PT.getSubregionExtent(zone_subregion, zone)
    if matching_region_path != PT.get_name(zone_subregion):
      PT.add_child(zone_subregion, PT.get_node_from_path(zone, matching_region_path + '/PointList'))
      PT.add_child(zone_subregion, PT.get_node_from_path(zone, matching_region_path + '/PointRange'))
    redistribute_pl_node(zone_subregion, distribution, comm)
    if matching_region_path != PT.get_name(zone_subregion):
      PT.rm_children_from_name(dist_zone_subregion, 'PointList')
      PT.rm_children_from_name(dist_zone_subregion, 'PointRange')
      PT.rm_child(dist_zone_subregion, MT.getDistribution(dist_zone_subregion))

    PT.rm_child(zone, zone_subregion)
    PT.add_child(zone, dist_zone_subregion)
    
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
def redistribute_tree(dist_tree, comm, policy='uniform'):
  """ Redistribute the data of the input tree according to the choosen distribution policy.

  Supported policies are:

  - ``uniform``  : each data array is equally reparted over all the processes
  - ``gather.N`` : all the data are moved on process N
  - ``gather``   : shortcut for ``gather.0``

  In both case, tree structure remains unchanged on all the processes
  (the tree is still a valid distributed tree).
  Input is modified inplace.

  Args:
    dist_tree (CGNSTree) : Distributed tree
    comm      (MPIComm)  : MPI communicator
    policy    (str)      : distribution policy (see above)

  Example:
    .. literalinclude:: snippets/test_algo.py
      :start-after: #redistribute_dist_tree@start
      :end-before: #redistribute_dist_tree@end
      :dedent: 2
  """
  if policy == 'gather':
    policy = 'gather.0'
  policy_split = policy.split('.')
  assert len(policy_split) in [1, 2]

  policy_type = policy_split[0]

  if policy_type == "uniform":
    distribution = par_utils.uniform_distribution
  elif policy_type == "gather":
    assert len(policy_split) == 2
    i_rank = int(policy_split[1])
    assert i_rank < comm.Get_size() 
    distribution = lambda n_elt, comm : par_utils.gathering_distribution(i_rank, n_elt, comm)
  else:
    raise ValueError("Unknown policy for redistribution")

  for zone in PT.iter_all_Zone_t(dist_tree):
    redistribute_zone(zone, distribution, comm)

# ---------------------------------------------------------------------------------------
# =======================================================================================
