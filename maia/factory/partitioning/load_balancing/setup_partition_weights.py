import numpy as np
import logging as LOG

import maia.pytree        as PT
import maia.pytree.sids   as SIDS

from .single_zone_balancing import homogeneous_repart
from .multi_zone_balancing  import balance_with_uniform_weights, balance_with_non_uniform_weights, karmarkar_karp

def compute_nosplit_weights(tree, comm):
  """Compute a zone_to_parts repartition without splitting the blocks.

  The initial blocks will be simply distributed over the available processes,
  minimizing the total number of cells affected to a proc. This leads to a poor load
  balancing and possibly to procs having no partitions at all.

  Args:
    tree (CGNSTree)  : (Minimal) distributed tree : only zone names and sizes are needed
    comm (MPI.Comm)  : MPI Communicator
  Returns:
    dict: ``zone_to_parts`` dictionnary expected by :func:`partition_dist_tree`

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #compute_nosplit_weights@start
        :end-before: #compute_nosplit_weights@end
        :dedent: 2
  """
  # Note : this is the multiway number partitioning problem
  # https://en.wikipedia.org/wiki/Multiway_number_partitioning
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  zone_paths = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  n_blocks   = len(zone_paths)
  nb_elts    = np.array([SIDS.Zone.n_cell(PT.get_node_from_path(tree, zone_path)) for zone_path in zone_paths])
  if min(nb_elts) == 0: # This is point cloud, without cells, so use vtx to compute balancing
    nb_elts    = np.array([SIDS.Zone.n_vtx(PT.get_node_from_path(tree, zone_path)) for zone_path in zone_paths])

  # Early return in simplified cases
  repart_per_zone = np.zeros((n_blocks, n_rank), dtype=int) #Full array for statistics
  if n_rank == 1:
    repart_per_zone[:,0] = nb_elts
  elif n_rank >= n_blocks:
    repart_per_zone[0:n_blocks,0:n_blocks] = np.diag(nb_elts)
  else:
    subset_indices = karmarkar_karp(nb_elts, n_rank)
    for j, rank_idx in enumerate(subset_indices):
      repart_per_zone[rank_idx, j] = nb_elts[rank_idx]

  zone_to_weights = {zone_path: [1.] for j, zone_path in enumerate(zone_paths) \
          if repart_per_zone[j][i_rank] > 0}

  return zone_to_weights

def npart_per_zone(tree, comm, n_part=1):
  """Compute a basic zone_to_parts repartition.

  Each process request n_part partitions on each original zone (n_part can differ
  for each proc).
  The weights of all the parts produced from a given zone are homogeneous
  and equal to the number of cells in the zone divided by the total
  number of partitions requested for this zone.

  Args:
    tree (CGNSTree)  : (Minimal) distributed tree : only zone names and sizes are needed
    comm (MPI.Comm)  : MPI Communicator
    n_part (int,optional) : Number of partitions to produce on each zone by the proc.
      Defaults to 1.
  Returns:
    dict: ``zone_to_parts`` dictionnary expected by :func:`partition_dist_tree`

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #compute_regular_weights@start
        :end-before: #compute_regular_weights@end
        :dedent: 2
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  zone_paths = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  nb_elmt_per_zone = {zone_path : SIDS.Zone.n_cell(PT.get_node_from_path(tree, zone_path)) for zone_path in zone_paths}
  if min(nb_elmt_per_zone.values()) == 0: # This is point cloud, without cells, so use vtx to compute balancing
    nb_elmt_per_zone = {zone_path : SIDS.Zone.n_vtx(PT.get_node_from_path(tree, zone_path)) for zone_path in zone_paths}

  n_part_np = np.asarray(n_part, dtype=np.int32)

  n_part_shift = np.empty(n_rank+1, dtype=np.int32)
  n_part_shift[0] = 0
  n_part_shift_view = n_part_shift[1:]
  comm.Allgather(n_part_np, n_part_shift_view)

  n_part_distri = np.cumsum(n_part_shift)
  n_part_tot    = n_part_distri[n_rank]

  start_idx = n_part_distri[i_rank]
  end_idx   = n_part_distri[i_rank+1]
  repart_per_zone = {zone_path : homogeneous_repart(n_cell, n_part_tot)[start_idx:end_idx]
      for zone_path, n_cell in nb_elmt_per_zone.items()}

  zone_to_weights = {zone_path : [k/nb_elmt_per_zone[zone_path] for k in repart]
      for zone_path, repart in repart_per_zone.items()}

  return zone_to_weights

def balance_multizone_tree(tree, comm, only_uniform=False):
  """Compute a well balanced zone_to_parts repartition.

  Each process request or not partitions with heterogeneous weight on each 
  original zone such that:

  - the computational load is well balanced, ie the total number of
    cells per process is nearly equal,
  - the number of splits within a given zone is minimized,
  - produced partitions are not too small.

  Note:
    Heterogeneous weights are not managed by ptscotch. Use parmetis as graph_part_tool
    for partitioning if repartition was computed with this function, or set optional
    argument only_uniform to True.

  Args:
    tree (CGNSTree)  : (Minimal) distributed tree : only zone names and sizes are needed
    comm (MPI.Comm)  : MPI Communicator
    only_uniform (bool, optional) : If true, an alternative balancing method is used
      in order to request homogeneous weights, but load balance is less equilibrated.
      Default to False.

  Returns:
    dict: ``zone_to_parts`` dictionnary expected by :func:`partition_dist_tree`

  Example:
      .. literalinclude:: snippets/test_factory.py
        :start-after: #compute_balanced_weights@start
        :end-before: #compute_balanced_weights@end
        :dedent: 2
  """

  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  zone_paths = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  nb_elmt_per_zone = {zone_path : SIDS.Zone.n_cell(PT.get_node_from_path(tree, zone_path)) for zone_path in zone_paths}
  if min(nb_elmt_per_zone.values()) == 0: # This is point cloud, without cells, so use vtx to compute balancing
      nb_elmt_per_zone = {zone_path : SIDS.Zone.n_vtx(PT.get_node_from_path(tree, zone_path)) for zone_path in zone_paths}

  repart_per_zone = balance_with_uniform_weights(nb_elmt_per_zone, n_rank) if only_uniform \
               else balance_with_non_uniform_weights(nb_elmt_per_zone, n_rank)


  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Verbose
  n_part     = [int(n_elmts[i_rank] > 0) for n_elmts in repart_per_zone.values()]
  tn_part    = [n_rank - n_elmts.count(0) for n_elmts in repart_per_zone.values()]
  proc_elmts = [n_elmts[i_rank] for n_elmts in repart_per_zone.values()]
  LOG.info(' '*2 + '-'*20 + " REPARTITION FOR RANK {0:04d} ".format(i_rank) + '-'*19)
  LOG.info(' '*4 + "    zoneName  zoneSize :  procElem nPart TnPart %ofZone %ofProc")
  if sum(proc_elmts) == 0:
    LOG.warning(f"Proc {i_rank} was not affected to any zone")
  for izone, zone_path in enumerate(repart_per_zone.keys()):
    zone_pc = np.around(100*proc_elmts[izone]/nb_elmt_per_zone[zone_path])
    try:
      proc_pc = np.around(100*proc_elmts[izone]/sum(proc_elmts))
    except ZeroDivisionError:
      proc_pc = 0.
    LOG.info(' '*4 + "{0:>12.12} {1:9d} : {2:9d} {3:>5} {4:>6}  {5:>6}  {6:>6}".format(
      zone_path, nb_elmt_per_zone[zone_path], proc_elmts[izone], n_part[izone], tn_part[izone], zone_pc, proc_pc))
  LOG.info('')
  tot_pc = np.around(100*sum(proc_elmts)/sum(nb_elmt_per_zone.values()))
  LOG.info(' '*4 + "       Total {1:9d} : {2:9d} {3:>5} {4:>6}  {5:>6}  {6:>6}".format(
    zone_path, sum(nb_elmt_per_zone.values()), sum(proc_elmts), sum(n_part), sum(tn_part), tot_pc, 100))
  LOG.info(' '*2 + "------------------------------------------------------------------ " )
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Convert to expected format and return
  zone_to_weights = {zone_path: [repart[i_rank]/nb_elmt_per_zone[zone_path]] \
      for zone_path, repart in repart_per_zone.items() if repart[i_rank] > 0}
  return zone_to_weights
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
