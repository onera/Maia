import numpy as np
import logging as LOG

import Converter.Internal as I

import maia.sids.sids as SIDS
from .single_zone_balancing import homogeneous_repart
from .multi_zone_balancing  import balance_with_uniform_weights, balance_with_non_uniform_weights
from .                      import balancing_quality

def npart_per_zone(tree, comm, n_part=1):
  """
  Basic repartition of the different zones of the mesh on the available procs :
  each proc take n_part partitions of each zone. n_part can be different for each proc.
  The weights are homogeneous ie equal, for each zone, to the nb of cells in the zone divided
  by the number of partitions.
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  nb_elmt_per_zone = {I.getName(zone) : SIDS.Zone.n_cell(zone) for zone in I.getZones(tree)}

  n_part_np = np.asarray(n_part, dtype=np.int32)

  n_part_shift = np.empty(n_rank+1, dtype=np.int32)
  n_part_shift[0] = 0
  n_part_shift_view = n_part_shift[1:]
  comm.Allgather(n_part_np, n_part_shift_view)

  n_part_distri = np.cumsum(n_part_shift)
  n_part_tot    = n_part_distri[n_rank]

  start_idx = n_part_distri[i_rank]
  end_idx   = n_part_distri[i_rank+1]
  repart_per_zone = {zone : homogeneous_repart(n_cell, n_part_tot)[start_idx:end_idx]
      for zone, n_cell in nb_elmt_per_zone.items()}

  zone_to_weights = {zone : [k/nb_elmt_per_zone[zone] for k in repart]
      for zone, repart in repart_per_zone.items()}

  return zone_to_weights

def balance_multizone_tree(tree, comm, only_uniform=False):
  """
  Repart the different zones of the mesh on the available procs in
  order to have a well balanced computational load (ie : total number
  of cells per proc must be equal).
  In addition, we try to minimize the number of splits within a given zone
  and we avoid to produce small partitions.

  Args:
      tree (pyTree)      : A (minimal) pyTree : only zone names and sizes
                           are needed
      comm (MPI.Comm)    : MPI Communicator (from mpi4py)
      only_uniform (bool): If true, the partition weights for a given zone are
                           homogeneous (= 1/nbProcForThisZone)
  Returns
      zone_to_weights(dict) : For each proc, dictionnary associating the name of
                              the zones (key:string) to the partitioning weights
                              attributed to this proc (value:list). Empty list means
                              that the proc is not concerned by this zone.
  """

  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  nb_elmt_per_zone = {I.getName(zone) : SIDS.Zone.n_cell(zone) for zone in I.getZones(tree)}

  repart_per_zone = balance_with_uniform_weights(nb_elmt_per_zone, n_rank) if only_uniform \
               else balance_with_non_uniform_weights(nb_elmt_per_zone, n_rank)

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  repart_per_zone_array = np.array([repart for repart in repart_per_zone.values()])
  balancing_quality.compute_balance_and_splits(repart_per_zone_array)
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Verbose
  n_part     = [int(n_elmts[i_rank] > 0) for n_elmts in repart_per_zone.values()]
  tn_part    = [n_rank - n_elmts.count(0) for n_elmts in repart_per_zone.values()]
  proc_elmts = [n_elmts[i_rank] for n_elmts in repart_per_zone.values()]
  LOG.info(' '*2 + '-'*20 + " REPARTITION FOR RANK {0:04d} ".format(i_rank) + '-'*19)
  LOG.info(' '*4 + "    zoneName  zoneSize :  procElem nPart TnPart %ofZone %ofProc")
  for izone, zone in enumerate(repart_per_zone.keys()):
    zone_pc = np.around(100*proc_elmts[izone]/nb_elmt_per_zone[zone])
    proc_pc = np.around(100*proc_elmts[izone]/sum(proc_elmts))
    LOG.info(' '*4 + "{0:>12.12} {1:9d} : {2:9d} {3:>5} {4:>6}  {5:>6}  {6:>6}".format(
      zone, nb_elmt_per_zone[zone], proc_elmts[izone], n_part[izone], tn_part[izone], zone_pc, proc_pc))
  LOG.info('')
  tot_pc = np.around(100*sum(proc_elmts)/sum(nb_elmt_per_zone.values()))
  LOG.info(' '*4 + "       Total {1:9d} : {2:9d} {3:>5} {4:>6}  {5:>6}  {6:>6}".format(
    zone, sum(nb_elmt_per_zone.values()), sum(proc_elmts), sum(n_part), sum(tn_part), tot_pc, 100))
  LOG.info(' '*2 + "------------------------------------------------------------------ " )
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  # > Convert to expected format and return
  zone_to_weights = {zone: [repart[i_rank]/nb_elmt_per_zone[zone]] if repart[i_rank] > 0 else []
      for zone, repart in repart_per_zone.items()}
  return zone_to_weights
  # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
