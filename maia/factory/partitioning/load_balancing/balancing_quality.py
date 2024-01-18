from mpi4py import MPI
import numpy as np
from math import ceil

from maia.utils import logging as mlog

def compute_balance_and_splits_seq(repart_per_zone, display=False):
  """
  Evaluate the quality of the input repartition. Input repartition
  must be a 2d array of size n_zone * n_rank : cell [i][j] stores
  the (int) number of cells affected to the jth rank on the ith zone.
  """
  assert repart_per_zone.ndim == 2
  n_zone, n_rank = repart_per_zone.shape

  max_part_size = np.max(repart_per_zone)
  min_part_size = np.min(np.ma.masked_equal(repart_per_zone, 0, copy=False)) #min, ignoring 0
  n_cuts        = np.count_nonzero(repart_per_zone)

  proc_load = np.sum(repart_per_zone, axis=0)
  min_load  = np.min(proc_load)
  max_load  = np.max(proc_load)

  exact_ideal_load = sum(proc_load) / float(n_rank)
  ideal_load       = ceil(exact_ideal_load)

  imbalance       = proc_load - ideal_load
  worse_imbalance = np.max(imbalance)

  rms   = np.linalg.norm(imbalance) / n_rank
  rmscp = np.linalg.norm(imbalance/ideal_load) / n_rank

  if display:
    mlog.stat("  ---> Mean   size : {0}".format(ideal_load))
    mlog.stat("  ---> rMini  size : {0}".format(min_load))
    mlog.stat("  ---> rMaxi  size : {0}".format(max_load))
    mlog.stat("  ---> rms         : {0}".format(rms))
    mlog.stat("  ---> rmscp       : {0}".format(rmscp))
    mlog.stat("  ---> worse delta : {0} ({1:.2f}%)".format(worse_imbalance, 100.*worse_imbalance/ideal_load))
    mlog.stat("  ---> n_cuts      : {0}".format(n_cuts))

  return ideal_load, min_load, max_load, rms, rmscp, n_cuts, min_part_size, max_part_size

def compute_balance_and_splits(repart_per_zone, comm, display=False):
  """
  Evaluate the quality of the input repartition. Input repartition
  must be a 1d array of size n_zone : cell [i] stores
  the (int) number of cells affected to the ith zone.

  This is a distributed version of compute_balance_and_splits_seq
  """
  
  n_rank = comm.Get_size()

  max_part_size = comm.allreduce(np.max(repart_per_zone), MPI.MAX)
  min_part_size = comm.allreduce(np.min(np.ma.masked_equal(repart_per_zone, 0, copy=False)), MPI.MIN) #min, ignoring 0
  n_cuts        = comm.allreduce(np.count_nonzero(repart_per_zone), MPI.SUM)

  proc_load = np.empty(n_rank, dtype=int)
  loc_load = repart_per_zone.sum() * np.ones(1, int)
  comm.Allgather(loc_load, proc_load)
  min_load  = np.min(proc_load)
  max_load  = np.max(proc_load)

  exact_ideal_load = sum(proc_load) / float(n_rank)
  ideal_load       = ceil(exact_ideal_load)

  imbalance       = proc_load - ideal_load
  worse_imbalance = np.max(imbalance)

  rms   = np.linalg.norm(imbalance) / n_rank
  rmscp = np.linalg.norm(imbalance/ideal_load) / n_rank

  if display:
    mlog.stat("  ---> Mean   size : {0}".format(ideal_load))
    mlog.stat("  ---> rMini  size : {0}".format(min_load))
    mlog.stat("  ---> rMaxi  size : {0}".format(max_load))
    mlog.stat("  ---> rms         : {0}".format(rms))
    mlog.stat("  ---> rmscp       : {0}".format(rmscp))
    mlog.stat("  ---> worse delta : {0} ({1:.2f}%)".format(worse_imbalance, 100.*worse_imbalance/ideal_load))
    mlog.stat("  ---> n_cuts      : {0}".format(n_cuts))

  return ideal_load, min_load, max_load, rms, rmscp, n_cuts, min_part_size, max_part_size
