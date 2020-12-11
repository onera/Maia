import numpy as np
from math import ceil

def compute_balance_and_splits(repart_per_zone):
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

  print(' '*2 + "-------------------- REPARTITION STATISTICS ---------------------- " )
  print(' '*4 + "---> Mean   size : {0} (rounded from {1:.1f})".format(ideal_load, exact_ideal_load))
  print(' '*4 + "---> rMini  size : {0}".format(min_load))
  print(' '*4 + "---> rMaxi  size : {0}".format(max_load))
  print(' '*4 + "---> rms         : {0}".format(rms))
  print(' '*4 + "---> rmscp       : {0}".format(rmscp))
  print(' '*4 + "---> worse delta : {0} ({1:.2f}%)".format(worse_imbalance, 100.*worse_imbalance/ideal_load))
  print(' '*4 + "---> n_cuts      : {0}".format(n_cuts))
  print(' '*2 + "------------------------------------------------------------------ " )

  return ideal_load, min_load, max_load, rms, rmscp, n_cuts, min_part_size, max_part_size
