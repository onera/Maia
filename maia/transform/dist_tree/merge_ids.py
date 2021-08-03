import numpy as np

import Pypdm.Pypdm        as PDM
from maia import npy_pdm_gnum_dtype as pdm_dtype

from maia.utils.parallel import utils as par_utils
from maia.tree_exchange.dist_to_part import data_exchange as BTP
from maia.tree_exchange.part_to_dist import data_exchange as PTB

def merge_distributed_ids(distri, ids, targets, comm, sign_rmvd=False):
  """
  Map some distributed elements (ids) to others (targets) and shift all the numbering,
  in a distributed way.
  ids and targets must be of same size and are distributed arrays
  Return an old_to_new array for all the elements in the distribution
  If sign_rmvd is True, input ids maps to -target instead of target
  in old_to_new array
  """

  distri = distri.astype(pdm_dtype)
  # Move data to procs holding ids, merging multiple elements
  part_data = {'Targets' : [targets]}
  pdm_distrib = par_utils.partial_to_full_distribution(distri, comm)

  PTB = PDM.PartToBlock(comm, [ids.astype(pdm_dtype)], pWeight=None, partN=1,
                        t_distrib=0, t_post=1, t_stride=0, userDistribution=pdm_distrib)
  dist_ids  = PTB.getBlockGnumCopy()

  dist_data = dict()
  PTB.PartToBlock_Exchange(dist_data, part_data)

  # Count the number of elements to be deleted (that is the number of elts received, after merge)
  n_rmvd_local  = len(dist_ids)
  n_rmvd_offset = par_utils.gather_and_shift(n_rmvd_local, comm)

  # Initial old_to_new
  total_ids_size = distri[1]-distri[0]
  old_to_new = np.empty(total_ids_size, dtype=ids.dtype)
  ids_local = dist_ids - distri[0] - 1
  p = np.ones(total_ids_size, dtype=bool)
  p[ids_local] = False
  unchanged_ids_size = total_ids_size - ids_local.size
  old_to_new[p] = np.arange(unchanged_ids_size) + distri[0] + 1

  # Shift global : for each index, substract the number of targets removed by preceding ranks
  old_to_new -= n_rmvd_offset[comm.Get_rank()]

  # Now we need to update old_to_new for ids to indicate new indices of targets.
  # Since the new index of target can be on another proc, we do a (fake) BTP to
  # get the data using target numbering
  dist_data2 = {'OldToNew' : old_to_new}
  part_data2 = BTP.dist_to_part(distri, dist_data2, [dist_data['Targets'].astype(pdm_dtype)], comm)

  marker = -1 if sign_rmvd else 1
  old_to_new[ids_local] = marker * part_data2['OldToNew'][0]

  return old_to_new
