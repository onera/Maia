import numpy as np

from maia.utils import np_utils, par_utils
from maia.transfer import protocols as EP


def remove_distributed_ids(distri, ids, comm):
  """
  Delete specified ids from global numbering and renumber the
  ids from 1 to n_elts - n_removed_elts.
  Return an old_id_to_new_id indirection of size dn_elts
  with values -1 at deleted positions.
  """
  GI = EP.GlobalIndexer(distri, ids-1, comm)
  not_ids_local = (GI.access_counts == 0) # Mask is True for unaccessed ids
  
  dn_elts = distri[1] - distri[0]

  n_rmvd_local  = not_ids_local.size - not_ids_local.sum()
  n_rmvd_offset = par_utils.gather_and_shift(n_rmvd_local, comm)

  old_to_new = -1*np.ones(dn_elts, dtype=ids.dtype)
  old_to_new[not_ids_local] = np.arange(dn_elts - n_rmvd_local) + distri[0] - n_rmvd_offset[comm.Get_rank()] + 1
  
  return old_to_new

def merge_distributed_ids(distri, ids, targets, comm, sign_rmvd=False):
  """
  Map some distributed elements (ids) to others (targets) and shift all the numbering,
  in a distributed way.
  ids and targets must be of same size and are distributed arrays.
  Elements should not appear both in ids and targets arrays.
  Return an old_to_new array for all the elements in the distribution.
  If sign_rmvd is True, input ids maps to -target instead of target
  in old_to_new array.
  """

  # Move data to procs holding ids, merging multiple elements

  GI = EP.GlobalIndexer(distri, ids-1, comm)
  ids_local = (GI.access_counts > 0) # True if elts are accessed

  dist_targets = GI.Put(targets)[ids_local]

  # Count the number of elements to be deleted (that is the number of elts received, after merge)
  n_rmvd_local  = ids_local.sum()
  n_rmvd_offset = par_utils.gather_and_shift(n_rmvd_local, comm)

  # Initial old_to_new
  total_ids_size = distri[1]-distri[0]
  unchanged_ids_size = total_ids_size - n_rmvd_local
  old_to_new = np.empty(total_ids_size, dtype=ids.dtype)
  old_to_new[~ids_local] = np.arange(unchanged_ids_size) + distri[0] + 1

  # Shift global : for each index, substract the number of targets removed by preceding ranks
  old_to_new -= n_rmvd_offset[comm.Get_rank()]

  # Now we need to update old_to_new for ids to indicate new indices of targets.
  # Since the new index of target can be on another proc, we do a (fake) BTP to
  # get the data using target numbering
  part_data2 = EP.block_to_part(old_to_new, distri, dist_targets-1, comm)

  marker = -1 if sign_rmvd else 1
  old_to_new[ids_local] = marker * part_data2

  return old_to_new