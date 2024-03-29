import pickle
import hashlib
import numpy as np

from maia.transfer import protocols as EP
from maia.utils    import np_utils, par_utils, py_utils

import Pypdm.Pypdm as PDM

from mpi4py import MPI

def dist_set_difference(ids, others, comm):
  """ Return the list of elements that belong to ids array but are absent from
  all the other array from others list
  ids = numpy array
  others = list of numpy arrays
  """
  ln_to_gn = [ids] + others
  
  PTB = EP.PartToBlock(None, ln_to_gn, comm, keep_multiple=True)

  part_data   = [np.ones(ids.size, dtype=bool)] + [np.zeros(other.size, dtype=bool) for other in others]
  part_stride = [np.ones(pdata.size, dtype=np.int32) for pdata in part_data]

  dist_stride, dist_data = PTB.exchange_field(part_data, part_stride)
  dist_data = np.logical_and.reduceat(dist_data, np_utils.sizes_to_indices(dist_stride)[:-1])

  selected = PTB.getBlockGnumCopy()[dist_data]
  distri_in  = par_utils.gather_and_shift(selected.size, comm, dtype=np.int32)  
  distri_out = par_utils.uniform_distribution(distri_in[-1], comm)


  # Si on veut se caller sur les points d'entrée
  # tt = EP.block_to_part(dist_data, PTB.getDistributionCopy(), [ids], comm)
  # distri = PTB.getDistributionCopy()
  # BTP = EP.BlockToPart(distri, [ids], comm)
  # d_stride = np.zeros(distri[comm.Get_rank()+1] - distri[comm.Get_rank()], np.int32)
  # d_stride[PTB.getBlockGnumCopy() - distri[comm.Get_rank()] - 1] = 1

  # ts, tt = BTP.exchange_field(dist_data, d_stride)

  # dist_data = EP.part_to_block(part_data, None, ln_to_gn, comm, reduce_func=reduce_prod)

  # Sur chaque rank, on a une liste d'id (qui étaient sur ids ou pas) et un flag valant 1
  # si faut les garder
  # Il reste à supprimer et rééquilibrer
  selected = EP.block_to_block(selected, distri_in, distri_out, comm)
  return selected


def compute_gnum(objects, comm, serialize=pickle.dumps):
  """ Attribute to each object of objects list a global id.
  Globals ids span from 1 to "number of unique objects across the procs"
  If a object appears several times in objects lists, it will receive
  the same global id.
  """
  n_rank = comm.Get_size()
  key_mod = 2**30

  # Serialize and compute hash of each object
  byte_objects = [serialize(obj) for obj in objects]

  #hash_key = [int(hashlib.sha1(b).hexdigest(), 16) % key_mod for b in byte_objects]
  iterable = (int(hashlib.sha1(b).hexdigest(), 16) % key_mod for b in byte_objects)
  hash_key = np.fromiter(iterable, dtype=PDM.npy_pdm_gnum_dtype)

  # Compute distribution, then search managing proc of each object
  # Also prepare the order to be used to sort send buffer according to rank ordering
  distri = PDM.compute_weighted_distribution([hash_key+1], [np.ones(len(hash_key))], comm)
  dest = np.searchsorted(distri, hash_key, side='right') - 1
  sort_idx = np.argsort(dest)

  # Exchange number of object to be managed by each proc
  send_n_items = np.zeros(n_rank, np.int32) #Number of objects to send to each rank
  recv_n_items = np.empty(n_rank, np.int32) #Number of object to recv from each rank
  np.add.at(send_n_items, dest, 1)

  comm.Alltoall(send_n_items, recv_n_items)

  send_items_idx = np_utils.sizes_to_indices(send_n_items)
  recv_items_idx = np_utils.sizes_to_indices(recv_n_items)

  # Exchange size of each object to be managed by the dest. rank
  send_n_bytes = np.array([len(byte_objects[i]) for i in sort_idx], np.int32) #Ordered for all_to_all
  recv_n_bytes = np.empty(sum(recv_n_items), np.int32)

  comm.Alltoallv((send_n_bytes, send_n_items), (recv_n_bytes, recv_n_items))

  # Temporary for all to all sizes : number of bytes going to each process
  #_sum_send_n_bytes = np.add.reduceat(send_n_bytes, send_items_idx[:-1]) # Fail if a rank should have 0 values
  #_sum_recv_n_bytes = np.add.reduceat(recv_n_bytes, recv_items_idx[:-1])
  _sum_send_n_bytes = [sum(send_n_bytes[send_items_idx[i]:send_items_idx[i+1]]) for i in range(n_rank)]
  _sum_recv_n_bytes = [sum(recv_n_bytes[recv_items_idx[i]:recv_items_idx[i+1]]) for i in range(n_rank)]

  # Now send bytes buffer to the dest rank
  send_buff = b''.join([byte_objects[i] for i in sort_idx])
  recv_buff = bytearray(sum(_sum_recv_n_bytes))
  comm.Alltoallv((send_buff, _sum_send_n_bytes), (recv_buff, _sum_recv_n_bytes))

  # Compute hash of received objects and compute index locally for the zipped sequence (hash, key)
  #   Hash are needed to ensure parallelism independant results.
  #   Key are needed to break ties between same hashes.
  # Since operators < and == are implemented for (int, bytes) tuples we can use the sorting function

  #Slice recv buffer using recv_byte lens
  recv_bytes_idx = np_utils.sizes_to_indices(recv_n_bytes)
  recv_bytes = [recv_buff[recv_bytes_idx[i]:recv_bytes_idx[i+1]] for i in range(recv_n_bytes.size)]
  recv_keys = [int(hashlib.sha1(b).hexdigest(), 16) % key_mod for b in recv_bytes]

  dist_gnum = py_utils.unique_idx(list(zip(recv_keys, recv_bytes)))
  dist_gnum = np.array(dist_gnum, np.int32)

  # Once we have local gnum, we need to shift it with a scan to have global gnum
  n_gnum = np.max(dist_gnum, initial=-1) + 1 # If dist_gnum is empty, we want 0 for n_gnum
  offset = par_utils.gather_and_shift(n_gnum, comm)[comm.Get_rank()]
  dist_gnum += offset + 1

  # Send back gnum to original rank, by switching send_n_items and recv_n_items
  # Then we need to "unsort" the received gnum which is sorted as the send buffer
  sorted_gnum = np.empty(len(byte_objects), np.int32)
  comm.Alltoallv((dist_gnum, recv_n_items), (sorted_gnum, send_n_items))
  gnum = sorted_gnum[np.argsort(sort_idx)]

  return gnum

class DistSorter:
  """ Argsort-like algorithm for distributed arrays.
  Class should be instanciated with an array 'key' (of int. values); then any
  arrays send to sort will be reorder to match key sorting order
  """
  def __init__(self, key, comm):
    self.ptb = EP.PartToBlock(None, [key], comm, weight=[np.ones(key.size, np.int32)])

  def sort(self, array):
    _, sorted = self.ptb.exchange_field([array])
    return sorted


def is_unique_strided_serialized(array, stride, comm):
  """
  For a distributed cst strided array (eg. a connectivity), return a local bool array indicating
  for each element if it appears only once (w/ considering ordering).
  Note: this function is slow because of serialization in `compute_gnum(...)`, should be better
  with PDM.from_nuplet_parent, but bugged for now. See after maia v1.3 deployment.
  """
  n_elt = array.size//stride
  
  strided_array = array.reshape(n_elt, stride)
  strided_array = np.sort(strided_array, axis=1)
  gnum = compute_gnum(strided_array, comm)
  
  # gen_gnum = PDM.GlobalNumbering(3, 1, 1, 0., comm)
  # gnum = gen_gnum.set_parents_nuplet(stride)
  # gnum = gen_gnum.set_from_parent(0, array.astype(PDM.npy_pdm_gnum_dtype))
  # gen_gnum.compute()
  # gnum = gen_gnum.get(0)

  unique_gnum, idx, count = np.unique(gnum, return_index=True, return_counts=True)
  max_gnum = comm.allreduce(np.max(unique_gnum), op=MPI.MAX)
  distri = par_utils.uniform_distribution(max_gnum, comm)

  dist_data = EP.part_to_block([count], distri, [unique_gnum], comm, reduce_func=EP.reduce_sum)
  is_unique = np.zeros(distri[1]-distri[0], dtype=bool)
  is_unique[dist_data==1] = True
  part_data = EP.block_to_part(is_unique, distri, [unique_gnum], comm)
  
  mask = np.zeros(n_elt, dtype=bool)
  ids  = idx[part_data[0]]
  mask[ids] = True

  return mask


def is_unique_strided(array, stride, comm):
  """
  For a distributed cst strided array (eg. a connectivity), return a local bool array indicating
  for each element if it appears only once (w/ considering ordering).
  """
  n_elt = array.size//stride
  distri = par_utils.dn_to_distribution(n_elt, comm)
  src_dist_gnum = np.arange(distri[0], distri[1], dtype=PDM.npy_pdm_gnum_dtype)+1
  
  array_idx = stride*np.arange(n_elt+1, dtype=np.int32)
  array_key = np.add.reduceat(array, array_idx[:-1])

  weights = np.ones(n_elt, float)
  ptb = EP.PartToBlock(None, [array_key], comm, weight=[weights], keep_multiple=True)
  cst_stride = np.ones(n_elt, np.int32)

  # Origin is not mandatory for TETRA because we just want the TRI ids at the end
  _, origin = ptb.exchange_field([src_dist_gnum], part_stride=[  cst_stride])
  _, tmp_ec = ptb.exchange_field([array]        , part_stride=[3*cst_stride])
  part_mask = np_utils.is_unique_strided(tmp_ec, 3, method='hash')

  # Retrieve mask on initial distribution
  mask = EP.part_to_block([part_mask], distri, [origin], comm)
  
  return mask


def gnum_isin(src, tgt, comm):
  """
  For a distributed src array of gnum, return a distributed bool array indicating
  for each element if it appears in a distributed tgt gnum array.
  """
  PTP   = EP.PartToPart([tgt], [src], comm)
  isin  = np.zeros(src.size, dtype=bool)
  isin[PTP.get_referenced_lnum2()[0]-1] = True

  return isin