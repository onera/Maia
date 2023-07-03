import numpy as np

from maia.transfer import protocols as EP
from maia.utils    import np_utils, par_utils

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
  # print(tt)

  # dist_data = EP.part_to_block(part_data, None, ln_to_gn, comm, reduce_func=reduce_prod)

  # Sur chaque rank, on a une liste d'id (qui étaient sur ids ou pas) et un flag valant 1
  # si faut les garder
  # Il reste à supprimer et rééquilibrer
  selected = EP.block_to_block(selected, distri_in, distri_out, comm)
  return selected


import pickle
import hashlib
def compute_gnum(objects, comm):
  """ Attribute to each object of objects list a global id.
  Globals ids span from 1 to "number of unique objects across the procs"
  If a object appears several times in objects lists, it will receive
  the same global id.
  """

  byte_objects = [pickle.dumps(obj) for obj in objects]
  #byte_objects = [obj.encode() for obj in objects] # Easier debug

  hash_key = [int(hashlib.sha1(b).hexdigest(), 16) % 2048 for b in byte_objects]

  # TODO : get distribution w/ PTB     #PDM_distrib_weight
  PTB = EP.PartToBlock(None, [np.array(hash_key)+1], comm, weight=[np.ones(len(hash_key))])
  distri = PTB.getDistributionCopy()
  if comm.Get_rank() == 0:
    print("Distribution", distri)

  dest = np.searchsorted(distri, hash_key, side='right') -1
  print(comm.rank, "Dest", dest)

  # Exchange number of object to be managed by each proc
  send_n_items = np.zeros(comm.Get_size(), np.int32) #Number of objects to send to each rank
  np.add.at(send_n_items, dest, 1)
  dest_count_idx = np_utils.sizes_to_indices(send_n_items)

  recv_n_items = np.empty(comm.Get_size(), np.int32)
  comm.Alltoall(send_n_items, recv_n_items)

  print("Hash rank", comm.rank, hash_key, "dest are", dest)
  send_n_items_idx = np_utils.sizes_to_indices(send_n_items)
  recv_n_items_idx = np_utils.sizes_to_indices(recv_n_items)

  sort_idx = np.argsort(dest, kind='stable')

  # Prepare send buffer, orderer from dest rank
  send_buff = b''.join([byte_objects[i] for i in sort_idx])

  send_byte_lens     = [len(byte_objects[i]) for i in sort_idx]

  # Exchange size of objects to be managed by each proc
  recv_byte_lens = np.empty(sum(recv_n_items), np.int32)
  comm.Alltoallv((np.array(send_byte_lens, np.int32), send_n_items), (recv_byte_lens, recv_n_items))

  print("rank" , comm.rank, "will send sizes", send_byte_lens, "with nb / rank", send_n_items)

  #_sum_send_byte_lens = np.add.reduceat(send_byte_lens, send_n_items_idx[:-1])
  #_sum_recv_byte_lens = np.add.reduceat(recv_byte_lens, recv_n_items_idx[:-1])
  _sum_send_byte_lens = [sum(send_byte_lens[send_n_items_idx[i]:send_n_items_idx[i+1]]) for i in range(comm.Get_size())]
  _sum_recv_byte_lens = [sum(recv_byte_lens[recv_n_items_idx[i]:recv_n_items_idx[i+1]]) for i in range(comm.Get_size())]

  # Exchange buffer
  recv_buff = bytearray(sum(_sum_recv_byte_lens))
  comm.Alltoallv((send_buff, _sum_send_byte_lens), (recv_buff, _sum_recv_byte_lens))

  #Slice recv buffer using byte lens
  recv_byte_lens_idx = np_utils.sizes_to_indices(recv_byte_lens)
  recv_bytes = [recv_buff[recv_byte_lens_idx[i]:recv_byte_lens_idx[i+1]] for i in range(recv_byte_lens_idx.size-1)]
  recv_keys = [int(hashlib.sha1(b).hexdigest(), 16) % 2048 for b in recv_bytes]

  #print(comm.rank, recv_bytes)

  _indices = unique_idx_sort_keys(recv_keys, [bytes(ba) for ba in recv_bytes]) #Todo : change underlying algo to avoid conversion

  offset = par_utils.gather_and_shift(max(_indices)+1, comm)
  indices = [i+offset[comm.Get_rank()] for i in _indices]
  #print(comm.rank, 'Sorted indices', _indices, '-->', indices)

  # print(comm.rank, 'Bytes', recv_bytes, '--> gnum', indices)

  # Send back indices to original rank, 
  # Data arrive in proc order so same order than buffsend ; because we sorted buffsend we must 
  # put back in input order
  # comm.Alltoallv((t1, counts), (recv, (rcounts, rdispl)))
  orig_indices = np.empty(len(byte_objects), np.int32)
  comm.Alltoallv((np.array(indices, np.int32), recv_n_items), (orig_indices, send_n_items))
  #print(comm.rank, "Back to initial indices", orig_indices)

  ordered_gnum = orig_indices[np.argsort(sort_idx)]
  return ordered_gnum

def unique_idx_sort_keys(keys, seq):
  if len(seq) == 0:
    return []


  zipped = list(zip(keys, seq))
  idx = sorted(range(len(keys)), key=zipped.__getitem__)

  out = [-1 for i in range(len(seq))]
  id = 0
  last = zipped[idx[0]]
  for i in idx:
    if zipped[i] != last:
      last = zipped[i]
      id += 1
    out[i] = id
  return out
  

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

