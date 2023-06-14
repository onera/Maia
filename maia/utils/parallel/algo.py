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

