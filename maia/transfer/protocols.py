import numpy as np

import Pypdm.Pypdm        as PDM

import maia
from maia.utils import par_utils, np_utils

def auto_expand_distri(distri, comm):
  """ Return a full distribution from a full or partial distribution """
  if distri.size == 3 and comm.Get_size() != 2:
    # Distri is partial
    return par_utils.partial_to_full_distribution(distri, comm)
  if distri.size == 3 and comm.Get_size() == 2:
    # This is the corner case, but rank 0 always have [0, s1, s1+s2]
    return comm.bcast(distri, root=0)
  else:
    #Distri is already full
    return distri

def BlockToBlock(distri_in, distri_out, comm):
  """
  Create a PDM BlockToBlock object, with auto gnum conversion
  and extended distribution
  """
  full_distri_in  = auto_expand_distri(distri_in, comm)
  full_distri_out = auto_expand_distri(distri_out, comm)
  _full_distri_in  = maia.utils.as_pdm_gnum(full_distri_in)
  _full_distri_out = maia.utils.as_pdm_gnum(full_distri_out)
  return PDM.BlockToBlock(_full_distri_in, _full_distri_out, comm)

def BlockToPart(distri, ln_to_gn_list, comm):
  """
  Create a PDM BlockToPart object, with auto gnum conversion
  and extended distribution
  """
  full_distri = auto_expand_distri(distri, comm)
  _full_distri = maia.utils.as_pdm_gnum(full_distri)
  _ln_to_gn_list  = [maia.utils.as_pdm_gnum(ln_to_gn) for ln_to_gn in ln_to_gn_list]
  return PDM.BlockToPart(_full_distri, comm, _ln_to_gn_list, len(_ln_to_gn_list))

def PartToBlock(distri, ln_to_gn_list, comm, *, weight=False, keep_multiple=False):
  """
  Create a PDM PartToBlock object, with auto gnum conversion
  and extended distribution
  """
  if distri is not None:
    full_distri = auto_expand_distri(distri, comm)
    _full_distri = maia.utils.as_pdm_gnum(full_distri)
  else:
    _full_distri = None
  _ln_to_gn_list  = [maia.utils.as_pdm_gnum(ln_to_gn) for ln_to_gn in ln_to_gn_list]
  
  t_post = 2 if keep_multiple else 1
  pWeight = [np.ones(lngn.size) for lngn in ln_to_gn_list] if weight else None

  return PDM.PartToBlock(comm, _ln_to_gn_list, pWeight=pWeight, partN=len(_ln_to_gn_list),
                         t_distrib=0, t_post=t_post, userDistribution=_full_distri)

def PartToPart(gnum1, gnum2, comm):
  """
  Create a PDM PartToPart object, with auto gnum conversion and "id-to-id"
  indirection
  """
  _part1_lngn  = [maia.utils.as_pdm_gnum(gnum) for gnum in gnum1]
  _part2_lngn  = [maia.utils.as_pdm_gnum(gnum) for gnum in gnum2]

  _part1_to_part2_idx = [np.arange(gnum.size+1, dtype=np.int32) for gnum in gnum1]

  return PDM.PartToPart(comm, _part1_lngn, _part2_lngn, _part1_to_part2_idx, _part1_lngn)


def block_to_block(data_in, distri_in, distri_out, comm):
  """
  Create and exchange using a BlockToBlock object.
  Allow single field or dict of fields
  """
  BTB = BlockToBlock(distri_in, distri_out, comm)

  if isinstance(data_in, dict):
    block_data_out = dict()
    for name, field in data_in.items():
      block_data_out[name] = BTB.exchange_field(field)
  else:
    block_data_out = BTB.exchange_field(data_in)

  return block_data_out

def block_to_part(dist_data, distri, ln_to_gn_list, comm):
  """
  Create and exchange using a BlockToPart object.
  Allow single field or dict of fields
  """
  BTP = BlockToPart(distri, ln_to_gn_list, comm)

  if isinstance(dist_data, dict):
    part_data = dict()
    for name, d_field in dist_data.items():
      part_data[name] = BTP.exchange_field(d_field)[1]
  else:
    _, part_data = BTP.exchange_field(dist_data)

  return part_data

def block_to_part_strided(dist_stride, dist_data, distri, ln_to_gn_list, comm):
  """
  Create and exchange using a BlockToPart object with variable stride.
  Allow single field or dict of fields
  """
  BTP = BlockToPart(distri, ln_to_gn_list, comm)

  if isinstance(dist_data, dict):
    part_data = dict()
    for name, d_field in dist_data.items():
      part_stride, _part_data = BTP.exchange_field(d_field, dist_stride)
      part_data[name] = _part_data
  else:
    part_stride, part_data = BTP.exchange_field(dist_data, dist_stride)

  return part_stride, part_data

def part_to_block(part_data, distri, ln_to_gn_list, comm, reduce_func=None, **kwargs):
  """
  Create and exchange using a PartToBlock object.
  Allow single field or dict of fields
  """
  if reduce_func is not None:
    PTB = PartToBlock(distri, ln_to_gn_list, comm, keep_multiple=True, **kwargs)
    def _exchange_one(part_fields):
      p_stride = [np.ones(p_f.size, dtype=np.int32) for p_f in part_fields]
      dist_stride, dist_data = PTB.exchange_field(part_fields, p_stride)
      dist_data = reduce_func(dist_data, dist_stride)
      return dist_data
  else:
    PTB = PartToBlock(distri, ln_to_gn_list, comm, **kwargs)
    def _exchange_one(part_fields):
      _, dist_data = PTB.exchange_field(part_fields)
      return dist_data

  if isinstance(part_data, dict):
    dist_data = {name: _exchange_one(p_field) for name, p_field in part_data.items()}
  else:
    dist_data = _exchange_one(part_data)  
  return dist_data

def part_to_part(send_data, gnum1, gnum2, comm):
  """
  Create and exchange using a PartToPart object with basic "id-to-id" indirection.
  Allow single field or dict of fields
  """
  _, recv_data = part_to_part_strided(1, send_data, gnum1, gnum2, comm)
  return recv_data

def part_to_part_strided(send_stride, send_data, gnum1, gnum2, comm):
  """
  Create and exchange using a PartToPart object with basic "id-to-id" indirection.
  Allow single field or dict of fields
  """
  PTP = PartToPart(gnum1, gnum2, comm)

  if isinstance(send_data, dict):
    recv_stride = None
    recv_data = dict()
    for name, field in send_data.items():
      request = PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                          PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                          field,
                          send_stride)
      recv_stride, recv_field = PTP.wait(request)
      recv_data[name] = recv_field
  else:
    request = PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                        PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1_TO_PART2,
                        send_data,
                        send_stride)
    recv_stride, recv_data = PTP.wait(request)
  return recv_stride, recv_data


def reduce_sum(dist_data,dist_stride):
  """
  Function that sum all data sharing the same global number
  """
  indices = np_utils.sizes_to_indices(dist_stride)[:-1]
  return np.add.reduceat(dist_data, indices)

def reduce_max(dist_data,dist_stride):
  """
  Function that return the maximum of all data sharing the same global number
  """
  indices = np_utils.sizes_to_indices(dist_stride)[:-1]
  return np.maximum.reduceat(dist_data, indices)

def reduce_min(dist_data,dist_stride):
  """
  Function that return the minimum of all data sharing the same global number
  """
  indices = np_utils.sizes_to_indices(dist_stride)[:-1]
  return np.minimum.reduceat(dist_data, indices)

def reduce_mean(dist_data,dist_stride):
  """
  Function that return the mean of all data sharing the same global number
  """
  indices = np_utils.sizes_to_indices(dist_stride)[:-1]
  return np.add.reduceat(dist_data, indices) / dist_stride
