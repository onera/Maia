import numpy as np

import Pypdm.Pypdm        as PDM

import maia
from maia.utils import par_utils

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
  else:
    PTB = PartToBlock(distri, ln_to_gn_list, comm, **kwargs)

  if isinstance(part_data, dict):
    dist_data = dict()
    for name, p_field in part_data.items():
      p_stride = []
      for p_f in p_field:
        p_stride.append(np.ones(p_f.size,dtype=np.int32))
      dist_stride, dist_data_tmp = PTB.exchange_field(p_field, p_stride)
      if reduce_func is not None:
        dist_data_tmp = reduce_func(dist_data_tmp, dist_stride)
      dist_data[name] = dist_data_tmp
  else:
    p_stride = []
    for p_f in part_data:
      p_stride.append(np.ones(p_f.size,dtype=np.int32))
    dist_stride, dist_data = PTB.exchange_field(part_data, p_stride)
    if reduce_func is not None:
      dist_data = reduce_func(dist_data, dist_stride)
  return dist_data

def compute_indices(dist_stride):
  nbElem      = np.shape(dist_stride)[0]
  indices     = np.empty(nbElem,dtype=dist_stride.dtype)
  indices[0]  = 0
  indices[1:] = np.cumsum(dist_stride[:-1])
  return indices

def reduce_sum(dist_data,dist_stride):
  indices = compute_indices(dist_stride)
  return np.add.reduceat(dist_data,indices)

def reduce_max(dist_data,dist_stride):
  indices = compute_indices(dist_stride)
  return np.maximum.reduceat(dist_data,indices)

def reduce_min(dist_data,dist_stride):
  indices = compute_indices(dist_stride)
  return np.minimum.reduceat(dist_data,indices)

def reduce_mean(dist_data,dist_stride):
  indices = compute_indices(dist_stride)
  return np.divide(np.add.reduceat(dist_data,indices, dtype=dist_data.dtype),dist_stride)

def reduce_weighted_mean(dist_data,dist_stride,dist_weight_data,dist_weight_stride):
  assert np.all(dist_stride == dist_weight_stride)
  assert np.shape(dist_data)[0] == np.shape(dist_weight_data)[0]
  indices = compute_indices(dist_stride)
  dist_weighted_data = np.multiply(dist_data,dist_weight_data, dtype=dist_data.dtype)
  return np.divide(np.add.reduceat(dist_weighted_data,indices, dtype=dist_weighted_data.dtype),
                    np.add.reduceat(dist_weight_data,indices, dtype=dist_weight_data.dtype))
