import numpy as np
from mpi4py import MPI

import Pypdm.Pypdm        as PDM

import maia
from maia.utils import par_utils, np_utils
from maia.utils import vstride as vs

from . import _protocols

from ._protocols import GlobalIndexer, GlobalMultiIndexer, ReduceOp

def _check_dict_keys(data_dict, comm):
  if comm.Get_size() == 0:
    return
  master_keys = comm.bcast(list(data_dict.keys()), 0)
  is_same = list(data_dict.keys()) == master_keys
  if not comm.allreduce(is_same, MPI.LAND):
    raise KeyError("Exchanged data keys must be identical on all ranks")

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
  if np.array_equal(_full_distri_in, _full_distri_out):
    return _protocols.SerialBlockToBlock(_full_distri_in, _full_distri_out)
  else:
    return _protocols.BlockToBlock(_full_distri_in, _full_distri_out, comm)

def BlockToPart(distri, ln_to_gn_list, comm, legacy=False):
  """
  Create a PDM BlockToPart object, with auto gnum conversion
  and extended distribution
  """
  full_distri = auto_expand_distri(distri, comm)
  if legacy:
    _full_distri = maia.utils.as_pdm_gnum(full_distri)
    _ln_to_gn_list  = [maia.utils.as_pdm_gnum(ln_to_gn) for ln_to_gn in ln_to_gn_list]
    return PDM.BlockToPart(_full_distri, comm, _ln_to_gn_list, len(_ln_to_gn_list))
  else:
    if isinstance(ln_to_gn_list, list):
      return GlobalMultiIndexer(full_distri, ln_to_gn_list, comm)
    else:
      return GlobalIndexer(full_distri, ln_to_gn_list, comm)

def PartToBlock(distri, ln_to_gn_list, comm, *, weight=False, keep_multiple=False, legacy=False):
  """
  Create a PDM PartToBlock object, with auto gnum conversion
  and extended distribution
  """
  if distri is not None:
    full_distri = auto_expand_distri(distri, comm)
    _full_distri = maia.utils.as_pdm_gnum(full_distri)
  else:
    assert legacy, "distri=None only supported for legacy version"
    _full_distri = None

  if legacy:
    _ln_to_gn_list  = [maia.utils.as_pdm_gnum(ln_to_gn) for ln_to_gn in ln_to_gn_list]
    
    t_post = 2 if keep_multiple else 1
    pWeight = [np.ones(lngn.size) for lngn in ln_to_gn_list] if weight else None

    return PDM.PartToBlock(comm, _ln_to_gn_list, pWeight=pWeight, partN=len(_ln_to_gn_list),
                          t_distrib=0, t_post=t_post, userDistribution=_full_distri)
  else:
    if isinstance(ln_to_gn_list, list):
      return GlobalMultiIndexer(_full_distri, ln_to_gn_list, comm)
    else:
      return GlobalIndexer(_full_distri, ln_to_gn_list, comm)

def PartToPart(gnum1, gnum2, comm):
  """
  Create a simplified PDM PartToPart object, where the gnum of the two partitioned views
  refer to the same entities in global numbering.
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
    _check_dict_keys(data_in, comm)
    block_data_out = dict()
    for name, field in data_in.items():
      block_data_out[name] = BTB.exchange_field(field)
  else:
    block_data_out = BTB.exchange_field(data_in)

  return block_data_out

def block_to_part(dist_data, distri, ln_to_gn_list, comm, legacy=False):
  """
  Create and exchange using a BlockToPart object.
  Allow single field or dict of fields
  """
  BTP = BlockToPart(distri, ln_to_gn_list, comm, legacy)

  if legacy:
    exch_one = lambda d_field: BTP.exchange_field(d_field)[1]
  else:
    def exch_one(d_field):
      if isinstance(d_field, vs.VStrideArray):
        out = BTP.Take_v((d_field.counts, d_field.values))
        if isinstance(BTP, GlobalIndexer): # out is a single VBuffer
          return vs.from_counts(*out)
        elif isinstance(BTP, GlobalMultiIndexer): # out is a list of VBuffer
          return [vs.from_counts(*vbuff) for vbuff in out]
      else: # Return a Buffer or a list of Buffer
        return BTP.Take(d_field)

  if isinstance(dist_data, dict):
    _check_dict_keys(dist_data, comm)
    part_data = dict()
    for name, d_field in dist_data.items():
      part_data[name] = exch_one(d_field)
  else:
    part_data = exch_one(dist_data)

  return part_data



def part_to_block(part_data, distri, ln_to_gn_list, comm, reduce_func=None, **kwargs):
  """
  Create and exchange using a PartToBlock object.
  Allow single field or dict of fields
  """
  legacy = kwargs.get('legacy', False)

  if legacy: # Legacy mode allow only fixed buff (with reduction)
    kwargs['keep_multiple'] = bool(reduce_func is not None)
    PTB = PartToBlock(distri, ln_to_gn_list, comm, **kwargs)
    if reduce_func is not None:
      def _exchange_one(part_fields):
        p_stride = [np.ones(p_f.size, dtype=np.int32) for p_f in part_fields]
        dist_stride, dist_data = PTB.exchange_field(part_fields, p_stride)
        dist_data = reduce_func(dist_data, dist_stride)
        return dist_data
    else:
      def _exchange_one(part_fields):
        return PTB.exchange_field(part_fields)[1]

  else:
    PTB = PartToBlock(distri, ln_to_gn_list, comm)
    if reduce_func is not None: # Reduce func => fixed buff
      def _exchange_one(part_fields):
        func_to_op = {reduce_sum: ReduceOp.SUM, reduce_min: ReduceOp.MIN, reduce_max: ReduceOp.MAX, reduce_mean:ReduceOp.SUM}
        dist_data = PTB.Put(part_fields, reduce=func_to_op[reduce_func])
        if reduce_func == reduce_mean:
          dist_data /= PTB.access_counts
        return dist_data
    else:
      append = kwargs.get('append', False) or kwargs.get('keep_multiple', False)
      if append: # Append mode => vbuffer
        def _exchange_one(part_fields):
          if isinstance(PTB, GlobalIndexer):
            assert isinstance(part_fields, vs.VStrideArray)
            return vs.from_counts(*PTB.Put_v((part_fields.counts, part_fields.values), extend=True))
          elif isinstance(PTB, GlobalMultiIndexer):
            assert all(isinstance(pf, vs.VStrideArray) for pf in part_fields)
            return vs.from_counts(*PTB.Put_v([(pf.counts, pf.values) for pf in part_fields], extend=True))
      elif isinstance(PTB, GlobalIndexer): # We can guess from input arg
        def _exchange_one(part_fields):
          return PTB.Put_v((part_fields.counts, part_fields.values)) if isinstance(part_fields, vs.VStrideArray) \
            else PTB.Put(part_fields)
      else: # We can not be sure => default to fixed buff
        _exchange_one = lambda part_fields : PTB.Put(part_fields)

  

  if isinstance(part_data, dict):
    _check_dict_keys(part_data, comm)
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
    _check_dict_keys(send_data, comm)
    recv_stride = None
    recv_data = dict()
    for name, field in send_data.items():
      request = PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P, # Point to point communication strategy
                          PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1, # data follows gnum1 layout
                          field,
                          send_stride)
      recv_stride, recv_field = PTP.wait(request)
      recv_data[name] = recv_field
  else:
    request = PTP.iexch(PDM._PDM_MPI_COMM_KIND_P2P,
                        PDM._PDM_PART_TO_PART_DATA_DEF_ORDER_PART1,
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
