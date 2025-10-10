import numpy as np
from mpi4py import MPI
from typing import overload

import Pypdm.Pypdm as PDM

import maia
from maia.typing import *
from maia.utils import vstride as vs
from maia.utils.parallel.utils import auto_expand_distri

from . import _protocols

from ._protocols import (
  ReduceOp,
  GlobalIndexer as _GlobalIndexer,
  GlobalMultiIndexer as _GlobalMultiIndexer
  )

# Type alias to designate a single array or dictionnary of arrays
T = TypeVar('T', bound=np.generic)
BasicDistData = Union[NDArray[T], Dict[str, NDArray[T]]]
BasicPartData = Union[List[NDArray[T]], Dict[str, List[NDArray[T]]]]

DistData = Union[NDArray[T], vs.VStrideArray, Mapping[str, Union[NDArray[T], vs.VStrideArray]]]
SPartData = Union[NDArray[T], vs.VStrideArray, Mapping[str, Union[NDArray[T], vs.VStrideArray]]]
MPartData = Union[List[NDArray[T]], List[vs.VStrideArray], Mapping[str, Union[List[NDArray[T]], List[vs.VStrideArray]]]]


def _check_dict_keys(data_dict: Dict[str, Any], comm: MPIComm) -> None:
  if comm.Get_size() == 0:
    return
  master_keys = comm.bcast(list(data_dict.keys()), 0)
  is_same = list(data_dict.keys()) == master_keys
  if not comm.allreduce(is_same, MPI.LAND):
    raise KeyError("Exchanged data keys must be identical on all ranks")

def BlockToBlock(distri_in: NDArray,
                 distri_out: NDArray,
                 comm: MPIComm):
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


@overload
def GlobalIndexer(distri: NDArray,
                  g_idx: NDArray,
                  comm: MPIComm,
                  *,
                  gnum_offset:int=0) -> _GlobalIndexer: ...
@overload
def GlobalIndexer(distri: NDArray,
                  g_idx: List[NDArray],
                  comm: MPIComm,
                  *,
                  gnum_offset:int=0) -> _GlobalMultiIndexer: ...
def GlobalIndexer(distri: NDArray,
                  g_idx: Union[NDArray, List[NDArray]],
                  comm: MPIComm,
                  *,
                  gnum_offset:int=0) -> Union[_GlobalIndexer, _GlobalMultiIndexer]:
  """
  Helper function creating a Global(Multi)Indexer protocol object, with the following behaviour:

  - Create a GlobalIndexer or a MultiGlobalIndexer depending of g_idx type
  - Auto expend distribution if a partial distribution is used
  - Shift (inplace) g_idx array(s) if gnum_offset is provided
  """
  assert distri is not None
  full_distri = auto_expand_distri(distri, comm)


  if isinstance(g_idx, list):
    if gnum_offset != 0:
      for _g_idx in g_idx:
        _g_idx -= gnum_offset
    GMI = _GlobalMultiIndexer(full_distri, g_idx, comm)
    if gnum_offset != 0:
      for _g_idx in g_idx:
        _g_idx += gnum_offset
    return GMI
  else:
    if gnum_offset != 0:
      g_idx -= gnum_offset
    GI = _GlobalIndexer(full_distri, g_idx, comm)
    if gnum_offset != 0:
      g_idx += gnum_offset
    return GI

def PartToPart(gnum1: List[NDArray],
               gnum2: List[NDArray],
               comm: MPIComm):
  """
  Create a simplified PDM PartToPart object, where the gnum of the two partitioned views
  refer to the same entities in global numbering.
  """
  _part1_lngn  = [maia.utils.as_pdm_gnum(gnum) for gnum in gnum1]
  _part2_lngn  = [maia.utils.as_pdm_gnum(gnum) for gnum in gnum2]

  _part1_to_part2_idx = [np.arange(gnum.size+1, dtype=np.int32) for gnum in gnum1]

  return PDM.PartToPart(comm, _part1_lngn, _part2_lngn, _part1_to_part2_idx, _part1_lngn)

@overload
def block_to_block(data_in: NDArray[T],
                   distri_in: NDArray,
                   distri_out: NDArray,
                   comm: MPIComm) -> NDArray[T]: ...
@overload
def block_to_block(data_in: Dict[str, NDArray[T]],
                   distri_in: NDArray,
                   distri_out: NDArray,
                   comm: MPIComm) -> Dict[str, NDArray[T]]: ...

def block_to_block(data_in: BasicDistData,
                   distri_in: NDArray,
                   distri_out: NDArray,
                   comm: MPIComm) -> BasicDistData:
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



@overload
def block_to_part(dist_data: vs.VStrideArray,
                  distri: NDArray,
                  ln_to_gn_list: NDArray,
                  comm: MPIComm,
                  **kwargs) -> vs.VStrideArray: ...
@overload
def block_to_part(dist_data: NDArray[T],
                  distri: NDArray,
                  ln_to_gn_list: NDArray,
                  comm: MPIComm,
                  **kwargs) -> NDArray[T]: ...
@overload
def block_to_part(dist_data: Mapping[str, Union[NDArray[T], vs.VStrideArray]],
                  distri: NDArray,
                  ln_to_gn_list: NDArray,
                  comm: MPIComm,
                  **kwargs) -> Mapping[str, Union[NDArray[T], vs.VStrideArray]]: ...
@overload
def block_to_part(dist_data: DistData,
                  distri: NDArray,
                  ln_to_gn_list: NDArray,
                  comm: MPIComm,
                  **kwargs) -> SPartData: ...

@overload
def block_to_part(dist_data: NDArray[T],
                  distri: NDArray,
                  ln_to_gn_list: List[NDArray],
                  comm: MPIComm,
                  **kwargs) -> List[NDArray[T]]: ...
@overload
def block_to_part(dist_data: vs.VStrideArray,
                  distri: NDArray,
                  ln_to_gn_list: List[NDArray],
                  comm: MPIComm,
                  **kwargs) -> List[vs.VStrideArray]: ...
@overload
def block_to_part(dist_data: Mapping[str, Union[NDArray[T], vs.VStrideArray]],
                  distri: NDArray,
                  ln_to_gn_list: List[NDArray],
                  comm: MPIComm,
                  **kwargs) -> Mapping[str, Union[List[NDArray[T]], List[vs.VStrideArray]]]: ...
@overload
def block_to_part(dist_data: DistData,
                  distri: NDArray,
                  ln_to_gn_list: List[NDArray],
                  comm: MPIComm,
                  **kwargs) -> MPartData: ...

def block_to_part(dist_data: DistData,
                  distri: NDArray,
                  ln_to_gn_list: Union[NDArray, List[NDArray]],
                  comm: MPIComm,
                  **kwargs) -> Union[SPartData, MPartData]:
  """ A wrapper creating a GlobalIndexer and using it for a Take exchange.

  This wrapper allows as input data buffer, variable buffer, or dictionnairies containing
  a combination of theses objects. The returned data matches the input kind, stored as list
  if a GlobalMultiIndexer is used or as single variable if a GlobalIndexer is used.
  """
  GI = GlobalIndexer(distri, ln_to_gn_list, comm, **kwargs)

  def exch_one(d_field):
    if isinstance(d_field, vs.VStrideArray):
      out = GI.Take_v((d_field.counts, d_field.values))
      if isinstance(GI, _GlobalIndexer): # out is a single VBuffer
        return vs.from_counts(*out)
      elif isinstance(GI, _GlobalMultiIndexer): # out is a list of VBuffer
        return [vs.from_counts(*vbuff) for vbuff in out]
    else: # Return a Buffer or a list of Buffer
      return GI.Take(d_field)

  if isinstance(dist_data, dict):
    _check_dict_keys(dist_data, comm)
    part_data = dict()
    for name, d_field in dist_data.items():
      part_data[name] = exch_one(d_field)
  else:
    part_data = exch_one(dist_data)

  return part_data

@overload
def part_to_block(part_data: NDArray[T],
                  distri: NDArray,
                  ln_to_gn_list: NDArray,
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> NDArray[T]: ...
@overload
def part_to_block(part_data: vs.VStrideArray,
                  distri: NDArray,
                  ln_to_gn_list: NDArray,
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> vs.VStrideArray: ...
@overload
def part_to_block(part_data: Mapping[str, Union[NDArray[T], vs.VStrideArray]],
                  distri: NDArray,
                  ln_to_gn_list: NDArray,
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> Mapping[str, Union[NDArray[T], vs.VStrideArray]]: ...
@overload
def part_to_block(part_data: SPartData,
                  distri: NDArray,
                  ln_to_gn_list: NDArray,
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> DistData: ...

@overload
def part_to_block(part_data: List[NDArray[T]],
                  distri: NDArray,
                  ln_to_gn_list: List[NDArray],
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> NDArray[T]: ...
@overload
def part_to_block(part_data: List[vs.VStrideArray],
                  distri: NDArray,
                  ln_to_gn_list: List[NDArray],
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> vs.VStrideArray: ...
@overload
def part_to_block(part_data: Mapping[str, Union[List[NDArray[T]], List[vs.VStrideArray]]],
                  distri: NDArray,
                  ln_to_gn_list: List[NDArray],
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> Mapping[str, Union[NDArray[T], vs.VStrideArray]]: ...
@overload
def part_to_block(part_data: MPartData,
                  distri: NDArray,
                  ln_to_gn_list: List[NDArray],
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> DistData: ...

def part_to_block(part_data: Union[SPartData, MPartData],
                  distri: NDArray,
                  ln_to_gn_list: Union[NDArray, List[NDArray]],
                  comm: MPIComm,
                  reduce_op:Optional[ReduceOp]=None,
                  extend:bool=False,
                  **kwargs: Any) -> DistData:
  """ A wrapper creating a GlobalIndexer and using it for a Put exchange.

  This wrapper allows as input data buffer, variable buffer, or dictionnairies containing
  a combination of theses objects, stored as list if a GlobalMultiIndexer is used or as single
  variable if a GlobalIndexer is used. The returned data matches the input kind.
  """

  GI = GlobalIndexer(distri, ln_to_gn_list, comm, **kwargs)
  if reduce_op is not None: # Reduce func => fixed buff
    assert not extend
    def _exchange_one(part_fields):
      return GI.Put(part_fields, reduce=reduce_op)
  else:
    if extend:
      assert reduce_op is None
      def _exchange_one(part_fields):
        if isinstance(GI, _GlobalIndexer):
          assert isinstance(part_fields, vs.VStrideArray)
          return vs.from_counts(*GI.Put_v((part_fields.counts, part_fields.values), extend=True))
        elif isinstance(GI, _GlobalMultiIndexer):
          assert all(isinstance(pf, vs.VStrideArray) for pf in part_fields)
          return vs.from_counts(*GI.Put_v([(pf.counts, pf.values) for pf in part_fields], extend=True))
    elif isinstance(GI, _GlobalIndexer): # We can guess from input arg
      def _exchange_one(part_fields):
        return GI.Put_v((part_fields.counts, part_fields.values)) if isinstance(part_fields, vs.VStrideArray) \
          else GI.Put(part_fields)
    else: # We can not be sure => default to fixed buff
      _exchange_one = lambda part_fields : GI.Put(part_fields)

  

  if isinstance(part_data, dict):
    _check_dict_keys(part_data, comm)
    dist_data = {name: _exchange_one(p_field) for name, p_field in part_data.items()}
  else:
    dist_data = _exchange_one(part_data)  
  return dist_data

@overload
def part_to_part(send_data: List[NDArray[T]],
                 gnum1: List[NDArray],
                 gnum2: List[NDArray],
                 comm: MPIComm) -> List[NDArray[T]]: ...
@overload
def part_to_part(send_data: Dict[str, List[NDArray[T]]],
                 gnum1: List[NDArray],
                 gnum2: List[NDArray],
                 comm: MPIComm) -> Dict[str, List[NDArray[T]]]: ...

def part_to_part(send_data: BasicPartData,
                 gnum1: List[NDArray],
                 gnum2: List[NDArray],
                 comm: MPIComm) -> BasicPartData:
  """
  Create and exchange using a PartToPart object with basic "id-to-id" indirection.
  Allow single field or dict of fields
  """
  _, recv_data = part_to_part_strided(1, send_data, gnum1, gnum2, comm)
  return recv_data

@overload
def part_to_part_strided(send_stride: Union[int, List[NDArray]],
                         send_data: List[NDArray[T]],
                         gnum1: List[NDArray],
                         gnum2: List[NDArray],
                         comm: MPIComm) -> Tuple[NDArray, List[NDArray[T]]]: ...
@overload
def part_to_part_strided(send_stride: Union[int, List[NDArray]],
                         send_data: Dict[str,List[NDArray[T]]],
                         gnum1: List[NDArray],
                         gnum2: List[NDArray],
                         comm: MPIComm) -> Tuple[NDArray, Dict[str,List[NDArray[T]]]]: ...

def part_to_part_strided(send_stride: Union[int, List[NDArray]],
                         send_data: BasicPartData,
                         gnum1: List[NDArray],
                         gnum2: List[NDArray],
                         comm: MPIComm) -> Tuple[NDArray, BasicPartData]:
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
  return recv_stride, recv_data #type:ignore[return-value] #(return None for debug)
