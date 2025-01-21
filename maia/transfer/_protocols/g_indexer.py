from mpi4py import MPI
import numpy as np
import pickle

from cmaia.utils import layouts

# Typing
from typing       import List, Tuple, Any
from numpy.typing import NDArray
NDArrayInt = NDArray[np.integer]
Buffer = NDArray[Any] 
VBuffer = Tuple[NDArrayInt, Buffer]  


def counting_sort(array, n_bins):
  return layouts.counting_sort(array, n_bins)

def counting_sort_mult(arrays, n_bins):
  return layouts.counting_sort_mult(arrays, n_bins)

def take_strided(a_counts, a_val, indices, out):
  """
  An equivalent to numpy.take (a[ind]), but with strided values in a.
  a is described by strides + values, eg
  a_counts = [3,1,2]
  a_val    = [10,11,12, 100, 1000, 1001] (3 values, then 1 value, then 2 values)

  indices is the list of idx to extract; for each indices, the whole "grap" of strided
  values will be extracted
  take_strided(a_counts, a_val, [2,0]) = [1000, 1001,  10,11,12]
  """
  layouts.take_stridedDI(a_counts, a_val, indices, out)

def put_strided(a, a_count, indices, read_counts, read):
  """
  Write in an array strided array (a, a_count) at provided indices
  from an input data (read, read_count).
  If an index occurs multiple times in indices array, it erase the previously written
  value. A check is performed on counts to write only compatible data
  """
  layouts.put_strided(a, a_count, indices, read_counts, read)

class GlobalMultiIndexer:
  """
  This is a generalization of :class:`~maia.transfer.protocols.GlobalIndexer` where each
  process can request access to several list of global indices.

  We thus need to introduce the additional notations, which are local for each rank:

  - :math:`N` number of global indices list provided (equal to ``len(g_idx_l)``)
  - :math:`pn_k`: for each index list :math:`k`, number of accessed indices (equal to ``len(g_idx_l[k])``)

  All the methods described in :class:`~maia.transfer.protocols.GlobalIndexer` are available,
  but the arguments related to the accessed indices (*ie* the output of ``take`` methods, and the
  input of ``put`` methods) are now lists of size :math:`N`.

  Note that in particular, a given process can set :math:`N=0`; in this case, empty lists should be used
  whenever a list is expected. 

  """

  def __init__(self, distri: NDArrayInt, g_idx_l: List[NDArrayInt], comm: MPI.Comm):
    """ Generalization of :func:`GlobalIndexer.__init__` for multi index access.

    Args:
      distri (integer array of size :math:`s+1`) : distribution of the collection
      g_idx_l (N integer arrays of size :math:`pn_k`) : accessed global indices
      comm (MPIComm) : communicator
    """
    assert distri.size == comm.Get_size() + 1

    # Binary search : for each gnum, find the corresponding rank in distribution
    owning_rank_l = [np.searchsorted(distri, g_idx, side='right') - 1 for g_idx in g_idx_l]

    ok = all( (0 <= owning_rank).all() and (owning_rank < comm.Get_size()).all() for owning_rank in owning_rank_l)
    if not comm.allreduce(ok, op=MPI.LAND):
      raise IndexError("Some idx not in distri")

    # Compute the sorting order to put lngn in increasing rank order
    sorting_idx_l, send_counts = counting_sort_mult(owning_rank_l, comm.Get_size())

    # Now send gnum to their managing rank, and move it to local indices
    recv_counts = np.empty(comm.Get_size(), int)
    comm.Alltoall(send_counts, recv_counts)

    send_data = np.empty(send_counts.sum(), int)
    recv_data = np.empty(recv_counts.sum(), int)

    for g_idx, sorting_idx in zip(g_idx_l, sorting_idx_l):
      send_data[sorting_idx] = g_idx
    comm.Alltoallv((send_data, send_counts), (recv_data, recv_counts))
    recv_data -= distri[comm.Get_rank()]

    self.comm = comm
    self.dn = distri[comm.Get_rank()+1] - distri[comm.Get_rank()]   # Number of managed indices
    self.pn = [g_idx.size for g_idx in g_idx_l]                     # Number of accessed indices
    self.dist_counts = recv_counts  # Number of managed indices accessed by each rank (with multiplicity)
    self.part_counts = send_counts  # Number of accessed indices managed by each rank (with multiplicity)
    self.dist_select_idx = recv_data   # Selection order of managed data to put it in MPI order
    self.part_write_pos  = sorting_idx_l # Position where accessed data should be put to have it in MPI order

    self._empty_dist = (np.diff(distri)==0).any() # True if at least one rank has dn == 0
    self._empty_part = None                       # True if at least one rank has len(g_idx) == 0

  @property
  def empty_dist(self) -> bool:
    return self._empty_dist
  @property
  def empty_part(self) -> bool:
    if self._empty_part is None:
      self._empty_part = self.comm.allreduce(len(self.pn) == 0, MPI.LOR)
    return self._empty_part


  def take(self, dist_data: List) -> List[List]:
    """ Generalization of :func:`GlobalIndexer.take` for multi index access.

    Args:
      dist_data (list of size :math:`dn`) : section of the distributed data
    Returns:
      :math:`N` list of size :math:`pn_k` : for each index list,
      values extracted at the requested indices
    """
    pickelized = [pickle.dumps(data) for data in dist_data]
    counts_in  = np.array([len(p) for p in pickelized], dtype=int)

    buff_in = np.frombuffer(b''.join(pickelized), dtype=np.int8)
    
    data_out_l = self.Take_v((counts_in, buff_in))
    
    res = list()
    for (counts_out, buff_out) in data_out_l:
      out = []
      r_start = 0
      for size in counts_out:
        out.append(pickle.loads(buff_out[r_start:r_start+size].tobytes()))
        r_start += size
      res.append(out)
    return res

  def put(self, local_data_l: List[List]) -> List:
    """ Generalization of :func:`GlobalIndexer.put` for multi index access.

    Args:
      local_data_l (:math:`N` list of size :math:`pn_k`) : for each index list, data to write
        at each accessed index
    Returns:
      list of size :math:`dn`: output distributed data
    """
    _data_in_l = list()
    for data_in in local_data_l:
      pickelized = [pickle.dumps(data) for data in data_in]
      counts_in = np.array([len(p) for p in pickelized], dtype=int)
      buff_in   = np.frombuffer(b''.join(pickelized), dtype=np.int8)
      _data_in_l.append((counts_in, buff_in))
    
    counts_out, buff_out = self.Put_v(_data_in_l)
    
    out = []
    r_start = 0
    for size in counts_out:
      if size != 0:
        out.append(pickle.loads(buff_out[r_start:r_start+size].tobytes()))
      else:
        out.append(None)
      r_start += size
    return out

  def _Take(self, dist_data: Buffer, local_data_l: List[Buffer], count=1):
    """ Generalization of :func:`GlobalIndexer.Take_into` for multi index access.

    Args:
      dist_data  (buffer) : section of the distributed data
      local_data_l (list of :math:`N` buffer) : preallocated buffers to store extracted values
        corresponding to each index list
      count (int) : scalar value of :math:`c`. Defaults to 1.
    """
    assert len(local_data_l) == len(self.pn)

    if dist_data.size - count*self.dn != 0:
      raise ValueError(f"Invalid size of input distributed buffer (expected {count*self.dn}, got {dist_data.size})")
    for ipart, (data_out, pn) in enumerate(zip(local_data_l, self.pn)):
      if data_out.size - count*pn != 0:
        raise ValueError(f"Invalid size of output local buffer n°{ipart} (expected {count*pn}, got {data_out.size})")


    send_buff = np.empty(count*self.dist_select_idx.size, dist_data.dtype)
    recv_buff = np.empty(count*sum(self.pn),              dist_data.dtype)

    if count == 1:
      send_buff[:] = dist_data[self.dist_select_idx]
    else:
      pull_idx = count*self.dist_select_idx
      for j in range(count):
        send_buff[j::count] = dist_data[pull_idx+j]

    self.comm.Alltoallv((send_buff, count*self.dist_counts),
                        (recv_buff, count*self.part_counts))

    # Data has been received in owning proc order : 'unsort' it to recover lngn ordering
    for data_out, part_write_pos in zip(local_data_l, self.part_write_pos):
      if count == 1:
        data_out[:] = recv_buff[part_write_pos]
      else:
        put_idx = count*part_write_pos
        for j in range(count):
          data_out[j::count] = recv_buff[put_idx+j]

  def _Put(self, local_data_l: List[Buffer] , dist_data: Buffer, count=1):
    """ Generalization of :func:`GlobalIndexer.Put_into` for multi index access.

    Args:
      local_data_l (list of :math:`N` buffer) : for each index list, data to write at each accessed index
      dist_data  (buffer) : preallocated buffer to store distributed data
      count (int) : scalar value of :math:`c`. Defaults to 1.
    """
    assert len(local_data_l) == len(self.pn)

    if dist_data.size - count*self.dn != 0:
      raise ValueError(f"Invalid size of output distributed buffer (expected {count*self.dn}, got {dist_data.size})")
    for ipart, (data_in, pn) in enumerate(zip(local_data_l, self.pn)):
      if data_in.size - count*pn != 0:
        raise ValueError(f"Invalid size of input local buffer n°{ipart} (expected {count*pn}, got {data_in.size})")


    send_buff = np.empty(count*sum([write_pos.size for write_pos in self.part_write_pos]), dtype=dist_data.dtype)
    recv_buff = np.empty(count*self.dist_counts.sum(),  dtype=dist_data.dtype)

    for data_in, part_write_pos in zip(local_data_l, self.part_write_pos):
      if count == 1:
        send_buff[part_write_pos] = data_in
      else:
        pull_idx = count*part_write_pos
        for j in range(count):
          send_buff[pull_idx+j] = data_in[j::count]

    self.comm.Alltoallv((send_buff, count*self.part_counts), 
                        (recv_buff, count*self.dist_counts))

    if count == 1:
      dist_data[self.dist_select_idx] = recv_buff
    else:
      put_idx = count*self.dist_select_idx
      for j in range(count):
        dist_data[put_idx+j] = recv_buff[j::count]

  def Take(self, dist_data: Buffer, local_data_l: List[Buffer]=None, count=1) -> List[Buffer]:
    """ Generalization of :func:`GlobalIndexer.Take` for multi index access.

    Args:
      dist_data (buffer of size :math:`c*dn`) : section of the distributed data
      local_data_l (list of :math:`N` buffer, optional) : preallocated buffers to store extracted values
        corresponding to each index list, or None
      count (int) : scalar value of :math:`c`. Defaults to 1.
    Returns:
      :math:`N` buffer of size :math:`c*pn_k`: for each index list,
      values extracted at the requested indices
    """
    if local_data_l is None:
      local_data_l = [np.empty(count*pn, dist_data.dtype) for pn in self.pn]
    self._Take(dist_data, local_data_l, count)
    return local_data_l

  def Put(self, local_data_l: List[Buffer], dist_data:Buffer=None, count=1) -> Buffer:
    """ Generalization of :func:`GlobalIndexer.Put` for multi index access.

    Args:
      local_data_l (:math:`N` buffer of size :math:`c*pn_k`) : for each index list,
        data to write at each accessed index
      dist_data (buffer, optional) : preallocated buffer to store distributed data or None
      count (int) : scalar value of :math:`c`. Defaults to 1.
    Returns:
      buffer of size :math:`c*dn`: output distributed data
    """
    assert len(local_data_l) == len(self.pn)

    if dist_data is None:
      dtype  = local_data_l[0].dtype.str if len(local_data_l) > 0 else ''
      if self.empty_part:
        dtype  = self.comm.allreduce(dtype,  MPI.MAX)
      dist_data = np.empty(count*self.dn, dtype)

    self._Put(local_data_l, dist_data, count)
    return dist_data




  def Take_v(self, dist_data: VBuffer, local_data_l: List[VBuffer]=None) -> List[VBuffer]:
    """ Generalization of :func:`GlobalIndexer.Take_v` for multi index access.

    Args:
      dist_data (variable buffer): section of the distributed data, ie pair of values
        (**dist_counts** (*np array of* :math:`dn` *int*), **dist_buff** (*buffer*))
      local_data_l (list of N variable buffer, optional): list of preallocated buffer to store
        extracted values or None
    Returns:
      list of N variable buffer: for each index list, data extracted as tuple of values \
        (**local_counts** (*np array of* :math:`pn_k` *int*), **local_buff** (*buffer*))
    """


    counts_in, buff_in = dist_data

    if not (isinstance(counts_in, np.ndarray) and np.issubdtype(counts_in.dtype, np.integer)):
      raise ValueError(f"Invalid kind of counts input")
    if counts_in.size != self.dn:
      raise ValueError(f"Invalid size of input counts (expected {self.dn}, got {counts_in.size})")
    if buff_in.size - counts_in.sum() != 0:
      raise ValueError(f"Invalid size of input distributed buffer (expected {counts_in.sum()}, got {buff_in.size})")

    _counts_in  = counts_in[self.dist_select_idx]

    if local_data_l is None:
      # Case 1: allocate output (local) data
      # Exchange counts_in. We do not call Take because we need the intermediate layout
      _counts_out = np.empty(sum(self.pn), _counts_in.dtype)
      self.comm.Alltoallv((_counts_in, self.dist_counts), (_counts_out, self.part_counts))
      local_data_l = list()
      for part_write_pos in self.part_write_pos:
        counts_out = _counts_out[part_write_pos]
        buff_out   = np.empty(counts_out.sum(), buff_in.dtype)
        local_data_l.append((counts_out, buff_out))
    else:
      # Case 2: output (local) data already allocatated : do some checks and recompute _counts_out
      _counts_out = np.empty(sum([data[0].size for data in local_data_l]), _counts_in.dtype)
      for i, local_data in enumerate(local_data_l):
        counts_out, buff_out = local_data
        if not (isinstance(counts_out, np.ndarray) and np.issubdtype(counts_out.dtype, np.integer)):
          raise ValueError(f"Invalid kind of counts input")
        if counts_out.size != self.pn[i]:
          raise ValueError(f"Invalid size of output counts (expected {self.pn[i]}, got {counts_out.size})")
        if buff_out.size - counts_out.sum() != 0:
          raise ValueError(f"Invalid size of input distributed buffer (expected {counts_out.sum()}, got {buff_out.size})")
        _counts_out[self.part_write_pos[i]] = counts_out


    # Count the actual number of items to send/recv, using stride array
    # (this is the partial sum of portion of the stride array related to the given rank)
    send_counts = np.empty(self.comm.Get_size(), int)
    recv_counts = np.empty(self.comm.Get_size(), int)
    idx_send = idx_recv = 0

    for i in range(self.comm.Get_size()):
      send_counts[i] = _counts_in [idx_send:idx_send+self.dist_counts[i]].sum()
      recv_counts[i] = _counts_out[idx_recv:idx_recv+self.part_counts[i]].sum()
      idx_send += self.dist_counts[i]
      idx_recv += self.part_counts[i]
    
    send_buff = np.empty(send_counts.sum(), buff_in.dtype)
    take_strided(counts_in, buff_in, self.dist_select_idx, send_buff)

    # Exchange data buffer
    recv_buff = np.empty(recv_counts.sum(), buff_in.dtype)
    self.comm.Alltoallv((send_buff, send_counts), (recv_buff, recv_counts))

    # Post treat recv buffer (data arrive in mpi layout, put it in requested layout)
    for local_data, part_write_pos in zip(local_data_l, self.part_write_pos):
      take_strided(_counts_out, recv_buff, part_write_pos, local_data[1])

    return local_data_l

  def Put_v(self, local_data_l: List[VBuffer], dist_data: VBuffer=None) -> VBuffer:
    """ Generalization of :func:`GlobalIndexer.Put_v` for multi index access.

    Args:
      local_data_l (list of N var. buffer): for each index list, values to write as pair \
        (**local_counts** (*np array of* :math:`pn_k` *int*), **local_buff** (*buffer*))
      dist_data (variable buffer, optional) : preallocated buffer to store distributed data or None
    Returns:
      variable buffer: output distributed data, returned as pair of values \
        (**dist_counts** (*np array of* :math:`dn` *int*), **dist_buff** (*buffer*))
    """
    # Note for later: extension to keep_multiple is not so complicated : 
    # compute counts_out with np.add.at(counts_out, self.dist_select_idx, _counts_out) instead of np.put
    # (because np.put is responsible of 'keeping last value')
    # Then update last put_strided to remove the check on the size : loop becomes
    # 
    # std::vector<int> offset(write_counts.size(), 0);
    # for (int i=0; i < write_idx.size(); ++i) {
    #   int idx = _write_idx[i];
    #   int w_start = write_displs[idx] + offset[idx];
    #   int w_end   = write_displs[idx+1];
    #
    #   std::copy_n(_read_buff + s_data*r_idx,
    #               _read_counts[i]*s_data,
    #               _write_buff + s_data*w_start);
    #
    #   offset[idx] += _read_counts[i];
    #   r_idx       += _read_counts[i];
    # } 

    # Variable stride

    counts_in_l = [data_in[0] for data_in in local_data_l]
    buff_in_l   = [data_in[1] for data_in in local_data_l]
    assert len(local_data_l) == len(self.pn)
    assert all(counts_in.size == pn for counts_in,pn in zip(counts_in_l, self.pn))
    assert all(np.issubdtype(counts_in.dtype, np.integer) for counts_in in counts_in_l)
    assert all(data_in.size == counts_in.sum() for data_in, counts_in in zip(buff_in_l, counts_in_l))

    if dist_data is None:
      cnts_dtype = counts_in_l[0].dtype.str if len(self.pn) > 0 else ''
      data_dtype = buff_in_l[0].dtype.str   if len(self.pn) > 0 else ''
      if self.empty_part:
        out_dtype = self.comm.allreduce(cnts_dtype+data_dtype,  MPI.MAX)
        cnts_dtype, data_dtype = out_dtype[:3], out_dtype[3:]
    else:
      counts_out, buff_out = dist_data
      if counts_out.size != self.dn:
        raise ValueError(f"Invalid size of output counts (expected {self.dn}, got {counts_out.size})")
      if buff_out.size - counts_out.sum() != 0:
        raise ValueError(f"Invalid size of output distributed buffer (expected {counts_out.sum()}, got {buff_out.size})")
      cnts_dtype = counts_out.dtype
      data_dtype = buff_out.dtype
      # Retrieve _counts_out from counts_out seems not possible because of data erasion, we will recompute it 


    # Exchange counts_in
    # _ : in all_to_all layout. Do not call Put because we need the intermediate layout
    _counts_in = np.empty(sum([part_write_pos.size for part_write_pos in self.part_write_pos]), dtype=cnts_dtype)
    _counts_out = np.empty(self.dist_counts.sum(),  dtype=_counts_in.dtype)
    for counts_in, part_write_pos in zip(counts_in_l, self.part_write_pos):
      _counts_in[part_write_pos] = counts_in

    self.comm.Alltoallv((_counts_in, self.part_counts), 
                        (_counts_out, self.dist_counts))

    if dist_data is None:
      counts_out  = np.zeros(self.dn, dtype=_counts_out.dtype)
      counts_out[self.dist_select_idx] = _counts_out
      buff_out = np.empty(counts_out.sum(), data_dtype)
    

    # Count the actual number of items to send/recv, using stride array
    # (this is the partial sum of portion of the stride array related to the given rank)
    send_counts = np.empty(self.comm.Get_size(), int)
    recv_counts = np.empty(self.comm.Get_size(), int)
    idx_send = idx_recv = 0
    for i in range(self.comm.Get_size()):
      send_counts[i] = _counts_in [idx_send:idx_send+self.part_counts[i]].sum()
      recv_counts[i] = _counts_out[idx_recv:idx_recv+self.dist_counts[i]].sum()
      idx_send += self.part_counts[i]
      idx_recv += self.dist_counts[i]

    # Prepare send buffer (put data in alltoall layout)
    send_buff = np.empty(send_counts.sum(), data_dtype)
    for i, part_write_pos in enumerate(self.part_write_pos):
      put_strided(send_buff, _counts_in, part_write_pos, counts_in_l[i], buff_in_l[i])

    # Exchange data buffer
    recv_buff = np.empty(recv_counts.sum(), send_buff.dtype)
    self.comm.Alltoallv((send_buff, send_counts), (recv_buff, recv_counts))

    # Post treat recv buffer (data arrive in mpi layout, put it in requested layout)
    put_strided(buff_out, counts_out, self.dist_select_idx, _counts_out, recv_buff)

    return counts_out, buff_out
  
  @property
  def access_counts(self) -> NDArrayInt:
    """ For each global index, total number of apparitions in the ``g_idx`` arrays,
    returned as an integer array of size :math:`dn`."""
    counts = np.zeros(self.dn, int)
    np.add.at(counts, self.dist_select_idx, 1)
    return counts





class GlobalIndexer:

  """
  A protocol object allowing to access distributed data in read or write mode.

  The documentation uses the following notations:

  - :math:`s` : number of processes (equal to ``comm.Get_size()``)
  - :math:`n` : global size of the distributed collection (equal to ``distri[s+1]``)
  - :math:`dn` : for each rank :math:`j`, size of its section of the collection
    (equal to ``distri[j+1]-distri[j]``)
  - :math:`pn` : for each rank :math:`j`, number of accessed indices (equal to ``len(g_idx)``)
  - :math:`c` : for constant buffer access, number of data per item of the collection

  """

  def __init__(self, distri:NDArrayInt, g_idx:NDArrayInt, comm:MPI.Comm):
    """ Create a GlobalIndexer protocol object

    The protocol object is described by two arrays of integer,
    satisfying these rules:
    
    - **distri** (size :math:`s+1`):

      - same value on all processes;
      - ``distri[0] == 0``, ``distri[s] == n``, and ``distri[j] <= distri[j+1]`` for all ``j``.

    - **g_idx** (size :math:`pn`):

      - different size and value on each process;
      - values in range :math:`[0, n-1]`.

    Args:
      distri (integer array of size :math:`s+1`) : distribution of the collection
      g_idx (integer array of size :math:`pn`) : accessed global indices
      comm (MPIComm) : communicator
    """
    self.GIndexer_m = GlobalMultiIndexer(distri, [g_idx], comm)
    self.GIndexer_m._empty_part = False

  def take(self, dist_data:List) -> List:
    """ ``take`` implementation for generic Python objects 
    
    Exchanged data are serialized using ``pickle`` module, which has
    a negative impact on performances; if data is a buffer object, it is
    strongly advised to use :func:`~GlobalIndexer.Take` method instead.

    Args:
      dist_data (list of size :math:`dn`) : section of the distributed data
    Returns:
      list of size :math:`pn`: values extracted at the requested indices
    """
    return self.GIndexer_m.take(dist_data)[0]

  def put(self, local_data: List) -> List:
    """ ``put`` implementation for generic Python objects 
    
    Exchanged data are serialized using ``pickle`` module, which has
    a negative impact on performances; if data is a buffer object, it is
    strongly advised to use :func:`~GlobalIndexer.Put` method instead.

    Note that:

    - if a global index does not appears in any idx list, its associated data in the output
      buffer will be ``None``;
    - if a global index appears more than once in the idx lists, the associated data in the output
      buffer will be the last encoutered (in increasing processes order)

    Args:
      local_data (list of size :math:`pn`) : data to write at each accessed index
    Returns:
      list of size :math:`dn`: output distributed data
    """
    return self.GIndexer_m.put([local_data])

  def Take(self, dist_data:Buffer, local_data:Buffer=None, count=1) -> Buffer:
    """ ``take`` implementation for buffer-like objects 

    Input buffer must be of size :math:`c*dn`, where :math:`c` is a
    positive integer.
    The value of :math:`c` and the datatype <T> of the input buffer must be the
    same across all the processes. 

    The output buffer is either:

    - provided by the caller as a buffer of size :math:`c*pn` and of datatype <T>,
    - or automatically allocated by the method as a new numpy array if ``local_data=None``.
    
    Args:
      dist_data (buffer of size :math:`c*dn`) : section of the distributed data
      local_data (buffer of size :math:`c*pn`, optional) : preallocated buffer
        to store extracted values or None
      count (int) : scalar value of :math:`c`. Defaults to 1.
    Returns:
      buffer of size :math:`c*pn`: values extracted at the requested indices
    """
    local_data_l = [local_data] if local_data is not None else None
    return self.GIndexer_m.Take(dist_data, local_data_l, count)[0]

  def Put(self, local_data: Buffer, dist_data:Buffer=None, count=1) -> Buffer:
    """ ``put`` implementation for buffer-like objects 

    Input buffer must be of size :math:`c*pn`, where :math:`c` is a
    positive integer.
    The value of :math:`c` and the datatype <T> of the input buffer must be the
    same across all the processes. 

    The output buffer is either:

    - provided by the caller as a buffer of size :math:`c*dn` and of datatype <T>,
    - or automatically allocated by the method as a new numpy array if ``dist_data=None``.

    Note that:

    - if a global index does not appears in any idx list, its associated data in the output
      buffer will be uninitialized;
    - if a global index appears more than once in the idx lists, the associated data in the output
      buffer will be the last encoutered (in increasing processes order)

    Args:
      local_data (buffer of size :math:`c*pn`) : data to write at each accessed index
      dist_data (buffer of size :math:`c*dn`, optional) : preallocated buffer
        to store distributed data or None
      count (int) : scalar value of :math:`c`. Defaults to 1.
    Returns:
      buffer of size :math:`c*dn`: output distributed data
    """
    return self.GIndexer_m.Put([local_data], dist_data, count)

  def Take_v(self, dist_data: VBuffer, local_data: VBuffer=None) -> VBuffer:
    """ ``take`` implementation for variable buffer-like objects 

    The input variable buffer is described by a tuple of two objets:

    1. an integer array ``dist_counts`` of size :math:`dn`;
    2. a buffer object of size ``dist_counts.sum()``.
       The datatype <T> of this input buffer must be the same across all the processes.

    Similarly, the output variable buffer is described by the tuple of two objects:

    1. an integer array ``local_counts`` of size :math:`pn`;
    2. a buffer object ``local_buff`` of size ``local_count.sum()`` and datatype <T>.

    This output data is either:

    - provided by the caller, in which case ``local_counts`` must be already filled, *eg.*
      with :obj:`Take(dist_counts, local_counts)`, and ``local_buff``
      must be prellocated at relevant size and datatype;
    - or automatically allocated by the method as a pair of new numpy array if ``local_data=None``.

    
    Args:
      dist_data (variable buffer): section of the distributed data, ie pair of values
        (**dist_counts** (*np array of* :math:`dn` *int*), **dist_buff** (*buffer*))
      local_data (variable buffer, optional): preallocated buffer to store extracted values or None
    Returns:
      variable buffer: data extracted at the requested indices, as a tuple of values \
        (**local_counts** (*np array of* :math:`pn` *int*), **local_buff** (*buffer*))
    """
    local_data_l = [local_data] if local_data is not None else None
    return self.GIndexer_m.Take_v(dist_data, local_data_l)[0]

  def Put_v(self, local_data: VBuffer, dist_data:VBuffer = None) -> VBuffer:
    """ ``put`` implementation for variable buffer-like objects 

    The variable input buffer is described by a tuple of two objets:

    1. an integer array ``local_counts`` of size :math:`pn`;
    2. a buffer object of size ``local_counts.sum()``.
       The datatype <T> of this input buffer must be the same across all the processes.

    Similarly, the output variable buffer is described by the tuple of two objects:

    1. an integer array ``dist_counts`` of size :math:`dn`;
    2. a buffer object ``dist_buff`` of size ``dist_counts.sum()`` and datatype <T>.

    This output data is either:

    - provided by the caller, in which case ``dist_counts`` must be already filled, *eg.*
      with :obj:`Put(local_counts, dist_counts)`, and ``dist_buff``
      must be prellocated at relevant size and datatype;
    - or automatically allocated by the method as a pair of new numpy array if ``dist_data=None``.

    Args:
      local_data (variable buffer): data to write at each accessed index, ie tuple 
        (**local_counts** (*np array of* :math:`pn` *int*), **local_buff** (*buffer*))
      dist_data (variable buffer, optional): preallocated buffer to store distributed data or None
    Returns:
      variable buffer: output distributed data, returned as the tuple of values \
        (**dist_counts** (*np array of* :math:`dn` *int*), **dist_buff** (*buffer*))
    """
    return self.GIndexer_m.Put_v([local_data], dist_data)

  @property
  def empty_dist(self) -> bool:
    return self.GIndexer_m.empty_dist
  @property
  def empty_part(self) -> bool:
    return self.GIndexer_m.empty_part

  @property
  def access_counts(self) -> NDArrayInt:
    """ For each global index, total number of apparitions in the ``g_idx`` arrays,
    returned as an integer array of size :math:`dn`."""
    return self.GIndexer_m.access_counts
