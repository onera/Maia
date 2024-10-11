from mpi4py import MPI
import numpy as np
import pickle


def counting_sort(array, n_bins):
  from cmaia.utils import layouts
  return layouts.counting_sort(array, n_bins)

def counting_sort_mult(arrays, n_bins):
  from cmaia.utils import layouts
  return layouts.counting_sort_mult(arrays, n_bins)
  counts = np.zeros(n_bins, int)
  for array in arrays:
    counts += np.bincount(array, minlength=n_bins)

  displ = np.empty(counts.size+1, int)
  displ[0] = 0
  np.cumsum(counts, out=displ[1:])

  counts *= 0
  out_list = list()
  for array in arrays:
    out = np.empty_like(array)
    for i in range(array.size):
      out[i] = displ[array[i]] + counts[array[i]]
      counts[array[i]] += 1
    out_list.append(out)

  return out_list, counts

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
  from cmaia.utils import layouts
  layouts.take_stridedDI(a_counts, a_val, indices, out)

def put_strided(a, indices, indices_count, read_counts, read):
  """
  Write in an array strided array (a) at provided indices (indices, indices_count)
  from an input data (read, read_count).
  If an index occurs multiple times in indices array, it erase the previously written
  value. A check is performed on counts to write only compatible data
  """
  from cmaia.utils import layouts
  layouts.put_strided(a, indices, indices_count, read_counts, read)

class GIndexer_m:

  def __init__(self, distri, g_idx_l, comm):
    assert distri.size == comm.Get_size() + 1

    # Binary search : for each gnum, find the corresponding rank in distribution
    owning_rank_l = [np.searchsorted(distri, g_idx-1, side='right') - 1 for g_idx in g_idx_l]

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
    recv_data -= (distri[comm.Get_rank()] + 1)

    self.comm = comm
    self.dn = distri[comm.Get_rank()+1] - distri[comm.Get_rank()]   # Number of managed indices
    self.pn = [g_idx.size for g_idx in g_idx_l]                     # Number of accessed indices
    self.dist_counts = recv_counts  # Number of managed indices accessed by each rank (with multiplicity)
    self.part_counts = send_counts  # Number of accessed indices managed by each rank (with multiplicity)
    self.dist_select_idx = recv_data   # Selection order of managed data to put it in MPI order
    self.part_write_pos  = sorting_idx_l # Position where accessed data should be put to have it in MPI order

    self._empty_dist = (np.diff(distri)==0).any() # True if at least one rank has dn == 0
    self._empty_part = None                       # True if at least one rank has pn == 0

  @property
  def empty_dist(self):
    return self._empty_dist
  @property
  def empty_part(self):
    if self._empty_part is None:
      is_empty = not any(pn > 0 for pn in self.pn)
      self._empty_part = self.comm.allreduce(is_empty, MPI.LOR)
    return self._empty_part


  def Take_into(self, data_in, data_out_l):
    """
    AlltoAll like with constant (guessed) counts
    """
    assert len(data_out_l) == len(self.pn)
    counts_in  = data_in.size  // self.dn if self.dn != 0 else 0
    counts_out = 0 # Init for procs having n_part == 0

    if data_in.size - counts_in*self.dn != 0:
      raise ValueError("Input data size is not a multiple of managed idx")
    for data_out, pn in zip(data_out_l, self.pn):
      counts_out = data_out.size // pn if pn != 0 else 0
      if data_out.size - counts_out*pn != 0:
        raise ValueError("Output data size is not a multiple of requested idx")
      if counts_in != counts_out and counts_in*counts_out != 0:
        raise ValueError("Input and output counts does not match")


    send_buff = np.empty(counts_in*self.dist_select_idx.size, data_in.dtype)
    recv_buff = np.empty(counts_out*sum(self.pn),             data_in.dtype)

    if counts_in == 1:
      send_buff[:] = data_in[self.dist_select_idx]
    else:
      pull_idx = counts_in*self.dist_select_idx
      for j in range(counts_in):
        send_buff[j::counts_in] = data_in[pull_idx+j]

    self.comm.Alltoallv((send_buff, counts_in*self.dist_counts),
                        (recv_buff, counts_out*self.part_counts))

    # Data has been received in owning proc order : 'unsort' it to recover lngn ordering
    for data_out, part_write_pos in zip(data_out_l, self.part_write_pos):
      if counts_out == 1:
        data_out[:] = recv_buff[part_write_pos]
      else:
        put_idx = counts_out*part_write_pos
        for j in range(counts_out):
          data_out[j::counts_out] = recv_buff[put_idx+j]

  def Take(self, data_in):
    counts = data_in.size // self.dn if self.dn != 0 else 0
    if self.empty_dist:
      counts = self.comm.allreduce(counts, MPI.MAX)

    data_out_l = [np.empty(counts*pn, data_in.dtype) for pn in self.pn]
    self.Take_into(data_in, data_out_l)
    return data_out_l


  def Take_v_into(self, data_in, counts_in, data_out_l, counts_out_l):
    # Retrive MPI order for counts
    _counts_in  = counts_in[self.dist_select_idx]
    _counts_out = np.empty(sum([c.size for c in counts_out_l]), counts_in.dtype)
    for counts_out, part_write_pos in zip(counts_out_l, self.part_write_pos):
      _counts_out[part_write_pos] = counts_out

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
    
    send_buff = np.empty(send_counts.sum(), data_in.dtype)
    take_strided(counts_in, data_in, self.dist_select_idx, send_buff)

    # Exchange data buffer
    recv_buff = np.empty(recv_counts.sum(), data_in.dtype)
    self.comm.Alltoallv((send_buff, send_counts), (recv_buff, recv_counts))

    # Post treat recv buffer (data arrive in mpi layout, put it in requested layout)
    for data_out, part_write_pos in zip(data_out_l, self.part_write_pos):
      take_strided(_counts_out, recv_buff, part_write_pos, data_out)


  def Take_v(self, data_in, counts_in):
    # Variable stride

    assert isinstance(counts_in, np.ndarray)
    assert counts_in.size == self.dn
    assert counts_in.dtype == int
    assert data_in.size == counts_in.sum()
    # Exchange counts_in
    # _ : in all_to_all layout. Do not call Take because we need the intermediate layout
    _counts_in  = counts_in[self.dist_select_idx]
    _counts_out = np.empty(sum(self.pn), counts_in.dtype)
    self.comm.Alltoallv((_counts_in, self.dist_counts), (_counts_out, self.part_counts))
    counts_out_l = [_counts_out[part_write_pos] for part_write_pos in self.part_write_pos]

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

    
    send_buff = np.empty(send_counts.sum(), data_in.dtype)
    take_strided(counts_in, data_in, self.dist_select_idx, send_buff)

    # Exchange data buffer
    recv_buff = np.empty(recv_counts.sum(), data_in.dtype)
    self.comm.Alltoallv((send_buff, send_counts), (recv_buff, recv_counts))

    # Post treat recv buffer (data arrive in mpi layout, put it in requested layout)
    data_out_l = list()
    for counts_out, part_write_pos in zip(counts_out_l, self.part_write_pos):
      data_out = np.empty(counts_out.sum(), recv_buff.dtype)
      take_strided(_counts_out, recv_buff, part_write_pos, data_out)
      data_out_l.append(data_out)

    return data_out_l, counts_out_l


  def take(self, data_in):
    """
    Take implementation for any python object.
    data_in must be a list of size dn on each rank.
    """
    pickelized = [pickle.dumps(data) for data in data_in]
    counts_in  = np.array([len(p) for p in pickelized], dtype=int)

    joined = np.frombuffer(b''.join(pickelized), dtype=np.int8)
    
    data_out_l, counts_out_l = self.Take_v(joined, counts_in)
    
    res = list()
    for data_out, counts_out in zip(data_out_l, counts_out_l):
      out = []
      r_start = 0
      for size in counts_out:
        out.append(pickle.loads(data_out[r_start:r_start+size].tobytes()))
        r_start += size
      res.append(out)
    return res


  def Put_into(self, data_in_l, data_out):
    assert len(data_in_l) == len(self.pn)
    counts_in  = 0
    counts_out = data_out.size // self.dn if self.dn != 0 else 0

    if data_out.size - counts_out*self.dn != 0:
      raise ValueError("Output data size is not a multiple of requested idx")
    for data_in, pn in zip(data_in_l, self.pn):
      counts_in  = data_in.size  // pn if pn != 0 else 0
      if data_in.size - counts_in*pn != 0:
        raise ValueError("Input data size is not a multiple of managed idx")
      if counts_in != counts_out and counts_in*counts_out != 0:
        raise ValueError("Input and output counts does not match")


    send_buff = np.empty(counts_in*sum([write_pos.size for write_pos in self.part_write_pos]), dtype=data_out.dtype)
    recv_buff = np.empty(counts_out*self.dist_counts.sum(),  dtype=data_out.dtype)

    for data_in, part_write_pos in zip(data_in_l, self.part_write_pos):
      if counts_in == 1:
        send_buff[part_write_pos] = data_in
      else:
        pull_idx = counts_in*part_write_pos
        for j in range(counts_in):
          send_buff[pull_idx+j] = data_in[j::counts_in]

    self.comm.Alltoallv((send_buff, counts_in*self.part_counts), 
                        (recv_buff, counts_out*self.dist_counts))

    if counts_out == 1:
      data_out[self.dist_select_idx] = recv_buff
    else:
      put_idx = counts_out*self.dist_select_idx
      for j in range(counts_out):
        data_out[put_idx+j] = recv_buff[j::counts_out]

  def Put(self, data_in_l):
    assert len(data_in_l) == len(self.pn)
    counts_l = [data_in.size // pn for data_in, pn in zip(data_in_l, self.pn) if pn != 0]
    assert all(count == counts_l[0] for count in counts_l)
    counts = counts_l[0]            if len(counts_l)  > 0 else 0
    dtype  = data_in_l[0].dtype.str if len(data_in_l) > 0 else ''
    if self.empty_part:
      counts = self.comm.allreduce(counts, MPI.MAX)
      dtype  = self.comm.allreduce(dtype,  MPI.MAX)

    data_out = np.empty(counts*self.dn, dtype)
    self.Put_into(data_in_l, data_out)
    return data_out

  def Put_v(self, data_in_l, counts_in_l):
    # Variable stride

    assert len(counts_in_l) == len(data_in_l) == len(self.pn)
    assert all(counts_in.size == pn for counts_in,pn in zip(counts_in_l, self.pn))
    assert all(counts_in.dtype == int for counts_in in counts_in_l)
    assert all(data_in.size == counts_in.sum() for data_in, counts_in in zip(data_in_l, counts_in_l))

    dtype  = data_in_l[0].dtype.str if len(data_in_l) > 0 else ''
    if self.empty_part:
      dtype = self.comm.allreduce(dtype,  MPI.MAX)


    # Exchange counts_in
    # _ : in all_to_all layout. Do not call Put because we need the intermediate layout
    _counts_in = np.empty(sum([part_write_pos.size for part_write_pos in self.part_write_pos]), dtype=int)
    _counts_out = np.empty(self.dist_counts.sum(),  dtype=int)
    for counts_in, part_write_pos in zip(counts_in_l, self.part_write_pos):
      _counts_in[part_write_pos] = counts_in

    self.comm.Alltoallv((_counts_in, self.part_counts), 
                        (_counts_out, self.dist_counts))
    counts_out  = np.zeros(self.dn, dtype=int)
    counts_out[self.dist_select_idx] = _counts_out

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
    send_buff = np.empty(send_counts.sum(), dtype)
    for i, part_write_pos in enumerate(self.part_write_pos):
      put_strided(send_buff, part_write_pos, _counts_in, counts_in_l[i], data_in_l[i])

    # Exchange data buffer
    recv_buff = np.empty(recv_counts.sum(), send_buff.dtype)
    self.comm.Alltoallv((send_buff, send_counts), (recv_buff, recv_counts))

    # Post treat recv buffer (data arrive in mpi layout, put it in requested layout)
    data_out = np.empty(counts_out.sum(), recv_buff.dtype)
    put_strided(data_out, self.dist_select_idx, counts_out, _counts_out, recv_buff)

    return data_out, counts_out


  def put(self, data_in_l):
    """
    Put implementation for any python object.
    data_in must be a list of size dn on each rank.
    """
    counts_in_l = list()
    joined_l    = list()
    for data_in in data_in_l:
      pickelized = [pickle.dumps(data) for data in data_in]
      counts_in_l.append(np.array([len(p) for p in pickelized], dtype=int))
      joined_l.append(np.frombuffer(b''.join(pickelized), dtype=np.int8))
    
    data_out, counts_out = self.Put_v(joined_l, counts_in_l)
    
    out = []
    r_start = 0
    for size in counts_out:
      if size != 0:
        out.append(pickle.loads(data_out[r_start:r_start+size].tobytes()))
      else:
        out.append(None)
      r_start += size
    return out


class GIndexer:

  """
  A proxy object allowing to access distributed data in read or write mode.

  Notations:

  - :math:`s` : number of processes (equal to ``comm.Get_size()``)
  - :math:`n` : global size of the distributed collection (equal to ``distri[s+1]``)
  - :math:`dn` : for each rank :math:`j`, size of its section of the collection
    (equal to ``distri[j+1]-distri[j]``)
  - :math:`pn` : for each rank :math:`j`, number of accessed indices (equal to ``len(g_idx)``)
  - :math:`c` : for constant buffer access, number of data per item of the collection

  """

  def __init__(self, distri, g_idx, comm):
    """ Create a GIndexer proxy object

    The proxy object is described by two arrays of integer,
    satisfying these rules:
    
    - **distri** (size :math:`s+1`):

      - same value on all processes;
      - ``distri[0] == 0``, ``distri[s] == n``, and ``distri[j] <= distri[j+1]`` for all ``j``.

    - **g_idx** (size :math:`pn`):

      - different size and value on each process;
      - values in range :math:`[1, n]`.

    Args:
      distri (integer array of size :math:`s+1`) : distribution of the collection
      g_idx (integer array of size :math:`pn`) : accessed global indices
      comm (MPIComm) : communicator
    """
    self.indexer = GIndexer_m(distri, [g_idx], comm)

  def take(self, data_in:list) -> list:
    """ ``take`` implementation for generic Python objects 
    
    Exchanged data are serialized using ``pickle`` module. 

    Args:
      data_in (list of size :math:`dn`) : section of the distributed data
    Returns:
      list of size :math:`pn`: values extracted at the requested indices
    """
    return self.indexer.take(data_in)[0]

  def put(self, data_in:list) -> list:
    """ ``put`` implementation for generic Python objects 
    
    Exchanged data are serialized using ``pickle`` module.

    Note that:

    - if a global index does not appears in any idx list, its associated data in the output
      buffer will be ``None``;
    - if a global index appears more than once in the idx lists, the associated data in the output
      buffer will be the last encoutered (in increasing processes order)

    Args:
      data_in (list of size :math:`pn`) : data to write at each accessed index
    Returns:
      list of size :math:`dn`: output distributed data
    """
    return self.indexer.put([data_in])

  def Take_into(self, data_in, data_out):
    """ Inplace ``take`` implementation for buffer-like objects 

    Input and output buffer must respectively be of size :math:`c*dn` and
    :math:`c*pn`, where :math:`c` is a positive integer. The datatype
    of the input and the ouput buffer must match.

    The value of :math:`c` and the datatype must be the
    same across all the processes. 
    
    Args:
      data_in  (buffer) : section of the distributed data
      data_out (buffer) : preallocated buffer to store extracted values
    """
    self.indexer.Take_into(data_in, [data_out])

  def Put_into(self, data_in, data_out):
    """ Inplace ``put`` implementation for buffer-like objects 

    Input and output buffer must respectively be of size :math:`c*pn` and
    :math:`c*dn`, where :math:`c` is a positive integer. The datatype
    of the input and the ouput buffer must match.

    The value of :math:`c` and the datatype must be the
    same across all the processes. 

    Note that:

    - if a global index does not appears in any idx list, its associated data in the output
      buffer will be unchanged;
    - if a global index appears more than once in the idx lists, the associated data in the output
      buffer will be the last encoutered (in increasing processes order)
    
    Args:
      data_in  (buffer) : data to write at each accessed index
      data_out (buffer) : preallocated buffer to store distributed data
    """
    self.indexer.Put_into([data_in], data_out)
    
  def Take(self, data_in) -> np.ndarray:
    """ ``take`` implementation for buffer-like objects 

    Input buffer must be of size :math:`c*dn`, where :math:`c` is a
    positive integer.
    The value of :math:`c` and the datatype of the input buffer must be the
    same across all the processes. 

    The output buffer is allocated as a numpy array of size :math:`c*pn`
    and of datatype equal to the one of the input data.
    
    Args:
      data_in (buffer of size :math:`c*dn`) : section of the distributed data
    Returns:
      buffer of size :math:`c*pn`: values extracted at the requested indices
    """
    return self.indexer.Take(data_in)[0]

  def Put(self, data_in) -> np.ndarray:
    """ ``put`` implementation for buffer-like objects 

    Input buffer must be of size :math:`c*pn`, where :math:`c` is a
    positive integer.
    The value of :math:`c` and the datatype of the input buffer must be the
    same across all the processes. 

    The output buffer is allocated as a numpy array of size :math:`c*dn`
    and of datatype equal to the one of the input data.

    Note that:

    - if a global index does not appears in any idx list, its associated data in the output
      buffer will be uninitialized;
    - if a global index appears more than once in the idx lists, the associated data in the output
      buffer will be the last encoutered (in increasing processes order)

    Args:
      data_in (buffer of size :math:`c*pn`) : data to write at each accessed index
    Returns:
      buffer of size :math:`c*pn`: output distributed data
    """

    return self.indexer.Put([data_in])

  def Take_v_into(self, data_in, counts_in, data_out, counts_out):
    self.indexer.Take_v_into(data_in, counts_in, [data_out], [counts_out])

  def Take_v(self, buff_in, counts_in):
    """ ``take`` implementation for variable buffer-like objects 

    The variable input buffer is described by two objets:

    - an integer array ``counts_in`` of size :math:`dn`;
    - a buffer object of size ``counts_in.sum()``.
      The datatype of this input buffer must be the same across all the processes.

    Similarly, the output data is returned as a pair of two newly allocated numpy arrays:

    - an integer array ``counts_out`` of size :math:`pn`;
    - a buffer object of size ``counts_out.sum()``.
      The datatype of this output buffer is set to be the same than the input buffer.

    Be aware that following mpi4py convention, the order of objects is ``(buff, counts)``.
    
    Args:
      buff_in (buffer) : values for the distributed data
      counts_in (np array of :math:`dn` int) : counts for the distributed data
    Returns:
      tuple: pair of values extracted at the requested indices, ie:

      - **buff_out** (*buffer*): extracted values
      - **counts_out** (np array of :math:`pn` int): extracted counts
    """
    #TODO : to follow mpi4py and other funcs signature, we should use tuple in input
    #def Take_v(self, data_in): with data_in == (buff_in, counts_in)
    buff_out_l, counts_out_l = self.indexer.Take_v(buff_in, counts_in)
    return buff_out_l[0], counts_out_l[0]
   
  def Put_v(self, buff_in, counts_in):
    """ ``put`` implementation for variable buffer-like objects 

    The variable input buffer is described by two objets:

    - an integer array ``counts_in`` of size :math:`pn`;
    - a buffer object of size ``counts_in.sum()``.
      The datatype of this input buffer must be the same across all the processes.

    Similarly, the output data is returned as a pair of two newly allocated numpy arrays:

    - an integer array ``counts_out`` of size :math:`dn`;
    - a buffer object of size ``counts_out.sum()``.
      The datatype of this output buffer is set to be the same than the input buffer.

    Be aware that following mpi4py convention, the order of objects is ``(buff, counts)``.

    Args:
      buff_in (buffer) : values to write
      counts_in (np array of :math:`pn` int) : number of values to write at each accessed index
    Returns:
      tuple: pair of values at each global index, ie:

      - **buff_out** (*buffer*): values for the output distributed data
      - **counts_out** (np array of :math:`dn` int): counts for output distributed data
    """
    return self.indexer.Put_v([buff_in], [counts_in])
