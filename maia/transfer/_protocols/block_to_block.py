import numpy as np

def _overlap_size(start1, end1, start2, end2):
  return max(min(end1, end2) - max(start1, start2), 0)

class BlockToBlock:

    def __init__(self, distri_in, distri_out, comm):
      assert distri_in.size == distri_out.size == comm.Get_size() + 1
      self.comm = comm

      rank = comm.Get_rank()
      self.dn_in  = distri_in[rank+1] - distri_in[rank]
      self.dn_out = distri_out[rank+1] - distri_out[rank]

      self.send_counts = np.empty(comm.Get_size(), int)
      self.recv_counts = np.empty(comm.Get_size(), int)

      for i in range(comm.Get_size()):
        self.send_counts[i] = _overlap_size(distri_in[rank], distri_in[rank+1], distri_out[i], distri_out[i+1])
        self.recv_counts[i] = _overlap_size(distri_out[rank],distri_out[rank+1], distri_in[i], distri_in[i+1])

      assert self.send_counts.sum() == self.dn_in  
      assert self.recv_counts.sum() == self.dn_out

      self.exchange_field = self.exchange # Compatibility


    def exchange_inplace(self, data_in, data_out, stride_in=1):
      """
      TODO : factorize exchange_inplace / exchange => to do after GlobalIndexer MR
      """

      # Constant stride
      if isinstance(stride_in, int):
        dtype = data_in.dtype.char
        assert data_out.size == stride_in*self.recv_counts.sum() and data_out.dtype == data_in.dtype
        #Really strange, without explicit datatype mpi4py can fail if dtype is float64
        #File "mpi4py/MPI/msgbuffer.pxi", line 145, in mpi4py.MPI.message_basic
        #KeyError: '<d'
        self.comm.Alltoallv((data_in, stride_in * self.send_counts, dtype), (data_out, stride_in * self.recv_counts, dtype))
      else:
        raise NotImplementedError('Inplace variable buffer not yet implemented')

    def exchange(self, data_in, stride_in=1):

      # Constant stride
      if isinstance(stride_in, int):
        data_out = np.empty(stride_in * self.dn_out, data_in.dtype)
        dtype = data_in.dtype.char
        #Really strange, without explicit datatype mpi4py can fail if dtype is float64
        #File "mpi4py/MPI/msgbuffer.pxi", line 145, in mpi4py.MPI.message_basic
        #KeyError: '<d'
        self.comm.Alltoallv((data_in, stride_in * self.send_counts, dtype), (data_out, stride_in * self.recv_counts, dtype))
        return data_out

      # Variable stride
      elif isinstance(stride_in, np.ndarray):
        assert stride_in.size == self.dn_in
        stride_out = np.empty(self.dn_out, stride_in.dtype)
        self.comm.Alltoallv((stride_in, self.send_counts), (stride_out, self.recv_counts))

        send_counts = np.empty(self.comm.Get_size(), int)
        recv_counts = np.empty(self.comm.Get_size(), int)
        idx_send = 0
        idx_recv = 0
        # Count the actual number of items to send/recv, using stride array
        # (this is the partial sum of portion of the stride array related to the given rank)
        for i in range(self.comm.Get_size()):
          send_counts[i] = stride_in [idx_send:idx_send+self.send_counts[i]].sum()
          recv_counts[i] = stride_out[idx_recv:idx_recv+self.recv_counts[i]].sum()
          idx_send += self.send_counts[i]
          idx_recv += self.recv_counts[i]

        data_out = np.empty(recv_counts.sum(), data_in.dtype)
        dtype = data_in.dtype.char #Really strange, without that mpi4py can fail if dtype is float64
        self.comm.Alltoallv((data_in, send_counts, dtype), (data_out, recv_counts, dtype))

        return stride_out, data_out

class SerialBlockToBlock():
  """
  A serial implementation for the specific case distri_in == distri_out
  Data is just copied to an output buffer
  """
  def __init__(self, distri_in, distri_out):
    assert np.array_equal(distri_in, distri_out)
    self.exchange_field = self.exchange # Compatibility

  def exchange(self, data_in, stride_in=1):
    if isinstance(stride_in, int):
      return np.copy(data_in)
    elif isinstance(stride_in, np.ndarray):
      return np.copy(stride_in), np.copy(data_in)

  def exchange_inplace(self, data_in, data_out, stride_in=1):
    if isinstance(stride_in, int):
      assert data_out.size == data_in.size and data_out.dtype == data_in.dtype
      data_out[:] = data_in[:]
    else:
      raise NotImplementedError('Inplace variable buffer not yet implemented')

class MultiBlockToBlock():
  """
  The multiblock to block pattern allows to remap N distributed arrays
  to a single distributed array, without interlacing the inputs
  (which is what we would have if we just concatenate arrays
  locally on input ranks)
  For exemple if we have 2 ranks and 3 distributed arrays
  Array0 [A B C | D E F G]
  Array1 [M N L | Q]
  Array2 [P S | K X W]    ('|' marks separation for P0/P1)

  Output will be [A B C D E F G M | N L Q P S K X W]
  (distribution of output array can be choosed)
  """

  def __init__(self, distri_in_l, distri_out, comm):
    assert sum(distri[-1] for distri in distri_in_l) == distri_out[-1]
    self.btb_l = list()
    start = 0
    for distri_in in distri_in_l:
      end = start + distri_in[-1]
      distri_out_loc = np.maximum(np.minimum(distri_out, end), start) - start
      self.btb_l.append(BlockToBlock(distri_in, distri_out_loc, comm))
      start = end

  def exchange(self, data_in_l, stride_in=1):
    
    # Constant stride
    if isinstance(stride_in, int):
      data_out = np.empty(stride_in * sum(btb.dn_out for btb in self.btb_l), data_in_l[0].dtype)

      loc_start = 0
      for data_in, btb in zip(data_in_l, self.btb_l):
        loc_end = loc_start + stride_in * btb.dn_out
        btb.exchange_inplace(data_in, data_out[loc_start:loc_end], stride_in=stride_in)
        loc_start = loc_end

      return data_out

    # Variable stride
    elif isinstance(stride_in, list):
      stride_out = self.exchange(stride_in) # Exchange with stride = 1 to get output vstride
      data_out = np.empty(stride_out.sum(), dtype=data_in_l[0].dtype)

      loc_start = 0
      for _stride_in, data_in, btb in zip(stride_in, data_in_l, self.btb_l):
        stride_out_loc, data_out_loc = btb.exchange(data_in, _stride_in)
        loc_end = loc_start + stride_out_loc.sum()
        data_out[loc_start:loc_end] = data_out_loc
        loc_start = loc_end

      return stride_out, data_out
