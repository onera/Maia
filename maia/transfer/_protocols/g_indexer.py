from mpi4py import MPI
import numpy as np
import pickle

from icecream import ic
import time


    
def counting_sortPY(array, n_bins):
  counts = np.bincount(array, minlength=n_bins)

  displ = np.empty(counts.size+1, int)
  displ[0] = 0
  np.cumsum(counts, out=displ[1:])

  counts *= 0
  out = np.empty_like(array)
  for i in range(array.size):
    out[i] = displ[array[i]] + counts[array[i]]
    counts[array[i]] += 1

  return out, counts

def counting_sortN(array, n_bins):
  sort_idx = np.argsort(array)
  out = np.empty_like(array)
  out[sort_idx] = np.arange(array.size)
  counts = np.bincount(array, minlength=n_bins)
  return out, counts

def counting_sort(array, n_bins):
  from cmaia.utils import layouts
  return layouts.counting_sort(array, n_bins)



def take_stridedPY(a_counts, a_val, indices, out):
  displ = np.empty(a_counts.size+1, dtype=int)
  displ[0] = 0
  np.cumsum(a_counts, out=displ[1:])
  w_start = w_end = 0
  for idx in indices:
    w_end = w_start + a_counts[idx]
    out[w_start:w_end] = a_val[displ[idx]:displ[idx+1]]
    w_start = w_end

def take_stridedN(a_counts, a_val, indices, out):
  from maia.utils import np_utils
  displ = np.empty(a_counts.size+1, dtype=int)
  displ[0] = 0
  np.cumsum(a_counts, out=displ[1:])
  out[:] = a_val[np_utils.multi_arange(displ[indices], displ[indices+1])]

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

class DIndexer:

  @classmethod
  def from_size(cls, size, lngn_out, comm):
    distri = np.empty(comm.Get_size()+1, int)
    distri[0] = 0
    comm.Allgather(np.array(size, int), distri[1:])
    np.cumsum(distri, out=distri)
    return cls(distri, lngn_out, comm)

  def __init__(self, distri, g_idx, comm):
    import time
    if comm.rank == 0:
      print('\n')
    assert distri.size == comm.Get_size() + 1

    st = time.time()
    # Binary search : for each gnum, find the corresponding rank in distribution
    owning_rank = np.searchsorted(distri, g_idx-1, side='right') - 1
    ed = time.time()
    if comm.rank == 0:
      print(f'  Binary search : {ed-st:.3f}')

    st = time.time()
    ok = (0 <= owning_rank).all() and (owning_rank < comm.Get_size()).all()
    if not comm.allreduce(ok, op=MPI.LAND):
      raise IndexError("Some idx not in distri")
    ed = time.time()
    if comm.rank == 0:
      print(f'  Distri check : {ed-st:.3f}')

    # Compute the sorting order to put lngn in increasing rank order
    st = time.time()
    sorting_idx, send_counts = counting_sort(owning_rank, comm.Get_size())
    ed = time.time()
    if comm.rank == 0:
      print(f'  Counting_sort : {ed-st:.3f}')



    st = time.time()
    # Now send gnum to their managing rank, and move it to local indices
    recv_counts = np.empty(comm.Get_size(), int)
    comm.Alltoall(send_counts, recv_counts)
    ed = time.time()
    if comm.rank == 0:
      print(f'  All to all : {ed-st:.3f}')

    send_data = np.empty(send_counts.sum(), int)
    recv_data = np.empty(recv_counts.sum(), int)

    st = time.time()
    send_data[sorting_idx] = g_idx
    ed = time.time()
    if comm.rank == 0:
      print(f'  Prepare send: {ed-st:.3f}')
    st = time.time()
    comm.Alltoallv((send_data, send_counts), (recv_data, recv_counts))
    ed = time.time()
    if comm.rank == 0:
      print(f'  All to all v: {ed-st:.3f}')
    st = time.time()
    recv_data -= (distri[comm.Get_rank()] + 1)
    ed = time.time()
    if comm.rank == 0:
      print(f'  Post recv : {ed-st:.3f}')

    st = time.time()
    self.comm = comm
    self.dn = distri[comm.Get_rank()+1] - distri[comm.Get_rank()]   # Number of managed indices
    self.pn = g_idx.size                                            # Number of accessed indices
    self.dist_counts = recv_counts  # Number of managed indices accessed by each rank (with multiplicity)
    self.part_counts = send_counts  # Number of accessed indices managed by each rank (with multiplicity)
    self.dist_select_idx = recv_data   # Selection order of managed data to put it in MPI order
    self.part_write_pos  = sorting_idx # Position where accessed data should be put to have it in MPI order

    self._empty_dist = (np.diff(distri)==0).any() # True if at least one rank has dn == 0
    self._empty_part = None                       # True if at least one rank has pn == 0
    ed = time.time()
    if comm.rank == 0:
      print(f'  Attrs fill : {ed-st:.3f}')

  @property
  def empty_dist(self):
    return self._empty_dist
  @property
  def empty_part(self):
    if self._empty_part is None:
      self._empty_part = self.comm.allreduce(self.pn == 0, MPI.LOR)
    return self._empty_part


  def Take_into_b_test(self, data_in, data_out):
    """
    AlltoAll like with constant (guessed) counts
    """
    view_in =  np.frombuffer(data_in, memoryview(data_in).format)
    view_out = np.frombuffer(data_out, memoryview(data_out).format)
    counts_in  = view_in.size  // self.dn if self.dn != 0 else 0
    counts_out = view_out.size // self.pn if self.pn != 0 else 0

    if len(view_in) - counts_in*self.dn != 0:
      raise ValueError("Input data size is not a multiple of managed idx")
    if len(view_out) - counts_out*self.pn != 0:
      raise ValueError("Output data size is not a multiple of requested idx")
    if counts_in != counts_out and counts_in*counts_out != 0:
      raise ValueError("Input and output counts does not match")


    send_buff = np.empty(counts_in*self.dist_select_idx.size, view_in.dtype)
    recv_buff = np.empty(counts_out*self.pn,                  view_out.dtype)

    if counts_in == 1:
      send_buff[:] = view_in[self.dist_select_idx]
    else:
      pull_idx = counts_in*self.dist_select_idx
      for j in range(counts_in):
        send_buff[j::counts_in] = view_in[pull_idx+j]

    self.comm.Alltoallv((send_buff, counts_in*self.dist_counts),
                        (recv_buff, counts_out*self.part_counts))

    # Data has been received in owning proc order : 'unsort' it to recover lngn ordering
    if counts_out == 1:
      view_out[:] = recv_buff[self.part_write_pos]
    else:
      put_idx = counts_out*self.part_write_pos
      for j in range(counts_out):
        view_out[j::counts_out] = recv_buff[put_idx+j]

  def Take_into(self, data_in, data_out):
    """
    AlltoAll like with constant (guessed) counts
    """
    counts_in  = data_in.size  // self.dn if self.dn != 0 else 0
    counts_out = data_out.size // self.pn if self.pn != 0 else 0

    if data_in.size - counts_in*self.dn != 0:
      raise ValueError("Input data size is not a multiple of managed idx")
    if data_out.size - counts_out*self.pn != 0:
      raise ValueError("Output data size is not a multiple of requested idx")
    if counts_in != counts_out and counts_in*counts_out != 0:
      raise ValueError("Input and output counts does not match")


    send_buff = np.empty(counts_in*self.dist_select_idx.size, data_in.dtype)
    recv_buff = np.empty(counts_out*self.pn,                  data_out.dtype)

    if counts_in == 1:
      send_buff[:] = data_in[self.dist_select_idx]
    else:
      pull_idx = counts_in*self.dist_select_idx
      for j in range(counts_in):
        send_buff[j::counts_in] = data_in[pull_idx+j]

    self.comm.Alltoallv((send_buff, counts_in*self.dist_counts),
                        (recv_buff, counts_out*self.part_counts))

    # Data has been received in owning proc order : 'unsort' it to recover lngn ordering
    if counts_out == 1:
      data_out[:] = recv_buff[self.part_write_pos]
    else:
      put_idx = counts_out*self.part_write_pos
      for j in range(counts_out):
        data_out[j::counts_out] = recv_buff[put_idx+j]

  def Take(self, data_in):
    counts = data_in.size // self.dn if self.dn != 0 else 0
    if self.empty_dist:
      counts = self.comm.allreduce(counts, MPI.MAX)

    data_out = np.empty(counts*self.pn, data_in.dtype)
    self.Take_into(data_in, data_out)
    return data_out


  def Take_v_into(self, data_in, counts_in, data_out, counts_out):
    # Retrive MPI order for counts
    _counts_in  = counts_in[self.dist_select_idx]
    _counts_out = np.empty(counts_out.size, counts_out.dtype)
    _counts_out[self.part_write_pos] = counts_out

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
    take_strided(_counts_out, recv_buff, self.part_write_pos, data_out)

  def Take_v_from_take_into_test(self, data_in, counts_in):
    counts_out = self.Take(counts_in)
    data_out = np.empty(counts_out.sum(), data_in.dtype)
    self.Take_v_into(data_in, counts_in, data_out, counts_out)
    return data_out, counts_out



  def Take_v(self, data_in, counts_in):
    # Variable stride

    st = time.time()
    assert isinstance(counts_in, np.ndarray)
    assert counts_in.size == self.dn
    assert counts_in.dtype == int
    assert data_in.size == counts_in.sum()
    # Exchange counts_in
    # _ : in all_to_all layout. Do not call Take because we need the intermediate layout
    _counts_in  = counts_in[self.dist_select_idx]
    _counts_out = np.empty(self.pn, counts_in.dtype)
    self.comm.Alltoallv((_counts_in, self.dist_counts), (_counts_out, self.part_counts))
    counts_out = _counts_out[self.part_write_pos]
    ed = time.time()
    if self.comm.rank == 0:  print("  exch counts in", ed-st)

    # Count the actual number of items to send/recv, using stride array
    # (this is the partial sum of portion of the stride array related to the given rank)
    st = time.time()
    send_counts = np.empty(self.comm.Get_size(), int)
    recv_counts = np.empty(self.comm.Get_size(), int)
    idx_send = idx_recv = 0

    for i in range(self.comm.Get_size()):
      send_counts[i] = _counts_in [idx_send:idx_send+self.dist_counts[i]].sum()
      recv_counts[i] = _counts_out[idx_recv:idx_recv+self.part_counts[i]].sum()
      idx_send += self.dist_counts[i]
      idx_recv += self.part_counts[i]
    ed = time.time()
    if self.comm.rank == 0:  print("  count send/recv", ed-st)

    
    st = time.time()
    send_buff = np.empty(send_counts.sum(), data_in.dtype)
    take_strided(counts_in, data_in, self.dist_select_idx, send_buff)

    ed = time.time()
    if self.comm.rank == 0:  print( "  prepare send buff", ed-st)


    # Exchange data buffer
    st = time.time()
    recv_buff = np.empty(recv_counts.sum(), data_in.dtype)
    self.comm.Alltoallv((send_buff, send_counts), (recv_buff, recv_counts))
    ed = time.time()
    if self.comm.rank == 0:  print("  exch data buff", ed-st)

    # Post treat recv buffer (data arrive in mpi layout, put it in requested layout)
    st = time.time()
    data_out = np.empty_like(recv_buff)
    take_strided(_counts_out, recv_buff, self.part_write_pos, data_out)
    ed = time.time()
    if self.comm.rank == 0:  print("  write recv buff", ed-st)


    return data_out, counts_out

  def take(self, data_in):
    """
    Take implementation for any python object.
    data_in must be a list of size dn on each rank.
    """
    pickelized = [pickle.dumps(data) for data in data_in]
    counts_in  = np.array([len(p) for p in pickelized], dtype=int)

    joined = np.frombuffer(b''.join(pickelized), dtype=np.int8)
    
    data_out, counts_out = self.Take_v(joined, counts_in)
    
    out = []
    r_start = 0
    for size in counts_out:
      out.append(pickle.loads(data_out[r_start:r_start+size].tobytes()))
      r_start += size
    return out


  def Put_into(self, data_in, data_out):
    counts_in  = data_in.size  // self.pn if self.pn != 0 else 0
    counts_out = data_out.size // self.dn if self.dn != 0 else 0

    if data_in.size - counts_in*self.pn != 0:
      raise ValueError("Input data size is not a multiple of managed idx")
    if data_out.size - counts_out*self.dn != 0:
      raise ValueError("Output data size is not a multiple of requested idx")
    if counts_in != counts_out and counts_in*counts_out != 0:
      raise ValueError("Input and output counts does not match")

    send_buff = np.empty(counts_in*self.part_write_pos.size, dtype=data_out.dtype)
    recv_buff = np.empty(counts_out*self.dist_counts.sum(),  dtype=data_in.dtype)

    if counts_in == 1:
      send_buff[self.part_write_pos] = data_in
    else:
      pull_idx = counts_in*self.part_write_pos
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

  def Put(self, data_in):
    counts  = data_in.size  // self.pn if self.pn != 0 else 0
    if self.empty_part:
      counts = self.comm.allreduce(counts, MPI.MAX)

    data_out = np.empty(counts*self.dn, data_in.dtype)
    self.Put_into(data_in, data_out)
    return data_out

  def Put_v(self, data_in, counts_in):
    # Variable stride

    assert isinstance(counts_in, np.ndarray)
    assert counts_in.size == self.pn
    assert counts_in.dtype == int
    assert data_in.size == counts_in.sum()

    # Exchange counts_in
    # _ : in all_to_all layout. Do not call Put because we need the intermediate layout
    st = time.time()
    _counts_in = np.empty(self.part_write_pos.size, dtype=int)
    _counts_out = np.empty(self.dist_counts.sum(),  dtype=int)
    _counts_in[self.part_write_pos] = counts_in

    self.comm.Alltoallv((_counts_in, self.part_counts), 
                        (_counts_out, self.dist_counts))
    counts_out  = np.zeros(self.dn, dtype=int)
    counts_out[self.dist_select_idx] = _counts_out
    ed = time.time()
    if self.comm.rank == 0:  print("  exch counts", ed-st)

    # Count the actual number of items to send/recv, using stride array
    # (this is the partial sum of portion of the stride array related to the given rank)
    st = time.time()
    send_counts = np.empty(self.comm.Get_size(), int)
    recv_counts = np.empty(self.comm.Get_size(), int)
    idx_send = idx_recv = 0
    for i in range(self.comm.Get_size()):
      send_counts[i] = _counts_in [idx_send:idx_send+self.part_counts[i]].sum()
      recv_counts[i] = _counts_out[idx_recv:idx_recv+self.dist_counts[i]].sum()
      idx_send += self.part_counts[i]
      idx_recv += self.dist_counts[i]
    ed = time.time()
    if self.comm.rank == 0:  print("  count send/recv", ed-st)

    # Prepare send buffer (put data in alltoall layout)
    inv_sort_idx = np.empty(self.part_write_pos.size, dtype=int)
    inv_sort_idx[self.part_write_pos] = np.arange(self.part_write_pos.size)

    st = time.time()
    #data_in_displ = np.empty(counts_in.size+1, dtype=int)
    #data_in_displ[0] = 0
    #np.cumsum(counts_in, out=data_in_displ[1:])
    send_buff = np.empty(send_counts.sum(), data_in.dtype)
    #w_start = w_end = 0
    #for idx in inv_sort_idx:
      #w_end = w_start + counts_in[idx]
      #send_buff[w_start:w_end] = data_in[data_in_displ[idx]:data_in_displ[idx+1]]
      #w_start = w_end
    take_strided(counts_in, data_in, inv_sort_idx, send_buff)
    ed = time.time()
    if self.comm.rank == 0:  print("  prepare send buffer", ed-st)

    # Exchange data buffer
    st = time.time()
    recv_buff = np.empty(recv_counts.sum(), data_in.dtype)
    self.comm.Alltoallv((send_buff, send_counts), (recv_buff, recv_counts))
    ed = time.time()
    if self.comm.rank == 0:  print("  exch data buff", ed-st)

    # Post treat recv buffer (data arrive in mpi layout, put it in requested layout)
    st = time.time()
    data_out = np.empty(counts_out.sum(), data_in.dtype)
    put_strided(data_out, self.dist_select_idx, counts_out, _counts_out, recv_buff)
    ed = time.time()
    if self.comm.rank == 0:  print("  write recv buff", ed-st)

    # TEST
    """
    data_out_displ = np.empty(counts.size+1, dtype=int)
    data_out_displ[0] = 0
    np.cumsum(counts, out=data_out_displ[1:])
    w_start = w_end = 0
    for idx in select_idx:
      w_end = w_start + counts[idx]
      buff[w_start:w_end] = data_in[displ[idx]:displ[idx+1]]
      w_start = w_end
    """
    # END TEST
    
    return data_out, counts_out


  def put(self, data_in):
    """
    Put implementation for any python object.
    data_in must be a list of size dn on each rank.
    """
    pickelized = [pickle.dumps(data) for data in data_in]
    counts_in  = np.array([len(p) for p in pickelized], dtype=int)

    joined = np.frombuffer(b''.join(pickelized), dtype=np.int8)
    
    data_out, counts_out = self.Put_v(joined, counts_in)
    
    out = []
    r_start = 0
    for size in counts_out:
      if size != 0:
        out.append(pickle.loads(data_out[r_start:r_start+size].tobytes()))
      else:
        out.append(None)
      r_start += size
    return out

