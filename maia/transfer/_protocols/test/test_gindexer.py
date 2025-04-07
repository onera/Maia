import pytest
import pytest_parallel

import numpy as np

from maia.transfer._protocols           import g_indexer
from maia.transfer._protocols.g_indexer import GlobalIndexer, GlobalMultiIndexer, ReduceOp

def test_put_strided():
    idx = np.array([1,0,2,1])
    counts_out = np.array([3,2,1], int)
    data_out   = np.empty(6, float)

    counts_in = np.array([2,3,1,1], int)  # => Only one compatible stride for idx 1
    data_in = np.array([1.1, 1.2,   2.1, 2.2, 2.3,   3.1,   4.1])
    data_out.fill(-1)

    g_indexer.put_strided(data_out, counts_out, idx, counts_in, data_in)
    assert (data_out == np.array([2.1,2.2,2.3,  1.1,1.2,  3.1])).all()

    counts_in = np.array([2,4,1,1], int) # => No compatible stride for idx 0
    data_in = np.array([1.1, 1.2,   2.1, 2.2, 2.3, 2.4,   3.1,   4.1])
    data_out.fill(-1)

    g_indexer.put_strided(data_out, counts_out, idx, counts_in, data_in)
    assert (data_out == np.array([-1,-1.,-1.,  1.1,1.2,  3.1])).all()

    counts_in = np.array([2,3,1,2], int) # => Two compatible stride for idx 1 (last is keep)
    data_in = np.array([1.1, 1.2,   2.1, 2.2, 2.3,   3.1,   4.1, 4.2]) 
    data_out.fill(-1)

    g_indexer.put_strided(data_out, counts_out, idx, counts_in, data_in)
    assert (data_out == np.array([2.1,2.2,2.3,  4.1,4.2,  3.1])).all()

    # In extend mode
    counts_out = np.array([3,4,1], int)
    data_out   = np.empty(8, float)
    counts_in = np.array([2,3,1,2], int) 
    data_in = np.array([1.1, 1.2,   2.1, 2.2, 2.3,   3.1,   4.1, 4.2])
    g_indexer.put_strided(data_out, counts_out, idx, counts_in, data_in, extend=True)
    assert (data_out == np.array([2.1,2.2,2.3,  1.1,1.2,4.1,4.2,  3.1])).all()

def test_guess_reduce_dt_and_identity():
  assert g_indexer._guess_reduce_dt_and_identity('f', g_indexer.ReduceOp.LAND) == ('?', True)
  assert g_indexer._guess_reduce_dt_and_identity('i', g_indexer.ReduceOp.SUM)  == ('i', 0)
  assert g_indexer._guess_reduce_dt_and_identity('?', g_indexer.ReduceOp.SUM)  == ('l', 0)
  assert g_indexer._guess_reduce_dt_and_identity('d', g_indexer.ReduceOp.MIN)  == ('d', np.inf)
  assert g_indexer._guess_reduce_dt_and_identity('f', g_indexer.ReduceOp.MAX)  == ('f', -np.inf)
  assert g_indexer._guess_reduce_dt_and_identity('i', g_indexer.ReduceOp.MAX)  == ('i', np.iinfo(np.int32).min)
  with pytest.raises(ValueError):
    assert g_indexer._guess_reduce_dt_and_identity('?', g_indexer.ReduceOp.PROD)
  with pytest.raises(ValueError):
    assert g_indexer._guess_reduce_dt_and_identity('d', g_indexer.ReduceOp.BAND)

@pytest_parallel.mark.parallel(4)
class Test_g_indexer:

  # GlobalIndexer wraps MPI AllToAll exchanges to perfom read/write operations
  # from global indices on a distributed array

  # To create the object, the following data are required :
  #  - for each rank, a list of the global indices he want to access. Note that
  #    the list can be empty and indices can occurs more than once
  #  - a global description of how the indices are distributed over the ranks

  def init_p(self, comm):
    distri = np.array([0, 5, 5, 8, 12])
    g_idx = [np.array([0,2,4,6,8]),
             np.array([11,9]),
             np.array([], int),      # Rank 2 requests no indices !
             np.array([0,0,9,0,1]) # Indices can be requested more than once
            ][comm.rank]

    self.g_idx = g_idx
    self.distri = distri
    return GlobalIndexer(distri, g_idx, comm)

  def test_create(self, comm):
    GI = self.init_p(comm)
    assert GI.empty_dist == True
    assert GI.empty_part == False
    excepted_counts = [[4,1,1,0,1],
                       [],
                       [0,1,0],
                       [1,2,0,1]
                      ][comm.rank]

    assert np.array_equal(GI.access_counts, excepted_counts)

  def test_python_obj(self, comm):
    GI = self.init_p(comm)

    # Accessing python objects: each rank holds a part of a global list, 
    # depending of which indices its manages

    data_in = [['a', 'b', 'letter c', 'd', 'e'], # Indices 0...4
              [], # Indices 0...0
              ['f', ['a', 'list', 'of', 'g'], 'h'], #Indices 5...7
              ['i', 42.0, 'k', 'l'] #Indices 8...11
              ][comm.rank]

    # We can access the desired indices of the global list with take function
    data_out = GI.take(data_in) 
    expected_out = [['a', 'letter c', 'e', ['a','list','of','g'],'i'], 
                    ['l', 42.0], 
                    [],
                    ['a', 'a', 42.0, 'a', 'b']
                    ][comm.rank]
    assert data_out == expected_out

    # When using put function, we enter with data sized and organized as the requested indices

    data_in = [['a', 'letter c', 'e', 'f', ['a', 'list', 'of', 'h']], # Values to put at indices 0,2,4,6,8
              ['l', 42.0], # Values to put at indices 11,9
              [], # No data to put, since rank 2 access to no indices
              ['a', 'aaa', 42.0, 'aaaa', 'b']
              ][comm.rank] # Values to put at indices 0,0,9,0,1

    data_out = GI.put(data_in)


    # Note that :
    # - global index for which nobody put a value will take the None value
    # - global index for with several values are put will only take the last one
    expected_out = [['aaaa', 'b', 'letter c', None, 'e'], 
                    [], 
                    [None, 'f', None],
                    [['a','list','of','h'],42.0,None,'l']
                    ][comm.rank]
    assert data_out == expected_out

  def test_buffer(self, comm):
    GI = self.init_p(comm)
    # Accessing buffer objects : following mpi4py convention, buffer objects
    # can be used with the uppercase counterpart of the functions :

    data_in = [np.array([10., 20., 30., 40., 50]), # Indices 0...4
              np.empty(0, float), # Indices 0...0
              np.array([60., 70, 80]), #Indices 5...7
              np.array([90., 100, 110, 120])
              ][comm.rank] #Indices 8...11

    data_out = GI.Take(data_in) 
    expected_out = [np.array([10.,30,50,70,90]),
                    np.array([120.,100]),
                    np.array([], float),
                    np.array([10.,10,100,10,20]),
                    ][comm.rank]
    assert np.allclose(data_out, expected_out)

    # We can also put result in a pre allocated buffer;
    # note that out buffer must have good size and datatype
    data_out = np.empty(self.g_idx.size, float)
    GI.Take(data_in, data_out)
    assert np.allclose(data_out, expected_out)

    # We can put values from a buffer object, using uppercase Put function :

    data_in = [np.array([10., 30, 50, 70, 90]), # Values to put at indices 0,2,4,6,8
              np.array([120., 100]), # Values to put at indices 11,9
              np.array([], float), # No data to put, since rank 2 access to no indices
              np.array([10.,10,100,10,20]) # Values to put at indices 0,0,9,0,1
              ][comm.rank]

    data_out = GI.Put(data_in)

    # When using buffer variants, unfilled indices take a random (unitialized) value
    # Again, we can put the result in a preallocated buffer:
    dn_size = self.distri[comm.rank+1] - self.distri[comm.rank]
    data_out = -1*np.ones(dn_size, float)
    GI.Put(data_in, data_out)

    expected_out = [np.array([10., 20, 30, -1, 50]),
                    np.array([], float),
                    np.array([-1., 70, -1]),
                    np.array([90., 100., -1, 120]),
                    ][comm.rank]
    assert np.allclose(data_out, expected_out)



    
   
  def test_cst_buffer(self, comm):
    GI = self.init_p(comm)

    # Input buffer is allowed to have more than  1 element per index : however, this number 
    # must remain constant:

    data_in = [
      np.array([10.,15, 20,25, 30,35, 40,45, 50,55]), # Indices 0...4, with 2 values per indices
      np.array([], float), # Indices 0...0, with 2 values per indices
      np.array([60.,65, 70,75, 80,85]), #Indices 5...7, with 2 values per indices
      np.array([90.,95, 100,105, 110,115, 120,125])
    ][comm.rank] #Indices 8...11, with 2 values per indices

    data_out = np.empty(2*self.g_idx.size, float)  # Out buffer will store 2 values per requested idx
    GI.Take(data_in, data_out, count=2) 

    expected_out = [np.array([10.,15, 30,35, 50,55, 70,75, 90,95]),
                    np.array([120.,125, 100,105]),
                    np.array([], float),
                    np.array([10.,15, 10,15, 100,105, 10,15, 20,25]),
                    ][comm.rank]
    assert np.allclose(data_out, expected_out)

    # Similar w/o preallocated buffer:
    data_out = GI.Take(data_in, count=2)
    assert np.allclose(data_out, expected_out)

    # and write more than 1 element per index 
    data_in = [
      np.array([10.,15, 30,35, 50,55, 70,75, 90,95]), # Values to put at indices 0,2,4,6,8, with 2 values per index
      np.array([120.,125, 100,105]), # Values to put at indices 11,9, with 2 values per index
      np.array([], float), # No data to put, since rank 2 access to no indices
      np.array([10.,15, 10,15, 100,105, 10,15, 20,25]) # Values to put at indices 0,0,9,0,1, with 2 values per index
    ][comm.rank]

    dn_size = self.distri[comm.rank+1] - self.distri[comm.rank]
    data_out = -1*np.ones(2*dn_size, float)
    GI.Put(data_in, data_out, count=2)
    data_out2 = GI.Put(data_in, count=2) #Equivalent w/o preallocated buffer

    expected_out = [np.array([10.,15, 20,25, 30,35, -1,-1, 50,55]),
                    np.array([], float),
                    np.array([-1.,-1, 70,75, -1,-1]),
                    np.array([90.,95, 100.,105, -1,-1, 120,125]),
                    ][comm.rank]
    assert np.allclose(data_out, expected_out)

  def test_variable_buffer(self, comm):
    rank = comm.rank
    GI = self.init_p(comm)

    # When it comes to variables sizes, Take_v must be used (as we would use
    # AllToAllv with mpi4py). The expected input is now a counting array 
    # (of size #managed idx) + a buffer of size counts.sum()

    if rank == 0:
      counts_in = np.array([0,1,0,1,2]) # Indices 0...4, with varibles values per indices
      data_in = np.array([20., 40, 50,55])  # 4 values in total
    elif rank == 1:
      counts_in = np.array([], int) # Indices 0...0, with variables values per indices
      data_in = np.empty(0, float) 
    elif rank == 2:
      counts_in = np.array([1,0,1]) #Indices 5...7, with variables values per indices
      data_in = np.array([60., 80])  # 2 values in total
    elif rank == 3:
      counts_in = np.array([0,2,0,1]) #Indices 8...11, with variables values per indices
      data_in = np.array([100.,105,  120])  # 3 values in total


    # Note that the function now return two array : a counting array for 
    # each requested idx + a buffer of size counts.sum()
    # If we requested an index for which no data has been provided
    # by the managing process (counts_in = 0), we will simply get
    # not data for this index (counts_out = 0)
    counts_out, data_out  = GI.Take_v((counts_in, data_in))

    # As for Take method, we can use a preallocated buffer, in this 
    # case the counts_out array must be already filled and data_out must have
    # relevant size 
    data_out2 = np.zeros_like(data_out)
    GI.Take_v((counts_in, data_in), (counts_out, data_out2)) 
    assert np.array_equal(data_out, data_out2)

    expected_out = [
      (np.array([0,0,2,0,0]), np.array([50.,55])),
      (np.array([1,2]), np.array([120., 100,105])),
      (np.array([], int), np.array([], float)),
      (np.array([0,0,2,0,1]), np.array([100.,105,  20])),
    ][rank]
                    
    assert np.array_equal(counts_out, expected_out[0])
    assert np.allclose(data_out, expected_out[1])

    # As before, variables sizes are managed throught the Put_v function, who
    # expects a counting array (of size #lngn) + a buffer of size counts.sum

    if rank == 0:
      counts_in = np.array([0,0,2,0,0])  # Sizes of data to put at indices 0,2,4,6,8
      data_in = np.array([50.,55]) 
    elif rank == 1:
      counts_in = np.array([1,2]) # Sizes of data to put at indices 11,9
      data_in = np.array([120., 100,105]) 
    elif rank == 2:
      counts_in = np.array([], int) # No data to put, since rank 2 access to no indices
      data_in = np.empty(0, float) 
    elif rank == 3:
      counts_in = np.array([0,0,2,0,1]) # Size of data to put at indices 0,0,9,0,1
      data_in =np.array([100.,105, 20]) 


    counts_out, data_out = GI.Put_v((counts_in, data_in))

    expected_out = [
      (np.array([0,1,0,0,2]), np.array([20., 50.,55])),
      (np.array([], int), np.array([], float)),
      (np.array([0,0,0]), np.array([], float)),
      (np.array([0,2,0,1]), np.array([100.,105,  120])),
    ][rank]
                    
    assert np.array_equal(counts_out, expected_out[0])
    assert np.allclose(data_out, expected_out[1])

    # The flag extend allows to keep all the data coming from a given gnum, in appartion order
    counts_out_app, data_out_app = GI.Put_v((counts_in, data_in), extend=True)
    expected_out_app = [
      (np.array([0,1,0,0,2]), np.array([20., 50.,55])),
      (np.array([], int), np.array([], float)),
      (np.array([0,0,0]), np.array([], float)),
      (np.array([0,4,0,1]), np.array([100.,105, 100.,105,  120])),
    ][rank]
    assert np.array_equal(counts_out_app, expected_out_app[0])
    assert np.allclose(data_out_app, expected_out_app[1])

    # As for Put method, we can use a preallocated buffer, in this 
    # case the counts_out array must be already filled and data_out must have
    # relevant size 
    data_out2 = np.zeros_like(data_out)
    GI.Put_v((counts_in, data_in), (counts_out, data_out2)) 
    assert np.array_equal(data_out, data_out2)

    # If we use extend + preallocated mode, counts_out can be compute with ReduceOp = SUM
    counts_out = GI.Put(counts_in, reduce=ReduceOp.SUM)
    data_out2 = np.zeros(counts_out.sum(), data_in.dtype)
    GI.Put_v((counts_in, data_in), (counts_out, data_out2), extend=True) 
    assert np.array_equal(data_out2, data_out_app)


  def test_failures(self, comm):
    # Creating a GI with an 'out of bounds' index should raise :
    distri = np.array([0, 5, 5, 8, 12])
    g_idx = [np.array([0,2,4,6,8]),
             np.array([11,9]),
             np.array([], int),      # Rank 2 requests no indices !
             np.array([0,0,14,0,1]) # Indices can be requested more than once
            ][comm.rank]
    with pytest.raises(IndexError):
      GI = GlobalIndexer(distri, g_idx, comm)



@pytest_parallel.mark.parallel(3)
def test_empty_part(comm):
  # This is to ensure that exchanges work well even in 0-size part case,
  # since cst_stride has to be guess in this case

  distri = np.array([0,1,2,2])
  gnum = [[np.array([], int)],
          [],
          [np.array([0,1], int), np.array([], int)]][comm.rank]
  field = [np.array([42], np.int32),
           np.array([24], np.int32),
           np.array([], np.int32)][comm.rank]

  GI = GlobalMultiIndexer(distri, gnum, comm)
  assert (GI.Put(GI.Take(field)) == field).all()

