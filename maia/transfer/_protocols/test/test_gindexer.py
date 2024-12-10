import pytest
import pytest_parallel

import numpy as np

from maia.transfer._protocols.g_indexer import GIndexer, GIndexer_m

@pytest_parallel.mark.parallel(4)
class Test_g_indexer:

  # GIndexer wraps MPI AllToAll exchanges to perfom read/write operations
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
    return GIndexer(distri, g_idx, comm)

  def test_create(self, comm):
    DI = self.init_p(comm)
    assert DI.empty_dist == True
    assert DI.empty_part == True
    excepted_counts = [[4,1,1,0,1],
                       [],
                       [0,1,0],
                       [1,2,0,1]
                      ][comm.rank]

    assert np.array_equal(DI.access_counts, excepted_counts)

  def test_python_obj(self, comm):
    DI = self.init_p(comm)

    # Accessing python objects: each rank holds a part of a global list, 
    # depending of which indices its manages

    data_in = [['a', 'b', 'letter c', 'd', 'e'], # Indices 1...5
              [], # Indices 0...0
              ['f', ['a', 'list', 'of', 'g'], 'h'], #Indices 6...8
              ['i', 42.0, 'k', 'l'] #Indices 9...12
              ][comm.rank]

    # We can access the desired indices of the global list with take function
    data_out = DI.take(data_in) 
    expected_out = [['a', 'letter c', 'e', ['a','list','of','g'],'i'], 
                    ['l', 42.0], 
                    [],
                    ['a', 'a', 42.0, 'a', 'b']
                    ][comm.rank]
    assert data_out == expected_out

    # When using put function, we enter with data sized and organized as the requested indices

    data_in = [['a', 'letter c', 'e', 'f', ['a', 'list', 'of', 'h']], # Values to put at indices 1,3,5,7,9
              ['l', 42.0], # Values to put at indices 12,10
              [], # No data to put, since rank 2 access to no indices
              ['a', 'aaa', 42.0, 'aaaa', 'b']
              ][comm.rank] # Values to put at indices 1,1,10,1,2

    data_out = DI.put(data_in)


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
    DI = self.init_p(comm)
    # Accessing buffer objects : following mpi4py convention, buffer objects
    # can be used with the uppercase counterpart of the functions :

    data_in = [np.array([10., 20., 30., 40., 50]), # Indices 1...5
              np.empty(0, float), # Indices 0...0
              np.array([60., 70, 80]), #Indices 6...8
              np.array([90., 100, 110, 120])
              ][comm.rank] #Indices 9...12

    data_out = DI.Take(data_in) 
    expected_out = [np.array([10.,30,50,70,90]),
                    np.array([120.,100]),
                    np.array([], float),
                    np.array([10.,10,100,10,20]),
                    ][comm.rank]
    assert np.allclose(data_out, expected_out)

    # We can also put result in a pre allocated buffer with the _into variant ;
    # note that out buffer must have good size and datatype
    data_out = np.empty(self.g_idx.size, float)
    DI.Take_into(data_in, data_out)
    assert np.allclose(data_out, expected_out)

    # We can put values from a buffer object, using uppercase Put function :

    data_in = [np.array([10., 30, 50, 70, 90]), # Values to put at indices 1,3,5,7,9
              np.array([120., 100]), # Values to put at indices 12,10
              np.array([], float), # No data to put, since rank 2 access to no indices
              np.array([10.,10,100,10,20]) # Values to put at indices 1,1,10,1,2
              ][comm.rank]

    data_out = DI.Put(data_in)

    # When using buffer variants, unfilled indices take a random (unitialized) value
    # Again, we can put the result in a preallocated buffer:
    dn_size = self.distri[comm.rank+1] - self.distri[comm.rank]
    data_out = -1*np.ones(dn_size, float)
    DI.Put_into(data_in, data_out)

    expected_out = [np.array([10., 20, 30, -1, 50]),
                    np.array([], float),
                    np.array([-1., 70, -1]),
                    np.array([90., 100., -1, 120]),
                    ][comm.rank]
    assert np.allclose(data_out, expected_out)



    
   
  def test_cst_buffer(self, comm):
    DI = self.init_p(comm)

    # Input buffer is allowed to have more than  1 element per index : however, this number 
    # must remain constant when using Take / Take_into :

    data_in = [
      np.array([10.,15, 20,25, 30,35, 40,45, 50,55]), # Indices 1...5, with 2 values per indices
      np.array([], float), # Indices 0...0, with 2 values per indices
      np.array([60.,65, 70,75, 80,85]), #Indices 6...8, with 2 values per indices
      np.array([90.,95, 100,105, 110,115, 120,125])
    ][comm.rank] #Indices 9...12, with 2 values per indices

    data_out = np.empty(2*self.g_idx.size, float)  # Out buffer will store 2 values per requested idx
    DI.Take_into(data_in, data_out) 

    expected_out = [np.array([10.,15, 30,35, 50,55, 70,75, 90,95]),
                    np.array([120.,125, 100,105]),
                    np.array([], float),
                    np.array([10.,15, 10,15, 100,105, 10,15, 20,25]),
                    ][comm.rank]
    assert np.allclose(data_out, expected_out)

    # Similar w/o preallocated buffer:
    data_out = DI.Take(data_in)
    assert np.allclose(data_out, expected_out)

    # and write more than 1 element per index 
    data_in = [
      np.array([10.,15, 30,35, 50,55, 70,75, 90,95]), # Values to put at indices 1,3,5,7,9, with 2 values per index
      np.array([120.,125, 100,105]), # Values to put at indices 12,10, with 2 values per index
      np.array([], float), # No data to put, since rank 2 access to no indices
      np.array([10.,15, 10,15, 100,105, 10,15, 20,25]) # Values to put at indices 1,1,10,1,2, with 2 values per index
    ][comm.rank]

    dn_size = self.distri[comm.rank+1] - self.distri[comm.rank]
    data_out = -1*np.ones(2*dn_size, float)
    DI.Put_into(data_in, data_out)
    data_out2 = DI.Put(data_in) #Equivalent w/o preallocated buffer

    expected_out = [np.array([10.,15, 20,25, 30,35, -1,-1, 50,55]),
                    np.array([], float),
                    np.array([-1.,-1, 70,75, -1,-1]),
                    np.array([90.,95, 100.,105, -1,-1, 120,125]),
                    ][comm.rank]
    assert np.allclose(data_out, expected_out)

  def test_variable_buffer(self, comm):
    rank = comm.rank
    DI = self.init_p(comm)

    # When it comes to variables sizes, Take_v must be used (as we would use
    # AllToAllv with mpi4py). The expected input is now a counting array 
    # (of size #managed idx) + a buffer of size counts.sum()

    if rank == 0:
      counts_in = np.array([0,1,0,1,2]) # Indices 1...5, with varibles values per indices
      data_in = np.array([20., 40, 50,55])  # 4 values in total
    elif rank == 1:
      counts_in = np.array([], int) # Indices 0...0, with variables values per indices
      data_in = np.empty(0, float) 
    elif rank == 2:
      counts_in = np.array([1,0,1]) #Indices 6...8, with variables values per indices
      data_in = np.array([60., 80])  # 2 values in total
    elif rank == 3:
      counts_in = np.array([0,2,0,1]) #Indices 9...12, with variables values per indices
      data_in = np.array([100.,105,  120])  # 3 values in total


    # Note that the function now return two array : a counting array for 
    # each requested idx + a buffer of size counts.sum()
    # If we requested an index for which no data has been provided
    # by the managing process (counts_in = 0), we will simply get
    # not data for this index (counts_out = 0)
    data_out, counts_out  = DI.Take_v((data_in, counts_in))

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
      counts_in = np.array([0,0,2,0,0])  # Sizes of data to put at indices 1,3,5,7,9
      data_in = np.array([50.,55]) 
    elif rank == 1:
      counts_in = np.array([1,2]) # Sizes of data to put at indices 12,10
      data_in = np.array([120., 100,105]) 
    elif rank == 2:
      counts_in = np.array([], int) # No data to put, since rank 2 access to no indices
      data_in = np.empty(0, float) 
    elif rank == 3:
      counts_in = np.array([0,0,2,0,1]) # Size of data to put at indices 1,1,10,1,2
      data_in =np.array([100.,105, 20]) 


    data_out, counts_out = DI.Put_v((data_in, counts_in))

    expected_out = [
      (np.array([0,1,0,0,2]), np.array([20., 50.,55])),
      (np.array([], int), np.array([], float)),
      (np.array([0,0,0]), np.array([], float)),
      (np.array([0,2,0,1]), np.array([100.,105,  120])),
    ][rank]
                    
    assert np.array_equal(counts_out, expected_out[0])
    assert np.allclose(data_out, expected_out[1])

  def test_failures(self, comm):
    # Creating a DI with an 'out of bounds' index should raise :
    distri = np.array([0, 5, 5, 8, 12])
    g_idx = [np.array([0,2,4,6,8]),
             np.array([11,9]),
             np.array([], int),      # Rank 2 requests no indices !
             np.array([0,0,14,0,1]) # Indices can be requested more than once
            ][comm.rank]
    with pytest.raises(IndexError):
      DI = GIndexer(distri, g_idx, comm)



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

  GI = GIndexer_m(distri, gnum, comm)
  assert (GI.Put(GI.Take(field)) == field).all()




@pytest_parallel.mark.parallel(8)
def test_perfo(comm):

  import time

  distri = 10000 * np.arange(comm.size+1)

  gnum = np.random.randint(0, distri[-1], 10000000)
  gnum1 = gnum+1
  comm.barrier()
  st = time.time()
  DI = GIndexer(distri, gnum, comm)
  ed = time.time()
  if comm.rank == 0:
    print("Creation time GIndexer", ed-st)

  import Pypdm.Pypdm as PDM
  #PDM.BlockToPart
  distri = distri.astype(np.int32)
  gnum1  = gnum1.astype(np.int32)
  comm.barrier()
  st = time.time()
  BTP = PDM.BlockToPart(distri, comm, [gnum1], 1)
  ed = time.time()
  if comm.rank == 0:
    print("Creation time BlockToPart", ed-st)

  comm.barrier()
  st = time.time()
  PTB = PDM.PartToBlock(comm, [gnum1], pWeight=None, partN=1, t_distrib=0, t_post=1, userDistribution=distri)
  ed = time.time()
  if comm.rank == 0:
    print("Creation time PartToBlock", ed-st)


  dn = distri[comm.rank+1] - distri[comm.rank]
  data_in = np.empty(dn, float)

  st = time.time()
  data_out = DI.Take(data_in)
  ed = time.time()
  if comm.rank == 0:
    print("Exchange time Take", ed-st)

  st = time.time()
  _, data_outO = BTP.exchange_field(data_in)
  ed = time.time()
  if comm.rank == 0:
    print("Exchange time BTP", ed-st)

  st = time.time()
  DI.Put(data_out)
  ed = time.time()
  if comm.rank == 0:
    print("Exchange time Put", ed-st)

  st = time.time()
  PTB.exchange_field(data_outO)
  ed = time.time()
  if comm.rank == 0:
    print("Exchange time PTB", ed-st)


  counts_in = np.random.randint(0,4+1,dn)
  data_in = np.empty(counts_in.sum(), float)
  st = time.time()
  data_out, counts_out = DI.Take_v((data_in, counts_in))
  ed = time.time()
  if comm.rank == 0:
    print("Exchange time Take_v", ed-st)

  counts_in = counts_in.astype(np.int32)
  st = time.time()
  counts_outO, data_outO = BTP.exchange_field(data_in, counts_in)
  ed = time.time()
  if comm.rank == 0:
    print("Exchange time BTPvar", ed-st)

  st = time.time()
  DI.Put_v((data_out, counts_out))
  ed = time.time()
  if comm.rank == 0:
    print("Exchange time Put_v", ed-st)

  st = time.time()
  PTB.exchange_field(data_outO, counts_outO)
  ed = time.time()
  if comm.rank == 0:
    print("Exchange time PTBvar", ed-st)


