import pytest_parallel

import sys
import numpy as np
import mpi4py.MPI as MPI

from maia.utils import par_utils, py_utils
from maia.utils.parallel import algo as par_algo

def py_version():
    version_info = sys.version_info
    return (version_info.major, version_info.minor, version_info.micro)

@pytest_parallel.mark.parallel(3)
def test_dist_set_difference(comm):
  if comm.Get_rank() == 0:
    ids = np.array([34,19])
    others = [np.array([4,9,2,11])]
    expected = np.array([12, 19])
  if comm.Get_rank() == 1:
    ids = np.array([11,12])
    others = [np.array([11,5])]
    expected = np.array([33])
  if comm.Get_rank() == 2:
    ids = np.array([5, 19, 33])
    others = [np.empty(0, dtype=int)]
    expected = np.array([34])

  assert (par_algo.dist_set_difference(ids, others, comm) == expected).all()

@pytest_parallel.mark.parallel(3)
def test_DistSorter(comm):
  if comm.Get_rank() == 0:
    key = np.array([34,19])
    data = np.array([1.,2.])
    expected = np.array([3., 2.])
  if comm.Get_rank() == 1:
    key = np.array([3,40,42,27])
    data = np.array([3., 4., 5., 6.])
    expected = np.array([6.,1.])
  if comm.Get_rank() == 2:
    key = np.empty(0, int)
    data = np.empty(0, float)
    expected = np.array([4.,5.])
  #Key order is 3,19,27,34,40,42

  sorter = par_algo.DistSorter(key, comm)
  assert (sorter.sort(data) == expected).all()


@pytest_parallel.mark.parallel([1,4])
def test_compute_gnum_s(comm):
  # Simple test with strings  
  keys = [['Pear', 'Banana', 'Peach', 'Banana'],
          [],
          ['Watermelon', 'Banana'],
          ['Melon', 'Peach', 'Tomato']]
  expected_gnum = [[5,1,4,1],
                   [],
                   [2,1],
                   [6,4,3]]

  if comm.Get_size() == 4:
    rank_keys = keys[comm.Get_rank()]
    rank_gnum = expected_gnum[comm.Get_rank()]
  else: #Results should be // independant
    rank_keys = py_utils.to_flat_list(keys)
    rank_gnum = py_utils.to_flat_list(expected_gnum)

  computed_gnum = par_algo.compute_gnum(rank_keys, comm)

  if py_version()[:2] == (3,9): # Gnum seems to be version dependant (pickles / hashlib)
    assert (computed_gnum == rank_gnum).all()

  assert len(computed_gnum) == len(rank_keys)
  assert comm.allreduce(min(computed_gnum) if rank_keys else 1000, MPI.MIN) == 1
  assert comm.allreduce(max(computed_gnum) if rank_keys else    0, MPI.MAX) == 6

@pytest_parallel.mark.parallel(2)
def test_compute_gnum_o(comm):
  # Test with various objects
  if comm.Get_rank() == 0:
    keys = [test_compute_gnum_s, 'Banana', np.array([1,2,3], np.int32)]
    expected_gnum = [5,1,2] #3,5,2
  elif comm.Get_rank() == 1:
    keys = [None, False, test_compute_gnum_s, 'Banana']
    expected_gnum = [4,3,5,1] #[4,1,3,5]

  computed_gnum = par_algo.compute_gnum(keys, comm)

  if py_version()[:2] == (3,9): # Gnum seems to be version dependant (pickles / hashlib)
    assert (computed_gnum == expected_gnum).all()

  assert len(computed_gnum) == len(keys)
  assert comm.allreduce(min(computed_gnum), MPI.MIN) == 1
  assert comm.allreduce(max(computed_gnum), MPI.MAX) == 5

  gathered = comm.gather(computed_gnum, root=0)
  if comm.Get_rank() == 0:
    assert gathered[0][0] == gathered[1][2]
    assert gathered[0][1] == gathered[1][3]


@pytest_parallel.mark.parallel([1,2,3])
def test_is_unique_strided_serialized(comm):
  n_elt  = 6
  stride = 3
  array  = np.array([1,2,3, 4,5,6, 7,8,9, 3,1,2, 2,1,7, 7,6,8])
  unique = np.array([False,  True,  True, False,  True,  True])
  
  distri = par_utils.uniform_distribution(n_elt, comm)
  array  = array [distri[0]*stride:distri[1]*stride]
  expected = unique[distri[0]       :distri[1]       ]

  unique = par_algo.is_unique_strided_serialized(array, stride, comm)
  assert np.array_equal(unique,expected)


@pytest_parallel.mark.parallel([1,2,3])
def test_is_unique_strided(comm):
  n_elt  = 6
  stride = 3
  array  = np.array([1,2,3, 4,5,6, 7,8,9, 3,1,2, 2,1,7, 7,6,8])
  unique = np.array([False,  True,  True, False,  True,  True])
  
  distri = par_utils.uniform_distribution(n_elt, comm)
  array  = array [distri[0]*stride:distri[1]*stride]
  expected = unique[distri[0]       :distri[1]       ]

  unique = par_algo.is_unique_strided(array, stride, comm)
  assert np.array_equal(unique ,expected)
