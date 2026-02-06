import pytest
import pytest_parallel

import numpy as np
from mpi4py import MPI

import maia.pytree as PT
from maia import npy_pdm_gnum_dtype

from maia.utils import py_utils
from maia.utils.parallel import utils

@pytest_parallel.mark.parallel(3)
def test_gathering_distribution(comm):
  for i_rank in range(0,3):
    distrib = utils.gathering_distribution(i_rank, 17, comm)
    assert isinstance(distrib, np.ndarray)
    assert distrib.dtype == npy_pdm_gnum_dtype
    if   comm.Get_rank()<i_rank:
      assert distrib[0]==0
      assert distrib[1]==0
      assert distrib[2]==17
    elif comm.Get_rank()==i_rank:
      assert distrib[0]==0
      assert distrib[1]==17
      assert distrib[2]==17
    else:
      assert distrib[0]==17
      assert distrib[1]==17
      assert distrib[2]==17
@pytest_parallel.mark.parallel(3)
def test_uniform_distribution(comm):
  distrib = utils.uniform_distribution(17, comm)
  assert isinstance(distrib, np.ndarray)
  assert distrib.dtype == npy_pdm_gnum_dtype
  assert (distrib[0:2] == py_utils.uniform_distribution_at(\
      17, comm.Get_rank(), comm.Get_size())).all()
  assert distrib[2] == 17

  if npy_pdm_gnum_dtype == np.int32:
    with pytest.raises(OverflowError):
      distrib = utils.uniform_distribution(2_500_000_000, comm)

@pytest_parallel.mark.parallel(3)
def test_dn_to_distribution(comm):
  dn = (comm.rank+1)**3 #1, 8, 27
  expt_distri_f = np.array([0, 1, 9, 36])
  distrib = utils.dn_to_distribution(dn, comm)
  assert isinstance(distrib, np.ndarray)
  assert distrib.dtype == npy_pdm_gnum_dtype
  assert (distrib == expt_distri_f[[comm.rank, comm.rank+1, comm.size]]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("weights", [False, True])
def test_distribution_from_gnum(weights, comm):
  if comm.rank == 0:
    lngn_list = [np.empty(0, npy_pdm_gnum_dtype), np.array([4,3,1,10], npy_pdm_gnum_dtype)]
    expt_distri = [0,2,10] if weights else [0,5,10]
  if comm.rank == 1:
    lngn_list = [np.array([2,3,1,1,5,4], npy_pdm_gnum_dtype)]
    expt_distri = [2,10,10] if weights else [5,10,10]
  assert (utils.distribution_from_gnum(lngn_list, comm, weights) == expt_distri).all()

@pytest_parallel.mark.parallel(3)
def test_partial_to_full_distribution(comm):
  if comm.Get_rank() == 0:
    partial_distrib_32 = np.array([0, 25, 75], dtype=np.int32)
    partial_distrib_64 = np.array([0, 25, 75], dtype=np.int64)
    partial_distrib_hole = np.array([0, 25, 75])
    partial_distrib_void = np.array([0, 0, 0])
  if comm.Get_rank() == 1:
    partial_distrib_32 = np.array([25, 55, 75], dtype=np.int32)
    partial_distrib_64 = np.array([25, 55, 75], dtype=np.int64)
    partial_distrib_hole = np.array([25, 25, 75])
    partial_distrib_void = np.array([0, 0, 0])
  if comm.Get_rank() == 2:
    partial_distrib_32 = np.array([55, 75, 75], dtype=np.int32)
    partial_distrib_64 = np.array([55, 75, 75], dtype=np.int64)
    partial_distrib_hole = np.array([25, 75, 75])
    partial_distrib_void = np.array([0, 0, 0])

  full_distri_32 = utils.partial_to_full_distribution(partial_distrib_32, comm)
  assert full_distri_32.dtype == np.int32
  assert (full_distri_32 == [0,25,55,75]).all()
  full_distri_64 = utils.partial_to_full_distribution(partial_distrib_64, comm)
  assert full_distri_64.dtype == np.int64
  assert (full_distri_64 == [0,25,55,75]).all()
  full_distri_hole = utils.partial_to_full_distribution(partial_distrib_hole, comm)
  assert (full_distri_hole == [0,25,25,75]).all()
  full_distri_void = utils.partial_to_full_distribution(partial_distrib_void, comm)
  assert (full_distri_void == [0,0,0,0]).all()

@pytest_parallel.mark.parallel(3)
def test_full_to_partial_distribution(comm):
  partial = utils.full_to_partial_distribution(np.array([0, 0, 0, 100], np.int32), comm)
  assert partial.dtype == np.int32
  if comm.rank < 2:
    assert (partial == [0,0,100]).all()
  else:
    assert (partial == [0,100,100]).all()

class Test_auto_expand_distri:
  
  @pytest_parallel.mark.parallel(3)
  def test_straightforward(self, comm):
    if comm.Get_rank() == 0:
      distri_partial = np.array([0, 10, 40])
    if comm.Get_rank() == 1:
      distri_partial = np.array([10, 20, 40])
    if comm.Get_rank() == 2:
      distri_partial = np.array([20,40,40])
    assert np.array_equal(utils.auto_expand_distri(distri_partial, comm), \
        np.array([0,10,20,40]))

    distri_full = np.array([0,10,20,40])
    assert utils.auto_expand_distri(distri_full, comm) is distri_full

  @pytest_parallel.mark.parallel(2)
  def test_corner_cases(self, comm):
    distri_partial = np.array([0, 10, 40]) if comm.Get_rank() == 0 else np.array([10, 40, 40])
    assert np.array_equal(utils.auto_expand_distri(distri_partial, comm), \
        np.array([0,10,40]))
    distri_partial = np.array([0, 40, 40]) if comm.Get_rank() == 0 else np.array([40, 40, 40])
    assert np.array_equal(utils.auto_expand_distri(distri_partial, comm), \
        np.array([0,40,40]))
    distri_partial = np.array([0, 0, 40]) if comm.Get_rank() == 0 else np.array([0, 40, 40])
    assert np.array_equal(utils.auto_expand_distri(distri_partial, comm), \
        np.array([0,0,40]))
    distri_partial = np.array([0, 0, 0]) if comm.Get_rank() == 0 else np.array([0, 0, 0])
    assert np.array_equal(utils.auto_expand_distri(distri_partial, comm), \
        np.array([0,0,0]))
    # Already full
    for distri_full in [[0,10,40], [0,40,40], [0,0,40], [0,0,0]]:
      _distri_full = np.array(distri_full)
      assert np.array_equal(utils.auto_expand_distri(_distri_full, comm), _distri_full)

@pytest_parallel.mark.parallel(3)
def test_is_same_distri(comm):
  distri1 = utils.uniform_distribution(100, comm)
  distri2 = utils.partial_to_full_distribution(utils.uniform_distribution(100, comm), comm)
  distri3 = utils.gathering_distribution(1, 100, comm)
  distri4 = utils.uniform_distribution(101, comm)
  assert utils.is_same_distri(distri1, distri2, comm)
  assert not utils.is_same_distri(distri1, distri3, comm)
  assert not utils.is_same_distri(distri1, distri4, comm)

@pytest_parallel.mark.parallel(3)
def test_gather_and_shift(comm):
  if comm.Get_rank() == 0:
    value = 6
  if comm.Get_rank() == 1:
    value = 9
  if comm.Get_rank() == 2:
    value = 2
  distri = utils.gather_and_shift(value, comm)
  assert distri.dtype == np.int64
  assert (distri == [0,6,15,17]).all()
  distri = utils.gather_and_shift(value, comm, np.int32)
  assert distri.dtype == np.int32
  assert (distri == [0,6,15,17]).all()

@pytest_parallel.mark.parallel(3)
def test_exscan_size(comm):
  size = [10, 20, 50][comm.rank]
  expt = [0, 10, 30][comm.rank]
  assert utils.exscan_size(size, comm) == expt

  size = [10, 0, 50][comm.rank]
  expt = [0, 10, 10][comm.rank]
  assert utils.exscan_size(size, comm) == expt

@pytest_parallel.mark.parallel(3)
def test_arrays_max(comm):
  if comm.Get_rank() == 0:
    arrays = [np.array([1,6,2]), np.array([3,4,2])]
  elif comm.Get_rank() == 1:
    arrays = [np.array([1,3,5,7]), np.empty(0, np.int64)]
  elif comm.Get_rank() == 2:
    arrays = []
  assert utils.arrays_max(arrays, comm) == 7

@pytest_parallel.mark.parallel(2)
def test_any_true(comm):
  f = lambda e: e < 10
  if comm.Get_rank() == 0:
    L1 = [15,18,1,19]
    L2 = []
  if comm.Get_rank() == 1:
    L1 = [1,4]
    L2 = [41,44]
  assert utils.any_true(L1, f, comm) == True
  assert utils.any_true(L2, f, comm) == False

@pytest_parallel.mark.parallel(2)
def test_exists_anywhere(comm):
  trees = []
  if comm.Get_rank() > 0:
    zone = PT.new_Zone()
    zbc  = PT.new_ZoneBC(parent=zone)
    bc   = PT.new_BC('BCA', parent=zbc)
    trees.append(zone)
  assert utils.exists_anywhere(trees, 'ZoneBC/BCA', comm) == True
  assert utils.exists_anywhere(trees, 'ZoneBC/BCB', comm) == False

@pytest_parallel.mark.parallel(3)
def test_exists_everywhere(comm):
  trees = []
  if comm.Get_rank() > 0:
    zone = PT.new_Zone()
    zbc  = PT.new_ZoneBC(parent=zone)
    bc   = PT.new_BC('BCA', parent=zbc)
    if comm.Get_rank() > 1:
      bc   = PT.new_BC('BCB', parent=zbc)
    trees.append(zone)
  assert utils.exists_everywhere(trees, 'ZoneBC/BCA', comm) == True
  assert utils.exists_everywhere(trees, 'ZoneBC/BCB', comm) == False

@pytest_parallel.mark.parallel(3)
def test_sets_intersection(comm):
  if comm.rank == 0:
    sets1 = [{'apple', 'banana'}]
    sets2 = [{'apple', 'banana'}]
  elif comm.rank == 1:
    sets1 = [] # No sets
    sets2 = [{'banana', 'watermelon'}, set()] # Empty set
  else:
    sets1 = [{'banana', 'pear', 'apple'}, {'peach', 'banana'}]
    sets2 = [{'banana', 'pear', 'apple'}, {'peach', 'banana'}]
  
  assert utils.sets_intersection(sets1, comm) == {'banana'}
  assert utils.sets_intersection(sets2, comm) == set()
  assert utils.sets_intersection([], comm) == None

@pytest_parallel.mark.parallel(3)
def test_sets_union(comm):
  if comm.rank == 0:
    sets1 = [{'apple', 'banana'}]
    sets2 = [{'apple', 'banana'}]
  elif comm.rank == 1:
    sets1 = [] # No sets
    sets2 = [{'banana', 'watermelon'}, set()] # Empty set
  else:
    sets1 = [{'banana', 'pear', 'apple'}, {'peach', 'banana'}]
    sets2 = [{'banana', 'pear', 'apple'}, {'peach', 'banana'}]
  
  assert utils.sets_union(sets1, comm) == {'apple', 'pear', 'banana', 'peach'}
  assert utils.sets_union(sets2, comm) == {'apple', 'pear', 'banana', 'peach', 'watermelon'}
  assert utils.sets_union([], comm) == None
  