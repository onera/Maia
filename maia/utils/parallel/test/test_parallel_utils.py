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

@pytest_parallel.mark.parallel(3)
def test_dn_to_distribution(comm):
  dn = (comm.rank+1)**3 #1, 8, 27
  expt_distri_f = np.array([0, 1, 9, 36])
  distrib = utils.dn_to_distribution(dn, comm)
  assert isinstance(distrib, np.ndarray)
  assert distrib.dtype == npy_pdm_gnum_dtype
  assert (distrib == expt_distri_f[[comm.rank, comm.rank+1, comm.size]]).all()


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
