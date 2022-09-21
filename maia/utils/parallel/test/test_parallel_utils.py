from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np
from mpi4py import MPI

import maia.pytree as PT
from maia import npy_pdm_gnum_dtype

from maia.utils import py_utils
from maia.utils.parallel import utils

@mark_mpi_test(3)
def test_uniform_distribution(sub_comm):
  distrib = utils.uniform_distribution(17, sub_comm)
  assert isinstance(distrib, np.ndarray)
  assert distrib.dtype == npy_pdm_gnum_dtype
  assert (distrib[0:2] == py_utils.uniform_distribution_at(\
      17, sub_comm.Get_rank(), sub_comm.Get_size())).all()
  assert distrib[2] == 17

@mark_mpi_test(3)
def test_dn_to_distribution(sub_comm):
  dn = (sub_comm.rank+1)**3 #1, 8, 27
  expt_distri_f = np.array([0, 1, 9, 36])
  distrib = utils.dn_to_distribution(dn, sub_comm)
  assert isinstance(distrib, np.ndarray)
  assert distrib.dtype == npy_pdm_gnum_dtype
  assert (distrib == expt_distri_f[[sub_comm.rank, sub_comm.rank+1, sub_comm.size]]).all()


@mark_mpi_test(3)
def test_partial_to_full_distribution(sub_comm):
  if sub_comm.Get_rank() == 0:
    partial_distrib_32 = np.array([0, 25, 75], dtype=np.int32)
    partial_distrib_64 = np.array([0, 25, 75], dtype=np.int64)
    partial_distrib_hole = np.array([0, 25, 75])
    partial_distrib_void = np.array([0, 0, 0])
  if sub_comm.Get_rank() == 1:
    partial_distrib_32 = np.array([25, 55, 75], dtype=np.int32)
    partial_distrib_64 = np.array([25, 55, 75], dtype=np.int64)
    partial_distrib_hole = np.array([25, 25, 75])
    partial_distrib_void = np.array([0, 0, 0])
  if sub_comm.Get_rank() == 2:
    partial_distrib_32 = np.array([55, 75, 75], dtype=np.int32)
    partial_distrib_64 = np.array([55, 75, 75], dtype=np.int64)
    partial_distrib_hole = np.array([25, 75, 75])
    partial_distrib_void = np.array([0, 0, 0])

  full_distri_32 = utils.partial_to_full_distribution(partial_distrib_32, sub_comm)
  assert full_distri_32.dtype == np.int32
  assert (full_distri_32 == [0,25,55,75]).all()
  full_distri_64 = utils.partial_to_full_distribution(partial_distrib_64, sub_comm)
  assert full_distri_64.dtype == np.int64
  assert (full_distri_64 == [0,25,55,75]).all()
  full_distri_hole = utils.partial_to_full_distribution(partial_distrib_hole, sub_comm)
  assert (full_distri_hole == [0,25,25,75]).all()
  full_distri_void = utils.partial_to_full_distribution(partial_distrib_void, sub_comm)
  assert (full_distri_void == [0,0,0,0]).all()

@mark_mpi_test(3)
def test_full_to_partial_distribution(sub_comm):
  partial = utils.full_to_partial_distribution(np.array([0, 0, 0, 100], np.int32), sub_comm)
  assert partial.dtype == np.int32
  if sub_comm.rank < 2:
    assert (partial == [0,0,100]).all()
  else:
    assert (partial == [0,100,100]).all()

@mark_mpi_test(3)
def test_gather_and_shift(sub_comm):
  if sub_comm.Get_rank() == 0:
    value = 6
  if sub_comm.Get_rank() == 1:
    value = 9
  if sub_comm.Get_rank() == 2:
    value = 2
  distri = utils.gather_and_shift(value, sub_comm)
  assert distri.dtype == np.int64
  assert (distri == [0,6,15,17]).all()
  distri = utils.gather_and_shift(value, sub_comm, np.int32)
  assert distri.dtype == np.int32
  assert (distri == [0,6,15,17]).all()

@mark_mpi_test(2)
def test_exists_anywhere(sub_comm):
  trees = []
  if sub_comm.Get_rank() > 0:
    zone = PT.new_Zone()
    zbc  = PT.new_ZoneBC(parent=zone)
    bc   = PT.new_BC('BCA', parent=zbc)
    trees.append(zone)
  assert utils.exists_anywhere(trees, 'ZoneBC/BCA', sub_comm) == True
  assert utils.exists_anywhere(trees, 'ZoneBC/BCB', sub_comm) == False

@mark_mpi_test(3)
def test_exists_everywhere(sub_comm):
  trees = []
  if sub_comm.Get_rank() > 0:
    zone = PT.new_Zone()
    zbc  = PT.new_ZoneBC(parent=zone)
    bc   = PT.new_BC('BCA', parent=zbc)
    if sub_comm.Get_rank() > 1:
      bc   = PT.new_BC('BCB', parent=zbc)
    trees.append(zone)
  assert utils.exists_everywhere(trees, 'ZoneBC/BCA', sub_comm) == True
  assert utils.exists_everywhere(trees, 'ZoneBC/BCB', sub_comm) == False
