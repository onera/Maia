from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np
from mpi4py import MPI
import Converter.Internal as I

from maia.utils.parallel import utils

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
    zone = I.newZone()
    zbc  = I.newZoneBC(parent=zone)
    bc   = I.newBC('BCA', parent=zbc)
    trees.append(zone)
  assert utils.exists_anywhere(trees, 'ZoneBC/BCA', sub_comm) == True
  assert utils.exists_anywhere(trees, 'ZoneBC/BCB', sub_comm) == False

@mark_mpi_test(3)
def test_exists_everywhere(sub_comm):
  trees = []
  if sub_comm.Get_rank() > 0:
    zone = I.newZone()
    zbc  = I.newZoneBC(parent=zone)
    bc   = I.newBC('BCA', parent=zbc)
    if sub_comm.Get_rank() > 1:
      bc   = I.newBC('BCB', parent=zbc)
    trees.append(zone)
  assert utils.exists_everywhere(trees, 'ZoneBC/BCA', sub_comm) == True
  assert utils.exists_everywhere(trees, 'ZoneBC/BCB', sub_comm) == False
