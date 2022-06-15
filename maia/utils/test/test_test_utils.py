import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import os

import maia.utils.test_utils as TU

@mark_mpi_test(3)
def test_create_collective_tmp_dir(sub_comm):
  dirpath = TU.create_collective_tmp_dir(sub_comm)
  assert sub_comm.allgather(dirpath) == sub_comm.Get_size() * [dirpath]
  assert os.path.exists(dirpath)

  TU.rm_collective_dir(dirpath, sub_comm)
  assert not os.path.exists(dirpath)

@mark_mpi_test(2)
def test_create_pytest_output_dir(sub_comm):
  dirpath = TU.create_pytest_output_dir(sub_comm)
  assert os.path.exists(dirpath)
  TU.rm_collective_dir(dirpath, sub_comm)
  #Cleanup : remove root dir if empty
  if not os.listdir(TU.pytest_output_prefix):
    TU.rm_collective_dir(TU.pytest_output_prefix, sub_comm)

@mark_mpi_test(2)
def test_collective_tmp_dir_cm(sub_comm):
  with TU.collective_tmp_dir(sub_comm) as tmp_dir:
    assert os.path.exists(tmp_dir)
  assert not os.path.exists(tmp_dir)
