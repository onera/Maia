import pytest
import pytest_parallel

import os

import maia.utils.test_utils as TU

@pytest_parallel.mark.parallel(3)
def test_create_collective_tmp_dir(comm):
  dirpath = TU.create_collective_tmp_dir(comm)
  assert comm.allgather(dirpath) == comm.Get_size() * [dirpath]
  assert os.path.exists(dirpath)

  TU.rm_collective_dir(dirpath, comm)
  assert not os.path.exists(dirpath)

@pytest_parallel.mark.parallel(2)
def test_create_pytest_output_dir(comm):
  dirpath = TU.create_pytest_output_dir(comm)
  assert os.path.exists(dirpath)
  TU.rm_collective_dir(dirpath, comm)
  #Cleanup : remove root dir if empty
  if not os.listdir(TU.pytest_output_prefix):
    TU.rm_collective_dir(TU.pytest_output_prefix, comm)

@pytest_parallel.mark.parallel(2)
def test_collective_tmp_dir_cm(comm):
  with TU.collective_tmp_dir(comm) as tmp_dir:
    assert os.path.exists(tmp_dir)
  assert not os.path.exists(tmp_dir)
