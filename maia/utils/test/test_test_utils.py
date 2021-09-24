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
