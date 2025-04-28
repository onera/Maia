import pytest_parallel

import maia.utils.test_utils as TU

from maia.io import utils


@pytest_parallel.mark.parallel([1,2])
def test_create_parent_folder(comm):
  tmp_dir = TU.create_collective_tmp_dir(comm)
  filename = tmp_dir / 'TESTDIR' / 'mycgns.cgns'

  # Basic test
  assert not filename.parent.exists()
  utils.create_parent_folder(filename, comm)
  assert filename.parent.exists()
  assert not filename.exists() # Only parent dir is created by function, not the file itself

  # Test with exising dir
  utils.create_parent_folder(filename, comm)
  assert filename.parent.exists()

  # TODO This one does not work
  filename = tmp_dir / 'OTHERTESTDIR' / 'SUBDIR' / 'mycgns.cgns'
  utils.create_parent_folder(filename, comm)
  assert filename.parent.exists()

  # Check w/o dir (nothing should happen)
  filename = 'mycgns.cgns'
  utils.create_parent_folder(filename, comm)

  TU.rm_collective_dir(tmp_dir, comm)