import pytest_parallel
import mpi4py.MPI as MPI

from maia.io.utils import create_parent_folder

from pathlib import Path

class LogCapture:
  def __init__(self):
    self.msg = ''
  def log(self, msg):
    self.msg = self.msg + msg


@pytest_parallel.mark.parallel([1,2])
def test_create_parent_folder(comm):

  filename = Path('toto') /'mycgns.cgns'
  create_parent_folder(comm, filename)

  assert filename.parent.exists()