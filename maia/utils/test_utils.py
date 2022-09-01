import os
from pathlib import Path
import tempfile
import shutil

import maia

mesh_dir        = Path(maia.__file__).parent.parent/'share/meshes'
sample_mesh_dir = Path(maia.__file__).parent.parent/'share/sample_meshes'
pytest_output_prefix = 'pytest_out'

def create_collective_tmp_dir(comm):
  """
  Create a unique temporary directory and return its path
  """
  if comm.Get_rank()==0:
    tmp_test_dir = tempfile.mkdtemp()
  else:
    tmp_test_dir = ""
  return Path(comm.bcast(tmp_test_dir,root=0))

def rm_collective_dir(path, comm):
  """
  Remove a directory from its path
  """
  comm.barrier()
  if comm.Get_rank() == 0:
    shutil.rmtree(path)
  comm.barrier()

class collective_tmp_dir:
  """
  Context manager creating a tmp dir in parallel and removing it at the
  exit
  """
  def __init__(self, comm):
    self.comm = comm
  def __enter__(self):
    self.path = create_collective_tmp_dir(self.comm)
    return self.path
  def __exit__(self, type, value, traceback):
    rm_collective_dir(self.path, self.comm)

def create_pytest_output_dir(comm):
  """
  Create (in parallel) a directory named from the name of the current
  test runned by pytest and prefixed by module variable pytest_output_prefix.
  Return the name of this directory
  """
  test_name = os.environ.get('PYTEST_CURRENT_TEST').split('::')[-1].split()[0]
  out_dir = Path(pytest_output_prefix)/test_name
  if comm.Get_rank() == 0:
    if not out_dir.exists():
      out_dir.mkdir(parents=True)
  comm.barrier()
  return out_dir

