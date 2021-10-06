import os
import tempfile
import shutil

mesh_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'share', 'meshes')
pytest_output_prefix = 'pytest_out'

def create_collective_tmp_dir(comm):
  """
  Create a unique temporary directory and return its path
  """
  if comm.Get_rank()==0:
    tmp_test_dir = tempfile.mkdtemp()
  else:
    tmp_test_dir = ""
  return comm.bcast(tmp_test_dir,root=0)

def rm_collective_dir(path, comm):
  """
  Remove a directory from its path
  """
  comm.barrier()
  if comm.Get_rank() == 0:
    shutil.rmtree(path)
  comm.barrier()

def create_pytest_output_dir(comm):
  """
  Create (in parallel) a directory named from the name of the current
  test runned by pytest and prefixed by module variable pytest_output_prefix.
  Return the name of this directory
  """
  test_name = os.environ.get('PYTEST_CURRENT_TEST').split('::')[-1].split()[0]
  out_dir = os.path.join(pytest_output_prefix, test_name) 
  if comm.Get_rank() == 0:
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)
  comm.barrier()
  return out_dir

