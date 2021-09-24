import os
import tempfile
import shutil

mesh_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'share', 'meshes')

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

