import os
from pathlib import Path
import tempfile
import shutil
import numpy as np

import maia

from packaging.version import Version
from Pypdm.Pypdm import __version__ as _PDM_VERSION
PDM_VERSION = Version(_PDM_VERSION)



mesh_dir        = Path(maia.__file__).parent.parent/'share/meshes'
sample_mesh_dir = Path(maia.__file__).parent.parent/'share/sample_meshes'
pytest_output_prefix = 'pytest_out'

def create_collective_tmp_dir(comm):
  """
  Create a unique temporary directory and return its path
  """
  if comm.Get_rank()==0:
    tmp_test_dir = tempfile.mkdtemp(dir=os.getcwd())
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

def portable_partitioning(dist_tree, wanted_cell_l, comm, **kwargs):
  """ Create a custom partioning (chosing cells for each part) to ensure portability 
  (require PDM >= 2.7)"""
  import maia.pytree      as PT
  import maia.pytree.maia as MT
  from maia.transfer import protocols as MEP
  from maia.utils import par_utils
  
  # Retrieve target part on distributed cells
  zone_paths = PT.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t')
  assert len(zone_paths) == 1
  cell_distri = MT.distribution_value(PT.find_node_from_path(dist_tree, zone_paths[0]), 'Cell')
  rank_offset = par_utils.gather_and_shift(len(wanted_cell_l), comm)[comm.rank]
  target_part_p = [np.full(w.size, rank_offset+i, np.int32) for i,w in enumerate(wanted_cell_l)]

  target_part = [MEP.part_to_block(target_part_p, cell_distri, wanted_cell_l, comm, gnum_offset=1)]
  zone_to_parts = {zone_paths[0] : [1.]*len(wanted_cell_l)} # Weights does not matter but len does

  return maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts,
                                          target_part=target_part, **kwargs)