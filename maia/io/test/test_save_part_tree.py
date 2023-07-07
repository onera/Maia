import pytest
import pytest_parallel
import mpi4py.MPI as MPI
import numpy      as np
from pathlib import Path

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.utils.test_utils as TU
from   maia.pytree.yaml    import parse_yaml_cgns
from   maia.utils.parallel import utils as par_utils

from maia.io import save_part_tree as SPT

class LogCapture:
  def __init__(self):
    self.msg = ''
  def log(self, msg):
    self.msg = self.msg + msg


@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize('single_file', [False, True])
def test_write_part_tree(single_file, comm):
  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)

  expected_n_files = 1 + comm.Get_size() * int(not single_file)

  with TU.collective_tmp_dir(comm) as tmpdir:
    filename = Path(tmpdir) / 'out.hdf'
    SPT.save_part_tree(tree, str(filename), comm, single_file)
    comm.barrier()
    assert filename.exists()
    assert len(list(Path(tmpdir).glob('*'))) == expected_n_files
    
@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize('single_file', [False, True])
def test_read_part_tree(single_file, comm):

  # Prepare test (produce part_tree_file)
  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)

  with TU.collective_tmp_dir(comm) as tmpdir:
    filename = Path(tmpdir) / 'out.hdf'
    SPT.save_part_tree(tree, str(filename), comm, single_file)
    comm.barrier()
    tree = SPT.read_part_tree(str(filename), comm)
    zones = PT.get_all_Zone_t(tree)
    assert len(zones) == 1
    assert PT.get_name(zones[0]) == f'zone.P{comm.Get_rank()}.N0'
    # Data should have been loaded
    assert PT.get_node_from_name(zones[0], 'CoordinateX')[1].size > 0

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('redispatch', [False, True])
def test_read_part_tree_redispatch(redispatch, comm):

  # To get logs in printer.msg  
  err_printer = LogCapture()
  war_printer = LogCapture()
  from maia.utils.logging import add_printer_to_logger
  add_printer_to_logger('maia-errors', err_printer)
  add_printer_to_logger('maia-warnings', war_printer)

  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)
  with TU.collective_tmp_dir(comm) as tmpdir:
    filename = Path(tmpdir) / 'out.hdf'
    SPT.save_part_tree(tree, str(filename), comm)
    comm.barrier()
    if comm.Get_rank() == 0:
      tree = SPT.read_part_tree(str(filename), MPI.COMM_SELF, redispatch=redispatch)
      if redispatch:
        assert len(PT.get_all_Zone_t(tree)) == 2
        assert PT.get_name(PT.get_all_Zone_t(tree)[1]) == 'zone.P0.N1'
        assert 'written for 2 procs' in war_printer.msg
      else:
        assert len(PT.get_all_Zone_t(tree)) == 1
        assert 'written for 2 procs' in err_printer.msg
