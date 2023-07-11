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

from maia.io import part_tree as PIO

class LogCapture:
  def __init__(self):
    self.msg = ''
  def log(self, msg):
    self.msg = self.msg + msg


@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize('single_file', [False, True])
def test_write_part_tree(mpi_tmpdir, single_file, comm):
  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)

  expected_n_files = 1 + comm.Get_size() * int(not single_file)

  filename = Path(mpi_tmpdir) / 'out.hdf'
  PIO.save_part_tree(tree, str(filename), comm, single_file)
  comm.barrier()
  assert filename.exists()
  assert len(list(Path(mpi_tmpdir).glob('*'))) == expected_n_files

  if comm.Get_rank() == 0:
    tree = maia.io.read_tree(str(filename))
    ref = parse_yaml_cgns.to_node("""
    Xmax BC_t "Null":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[19,20,21,22,23,24]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t [2,3,4,5,6,7]:
    """)
    assert PT.is_same_tree(PT.get_node_from_path(tree, 'Base/zone.P1.N0/ZoneBC/Xmax'), ref)

@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize('single_file', [False, True])
def test_read_part_tree(mpi_tmpdir, single_file, comm):

  # Prepare test (produce part_tree_file)
  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)
  filename = Path(mpi_tmpdir) / 'out.hdf'
  PIO.save_part_tree(tree, str(filename), comm, single_file)
  comm.barrier()

  # Actual test
  tree = PIO.read_part_tree(str(filename), comm)
  zones = PT.get_all_Zone_t(tree)

  # Check: data should have been loaded
  assert len(zones) == 1
  assert PT.get_name(zones[0]) == f'zone.P{comm.Get_rank()}.N0'
  assert PT.get_node_from_name(zones[0], 'CoordinateX')[1].size > 0
  if comm.Get_rank() == 0:
    expected = np.array([0,1,2,3,0,1,2,3,1,2,0,1,2,3,0,1,2,3,0,1,2,0,1,0,1,0,1,0,1,0,1]) / 3.
    assert (PT.get_node_from_name(zones[0], 'CoordinateX')[1] == expected).all()
  elif comm.Get_rank() == 3:
    expected = np.array([1,2,1,2,0,1,2,0,1,2,3,1,2,3,2,3,0,1,2,0,1,2,3,1,2,3,2,3.]) / 3.
    assert (PT.get_node_from_name(zones[0], 'CoordinateX')[1] == expected).all()


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize('redispatch', [False, True])
def test_read_part_tree_redispatch(mpi_tmpdir, redispatch, comm):

  # To get logs in printer.msg  
  err_printer = LogCapture()
  war_printer = LogCapture()
  from maia.utils.logging import add_printer_to_logger
  add_printer_to_logger('maia-errors', err_printer)
  add_printer_to_logger('maia-warnings', war_printer)

  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)

  filename = Path(mpi_tmpdir) / 'out.hdf'
  PIO.save_part_tree(tree, str(filename), comm)
  comm.barrier()
  if comm.Get_rank() == 0:
    tree = PIO.read_part_tree(str(filename), MPI.COMM_SELF, redispatch=redispatch)
    if redispatch:
      assert len(PT.get_all_Zone_t(tree)) == 2
      assert PT.get_name(PT.get_all_Zone_t(tree)[1]) == 'zone.P0.N1'
      assert 'written for 2 procs' in war_printer.msg
    else:
      assert len(PT.get_all_Zone_t(tree)) == 1
      assert 'written for 2 procs' in err_printer.msg
