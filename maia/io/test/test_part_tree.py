import pytest
import pytest_parallel
import mpi4py.MPI as MPI
import numpy      as np
from pathlib import Path

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.io import part_tree as PIO

dtype = 'I8' if maia.npy_pdm_gnum_dtype == np.int64 else 'I4'

class LogCapture:
  def __init__(self):
    self.msg = ''
  def log(self, msg):
    self.msg = self.msg + msg


@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize('single_file', [False, True])
@pytest.mark.parametrize('user_links', [True, False])
def test_write_part_tree(mpi_tmpdir, user_links, single_file, comm):
  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)

  PT.new_UserDefinedData('TopLevelCustomNode', parent=tree)

  links = []
  if user_links:
    links = [] if comm.Get_rank() == 1 else [('.', 'this/hdf/file.hdf', 'this/other_node', f'Base/zone.P{comm.rank}.N0/GridCoordinates/CoordinateZ')]

  expected_n_files = 1 + comm.Get_size() * int(not single_file)

  filename = Path(mpi_tmpdir) / 'out.hdf'
  PIO.save_part_tree(tree, str(filename), comm, single_file, links)
  comm.barrier()
  assert filename.exists()
  assert len(list(Path(mpi_tmpdir).glob('*'))) == expected_n_files

  if comm.Get_rank() == 0 and not user_links:
    tree = maia.io.read_tree(str(filename))
    for rank in range(comm.Get_size()):
      assert PT.get_node_from_path(tree, f'Base/zone.P{rank}.N0') is not None
    assert PT.get_value(PT.get_node_from_path(tree, 'Base/zone.P1.N0/ZoneType')) == 'Unstructured'
    assert PT.get_label(PT.get_node_from_path(tree, 'TopLevelCustomNode')) == 'UserDefinedData_t'

    # Parallelism dependant ...
    # ref = PT.yaml.to_node(f"""
    # Xmax BC_t "Null":
      # GridLocation GridLocation_t "FaceCenter":
      # PointList IndexArray_t [[19,20,21,22,23,24]]:
      # :CGNS#GlobalNumbering UserDefinedData_t:
        # Index DataArray_t {dtype} [2,3,4,5,6,7]:
    # """)
    # assert PT.is_same_tree(PT.get_node_from_path(tree, 'Base/zone.P1.N0/ZoneBC/Xmax'), ref)
  elif comm.Get_rank() == 0 and user_links:
    if single_file:
      links = maia.io.read_links(str(filename))
      assert links[0] == ['.', 'this/hdf/file.hdf', 'this/other_node', 'Base/zone.P0.N0/GridCoordinates/CoordinateZ']
      assert links[1] == ['.', 'this/hdf/file.hdf', 'this/other_node', 'Base/zone.P2.N0/GridCoordinates/CoordinateZ']
      assert links[2] == ['.', 'this/hdf/file.hdf', 'this/other_node', 'Base/zone.P3.N0/GridCoordinates/CoordinateZ']
    else:
      for i in range(4):
        links = maia.io.read_links(str(filename)[:-4] + f'_sub_{i}.hdf')
        if i == 1:
          assert links == []
        else:
          assert links == [['.', 'this/hdf/file.hdf', 'this/other_node', f'Base/zone.P{i}.N0/GridCoordinates/CoordinateZ']]

@pytest_parallel.mark.parallel(4)
@pytest.mark.parametrize('single_file', [False, True])
def test_read_part_tree(mpi_tmpdir, single_file, comm):

  # Prepare test (produce part_tree_file)
  dtree = maia.factory.generate_dist_block(4, 'Poly', comm)
  tree  = maia.factory.partition_dist_tree(dtree, comm)
  expected = np.copy(PT.get_node_from_name(tree, 'CoordinateX')[1]) #Backup array for later check
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
