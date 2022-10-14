import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import numpy as np
from pathlib import Path

import maia.pytree as PT
from maia.pytree.yaml import parse_yaml_cgns

import maia.utils.test_utils as TU

from maia.io import _hdf_io_h5py as IOH

def test_load_data():
  assert IOH.load_data(('Base', 'CGNSBase_t'), ('Zone', 'Zone_t')) == True
  assert IOH.load_data(('GCo', 'GridCoordinate_t'), ('CX', 'DataArray_t')) == False
  assert IOH.load_data(('GCo', 'UserDefinedData_t'), ('CX', 'DataArray_t')) == True
  assert IOH.load_data(('Periodic', 'Periodic_t'), ('RotationAngle', 'DataArray_t')) == True
  assert IOH.load_data(('BC', 'BC_t'), ('PointList', 'IndexArray_t')) == False
  assert IOH.load_data(('BC', 'BC_t'), ('PointRange', 'IndexRange_t')) == True

@mark_mpi_test(3)
def test_load_collective_size_tree(sub_comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  yt = """
  Base CGNSBase_t [2,2]:
    ZoneU Zone_t [[6, 0, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t:
        CoordinateX#Size DataArray_t I8 [6]:
        CoordinateY DataArray_t:
        CoordinateY#Size DataArray_t I8 [6]:
    ZoneS Zone_t [[2, 1, 0], [2, 1, 0]]:
      ZoneType ZoneType_t 'Structured':
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t:
        CoordinateX#Size DataArray_t I8 [2,2]:
        CoordinateY DataArray_t:
        CoordinateY#Size DataArray_t I8 [2,2]:
  """
  sizetree = IOH.load_collective_size_tree(filename, sub_comm)
  assert PT.is_same_tree(sizetree, parse_yaml_cgns.to_cgns_tree(yt))

def test_load_partial():
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  yt = """
  Base CGNSBase_t [2,2]:
    ZoneU Zone_t [[6, 0, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t:
        CoordinateY DataArray_t:
  """
  # Nb : Low level load/write are tested in other file
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  hdf_filter = {'/Base/ZoneU/GridCoordinates/CoordinateX' : [[0], [1], [2], [1], [4], [1], [2], [1], [6], [1]]}
  IOH.load_partial(filename, tree, hdf_filter)
  assert np.allclose(PT.get_node_from_name(tree, 'CoordinateX')[1], [5., 6.])

@mark_mpi_test(2)
def test_write_partial(sub_comm, tmp_path):
  if sub_comm.rank == 0:
    yt = """
    Base CGNSBase_t [2,2]:
      ZoneU Zone_t [[6, 0, 0]]:
        ZoneType ZoneType_t "Unstructured":
        GridCoordinates GridCoordinates_t:
          CoordinateX DataArray_t R8 [1,2,3,4]:
          CoordinateY DataArray_t R8 [-1,-2,-3,-4]:
    """
    hdf_filter = {'/Base/ZoneU/GridCoordinates/CoordinateX' : [[0], [1], [4], [1], [0], [1], [4], [1], [6], [1]],
                  '/Base/ZoneU/GridCoordinates/CoordinateY' : [[0], [1], [4], [1], [0], [1], [4], [1], [6], [1]]}
  else:
    yt = """
    Base CGNSBase_t [2,2]:
      ZoneU Zone_t [[6, 0, 0]]:
        ZoneType ZoneType_t "Unstructured":
        GridCoordinates GridCoordinates_t:
          CoordinateX DataArray_t R8 [5,6]:
          CoordinateY DataArray_t R8 [-5,-6]:
    """
    hdf_filter = {'/Base/ZoneU/GridCoordinates/CoordinateX' : [[0], [1], [2], [1], [4], [1], [2], [1], [6], [1]],
                  '/Base/ZoneU/GridCoordinates/CoordinateY' : [[0], [1], [2], [1], [4], [1], [2], [1], [6], [1]]}
  # Nb : Low level load/write are tested in other file
  tree = parse_yaml_cgns.to_cgns_tree(yt)
  with TU.collective_tmp_dir(sub_comm) as tmpdir:
    filename = str(Path(tmpdir) / 'out.hdf')
    IOH.write_partial(filename, tree, hdf_filter, sub_comm)
    sub_comm.barrier()

    if sub_comm.rank == 0:
      tree = IOH.read_full(filename)
      ytfull = """
      Base CGNSBase_t [2,2]:
        ZoneU Zone_t [[6, 0, 0]]:
          ZoneType ZoneType_t "Unstructured":
          GridCoordinates GridCoordinates_t:
            CoordinateX DataArray_t R8 [1,2,3,4,5,6]:
            CoordinateY DataArray_t R8 [-1,-2,-3,-4,-5,-6]:
      """
      assert PT.is_same_tree(tree, parse_yaml_cgns.to_cgns_tree(ytfull))
