import pytest
import pytest_parallel

import numpy as np
from pathlib import Path

import maia.pytree as PT
from maia.pytree.yaml import parse_yaml_cgns

import maia.utils.test_utils as TU

from maia.io import _hdf_io_h5py as IOH

def test_load_data():
  names, labels = 'Base/Zone', 'CGNSBase_t/Zone_t'
  assert IOH.load_data(names.split('/'), labels.split('/')) == True
  names, labels = 'Base/Zone/GCo/CX', 'CGNSBase_t/Zone_t/GridCoordinates_t/DataArray_t'
  assert IOH.load_data(names.split('/'), labels.split('/')) == False
  names, labels = 'Base/Zone/GCo/CX', 'CGNSBase_t/Zone_t/UserDefinedData_t/DataArray_t'
  assert IOH.load_data(names.split('/'), labels.split('/')) == True
  names, labels = 'GC/GCP/Perio/RotationAngle', 'GridConnectivity_t/GridConnectivityProperty_t/Periodic_t/DataArray_t'
  assert IOH.load_data(names.split('/'), labels.split('/')) == True
  names, labels = 'ZBC/BC/PointList', 'ZoneBC_t/BC_t/IndexArray_t'
  assert IOH.load_data(names.split('/'), labels.split('/')) == False
  names, labels = 'Base/Zone/:elsA#Hybrid/IndexNGONCrossTable', 'CGNSBase_t/Zone_t/UserDefinedData_t/DataArray_t'
  assert IOH.load_data(names.split('/'), labels.split('/')) == False
  names, labels = 'ZBC/BC/PointRange', 'ZoneBC_t/BC_t/IndexRange_t'
  assert IOH.load_data(names.split('/'), labels.split('/')) == True
  names, labels = 'FSSeq/GM/Coeff', 'FlowEquationSet_t/GasModel_t/DataArray_t'
  assert IOH.load_data(names.split('/'), labels.split('/')) == True

@pytest_parallel.mark.parallel(3)
def test_load_size_tree(comm):
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
  sizetree = IOH.load_size_tree(filename, comm)
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

@pytest_parallel.mark.parallel(2)
def test_write_partial(comm, tmp_path):
  if comm.rank == 0:
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
  with TU.collective_tmp_dir(comm) as tmpdir:
    filename = str(Path(tmpdir) / 'out.hdf')
    IOH.write_partial(filename, tree, hdf_filter, comm)
    comm.barrier()

    if comm.rank == 0:
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
