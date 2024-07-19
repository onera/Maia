import pytest
import pytest_parallel

import maia.io
import maia.pytree as PT

import maia.utils.test_utils as TU

import os

@pytest_parallel.mark.parallel(1)
def test_dist_tree_to_file_1proc(comm):
  yt = """
Base CGNSBase_t I4 [3, 3]:
  Zone Zone_t I4 [[4, 0, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [0., 1., 2., 3.]:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [0, 4, 4]:
      Cell DataArray_t I4 [0, 0, 0]:
"""

  dist_tree = PT.yaml.to_cgns_tree(yt)

  tmp_dir = TU.create_collective_tmp_dir(comm)
  out_file = os.path.join(tmp_dir, 'yt.cgns')
  maia.io.dist_tree_to_file(dist_tree, out_file, comm)

  t = maia.io.cgns_io_tree.read_tree(out_file)
  assert (PT.get_value(PT.get_node_from_name(t,"CoordinateX")) == [0.,1.,2.,3.]).all()
  TU.rm_collective_dir(tmp_dir, comm)


@pytest.mark.parametrize("user_links", [False, True])
@pytest_parallel.mark.parallel(2)
def test_dist_tree_to_file_2procs(user_links, comm):
  if comm.Get_rank()==0:
    yt = """
Base CGNSBase_t I4 [3, 3]:
  Zone Zone_t I4 [[4, 0, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [0., 1.]:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [0, 2, 4]:
      Cell DataArray_t I4 [0, 0, 0]:
"""
  else:
    yt = """
Base CGNSBase_t I4 [3, 3]:
  Zone Zone_t I4 [[4, 0, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [2., 3.]:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [2, 4, 4]:
      Cell DataArray_t I4 [0, 0, 0]:
"""

  dist_tree = PT.yaml.to_cgns_tree(yt)

  if user_links:
    links = [['.', 'this/hdf/file.hdf', 'this/node', 'Base/Zone/GridCoordinates/CoordinateX'],
             ['.', 'this/hdf/file.hdf', 'this/other_node', 'Base/Zone/ZoneBC_t/BCA']] #This one should be ignored
  else:
    links = []

  tmp_dir = TU.create_collective_tmp_dir(comm)
  out_file = os.path.join(tmp_dir, 'yt.cgns')
  maia.io.dist_tree_to_file(dist_tree, out_file, comm, links)

  if comm.Get_rank()==0:
    if user_links:
      file_links = maia.io.read_links(out_file)
      assert file_links == [links[0]]
    else:
      t = maia.io.cgns_io_tree.read_tree(out_file)
      assert (PT.get_value(PT.get_node_from_name(t,"CoordinateX")) == [0.,1.,2.,3.]).all()
  TU.rm_collective_dir(tmp_dir, comm)

@pytest_parallel.mark.parallel(2)
def test_read_wrong_file(comm):
  tmp_dir = TU.create_collective_tmp_dir(comm)
  tmp_file = os.path.join(tmp_dir, 'test.py')

  # Prepare file for test
  if comm.Get_rank() == 0:
    with open(tmp_file, 'w') as f:
      f.write('import maia\n')
      f.write('print(maia.__version__)')

  with pytest.raises(ValueError):
    maia.io.file_to_dist_tree(tmp_file, comm)
  TU.rm_collective_dir(tmp_dir, comm)
