import os
import pytest
import pytest_parallel
import maia.io 
import maia.pytree as PT
import maia.utils.test_utils as TU
import warnings 
import maia
from maia.io import cgns_io_tree as IOT
from maia.io.cgns_io_tree import load_tree_from_filter
from maia.io.cgns_io_tree import  create_tree_hdf_filter 
from maia.io.cgns_io_tree import add_distribution_info


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

  t = maia.io.read_tree(out_file)
  assert (PT.get_value(PT.get_node_from_name(t,"CoordinateX")) == [0.,1.,2.,3.]).all()
  maia.io.dist_tree_to_file(dist_tree, out_file, comm, legacy=True)
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
      file_links = maia.io.read_links(out_file, legacy=True)
      assert file_links == [links[0]]
    else:
      t = maia.io.read_tree(out_file)
      assert (PT.get_value(PT.get_node_from_name(t,"CoordinateX")) == [0.,1.,2.,3.]).all()
      t = maia.io.read_tree(out_file, legacy=True)
      
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
    maia.io.file_to_dist_tree(tmp_file, comm, legacy=True)
  TU.rm_collective_dir(tmp_dir, comm)

@pytest_parallel.mark.parallel(3)
def test_write_trees(comm):
  rank=comm.Get_rank()
  tree = maia.factory.generate_dist_block(4, "TRI_3", comm)
  tmp_dir = TU.create_collective_tmp_dir(comm)
  tmp_file = os.path.join(tmp_dir, f'test_rank_{rank}.cgns')
  for i in tmp_file:
    maia.io.write_trees(tree, tmp_file, comm)
  legacy_file = os.path.join(tmp_dir, f'legacy_test_rank_{rank}.cgns')
  with pytest.warns(DeprecationWarning, match=".*"):  
    maia.io.write_trees(tree, legacy_file, comm, legacy=True)
  assert legacy_file is not None
  

@pytest_parallel.mark.parallel(2)
def test_fill_size_tree(comm): 
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  dist_tree = IOT.load_size_tree(filename, comm)
  node = PT.get_nodes_from_name(dist_tree, 'ZoneU')
  assert node is not None
  IOT.fill_size_tree(dist_tree, filename, comm, False)
  assert len(node) == 1
  with warnings.catch_warnings(record=True) as w:
    IOT.fill_size_tree(dist_tree, filename, comm, True)
    

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize('legacy', [False, True])  
def test_load_size_tree(legacy, comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  expected_size_tree_yaml  = """
CGNSTree CGNSTree_t:
  Base CGNSBase_t [2, 2]:
    ZoneU Zone_t [[6, 0, 0]]:
      ZoneType ZoneType_t 'Unstructured':
      GridCoordinates GridCoordinates_t:
        CoordinateX#Size DataArray_t I8 [6]:
        CoordinateX DataArray_t:
        CoordinateY#Size DataArray_t I8 [6]:
        CoordinateY DataArray_t:
    ZoneS Zone_t [[2, 1, 0], [2, 1, 0]]:
      ZoneType ZoneType_t 'Structured':
      GridCoordinates GridCoordinates_t:
        CoordinateX#Size DataArray_t I8 [2, 2]:
        CoordinateX DataArray_t:
        CoordinateY#Size DataArray_t I8[2, 2]:
        CoordinateY DataArray_t:
  CGNSLibraryVersion CGNSLibraryVersion_t 4.2:
  """

  expected_size_tree = PT.yaml.to_cgns_tree(expected_size_tree_yaml)
  
  size_tree = IOT.load_size_tree(filename, comm, legacy)
  assert PT.is_same_tree(size_tree, expected_size_tree)


@pytest_parallel.mark.parallel(2)
def test_load_partial(comm):
    
  if comm.rank == 0:
    hdf_filter = {
      'Base/ZoneU/GridCoordinates/CoordinateX': [[0], [1], [3], [1], [0], [1], [3], [1], [6], [0]], 
      'Base/ZoneU/GridCoordinates/CoordinateY': [[0], [1], [3], [1], [0], [1], [3], [1], [6], [0]], 
      'Base/ZoneS/GridCoordinates/CoordinateX': [[0], [1], [2], [1], [[0, 0], [1, 1], [2, 1], [1, 1]], [2, 2], [0]], 
      'Base/ZoneS/GridCoordinates/CoordinateY': [[0], [1], [2], [1], [[0, 0], [1, 1], [2, 1], [1, 1]], [2, 2], [0]]
    }
  elif comm.rank == 1:
    hdf_filter = {
      'Base/ZoneU/GridCoordinates/CoordinateX': [[0], [1], [3], [1], [3], [1], [3], [1], [6], [0]], 
      'Base/ZoneU/GridCoordinates/CoordinateY': [[0], [1], [3], [1], [3], [1], [3], [1], [6], [0]], 
      'Base/ZoneS/GridCoordinates/CoordinateX': [[0], [1], [2], [1], [[0, 1], [1, 1], [2, 1], [1, 1]], [2, 2], [0]], 
      'Base/ZoneS/GridCoordinates/CoordinateY': [[0], [1], [2], [1], [[0, 1], [1, 1], [2, 1], [1, 1]], [2, 2], [0]]
    }


  dist_tree = PT.yaml.to_cgns_tree(f"""
  Base CGNSBase_t [2,2]:
    ZoneU Zone_t [[6,0,0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t:
        CoordinateY DataArray_t:
    ZoneS Zone_t [[2,1,0], [2,1,0]]:
      ZoneType ZoneType_t "Structured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t:
        CoordinateY DataArray_t:
  """)
 
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  IOT.load_partial(filename, dist_tree, hdf_filter, comm)

  if comm.rank == 0:
    assert (PT.get_node_from_path(dist_tree, 'Base/ZoneU/GridCoordinates/CoordinateX')[1] == [1., 2, 3]).all()
    assert (PT.get_node_from_path(dist_tree, 'Base/ZoneU/GridCoordinates/CoordinateY')[1] == [-1., -2, -3]).all()
    assert (PT.get_node_from_path(dist_tree, 'Base/ZoneS/GridCoordinates/CoordinateX')[1] == [1., 3]).all()
    assert (PT.get_node_from_path(dist_tree, 'Base/ZoneS/GridCoordinates/CoordinateY')[1] == [-1., 0]).all()
  elif comm.rank == 1:
    assert (PT.get_node_from_path(dist_tree, 'Base/ZoneU/GridCoordinates/CoordinateX')[1] == [4., 5, 6]).all()
    assert (PT.get_node_from_path(dist_tree, 'Base/ZoneU/GridCoordinates/CoordinateY')[1] == [-4., -5, -6]).all()
    assert (PT.get_node_from_path(dist_tree, 'Base/ZoneS/GridCoordinates/CoordinateX')[1] == [2., 4]).all()
    assert (PT.get_node_from_path(dist_tree, 'Base/ZoneS/GridCoordinates/CoordinateY')[1] == [-1., 0]).all()

def test_load_from_filter(comm):  
    dist_tree = maia.factory.generate_dist_block([4,2,2], 'S', comm)
    maia.algo.dist.convert_s_to_ngon(dist_tree, comm)
    tmp_dir = TU.create_collective_tmp_dir(comm)
    filename = os.path.join(tmp_dir, 'tree.cgns')
    maia.io.dist_tree_to_file(dist_tree, filename, comm)
    hdf_filter = create_tree_hdf_filter(dist_tree)
    hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}
    IOT.load_tree_from_filter(filename, dist_tree, comm, hdf_filter)

 

    

  




      