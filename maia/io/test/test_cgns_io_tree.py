import os
import warnings 
import pytest
import pytest_parallel
from mpi4py import MPI
import maia
import maia.pytree as PT
import maia.io.cgns_io_tree as IOT
import maia.utils.test_utils as TU


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
    

@pytest_parallel.mark.parallel(2)
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


def create_example_mesh(comm):
  tmp_dir = TU.create_collective_tmp_dir(comm)
  filename = os.path.join(tmp_dir, 'tree.cgns')

  if comm.rank == 0:
    dist_tree = maia.factory.generate_dist_block([4,2,2], 'S', MPI.COMM_SELF)
    maia.algo.dist.convert_s_to_ngon(dist_tree, MPI.COMM_SELF)
    maia.io.dist_tree_to_file(dist_tree, filename, MPI.COMM_SELF)
    
  comm.barrier()
  return filename

@pytest_parallel.mark.parallel(2)
def test_load_from_filter(comm):
  filename=create_example_mesh(comm)
  size_tree = maia.io.cgns_io_tree.load_size_tree(filename, comm)
  IOT.add_distribution_info(size_tree, comm)
  hdf_filter = IOT.create_tree_hdf_filter(size_tree)
  hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')} 
  IOT.load_tree_from_filter(filename, size_tree, comm, hdf_filter)
  ngon_node = PT.Zone.NGonNode(PT.get_all_Zone_t(size_tree)[0])
  if comm.rank==0:
    ngon_rank0=("""
    NGonElements Elements_t I4 [22, 0]:
      ElementRange IndexRange_t I4 [1, 16]:
      ElementStartOffset#Size DataArray_t I8 [17]:
      ElementStartOffset DataArray_t I4 [0, 4, 8, 12, 16, 20, 24, 28, 32]:
      ElementConnectivity#Size DataArray_t I8 [64]:
      ElementConnectivity DataArray_t:
        I4 : [1, 9, 13, 5, 2, 6, 14, 10, 3, 7, 15, 11, 4, 8, 16, 12, 1, 2, 10, 9, 2, 3, 11, 10, 3, 4, 12, 11, 5, 13, 14, 6]
      ParentElements#Size DataArray_t I8 [16, 2]:
      ParentElements DataArray_t I4 [[17, 0], [17, 18], [18, 19], [19, 0], [17, 0], [18, 0], [19, 0], [17, 0]]:
      :CGNS#Distribution UserDefinedData_t:
        Element DataArray_t I4 [0, 8, 16]:
        ElementConnectivity DataArray_t I4 [0, 32, 64]:
                """)
    ngon_expected = PT.yaml.to_node(ngon_rank0)
  elif comm.rank==1:
    ngon_rank1=("""
    NGonElements Elements_t I4 [22, 0]:
      ElementRange IndexRange_t I4 [1, 16]:
      ElementStartOffset#Size DataArray_t I8 [17]:
      ElementStartOffset DataArray_t I4 [32, 36, 40, 44, 48, 52, 56, 60, 64]:
      ElementConnectivity#Size DataArray_t I8 [64]:
      ElementConnectivity DataArray_t:
        I4 : [6, 14, 15, 7, 7, 15, 16, 8, 1, 5, 6, 2, 2, 6, 7, 3, 3, 7, 8, 4, 9, 10, 14, 13, 10, 11, 15, 14, 11, 12, 16, 15]
      ParentElements#Size DataArray_t I8 [16, 2]:
      ParentElements DataArray_t I4 [[18, 0], [19, 0], [17, 0], [18, 0], [19, 0], [17, 0], [18, 0], [19, 0]]:
      :CGNS#Distribution UserDefinedData_t:
        Element DataArray_t I4 [8, 16, 16]:
        ElementConnectivity DataArray_t I4 [32, 64, 64]:
      """)
    ngon_expected = PT.yaml.to_node(ngon_rank1)
  assert PT.is_same_node(ngon_node, ngon_expected)
    
