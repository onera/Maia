import os
import maia.io.fix_tree
import pytest
import pytest_parallel
import maia.io 
import maia.pytree as PT
import maia.utils.test_utils as TU
import maia
from pathlib import Path
from maia.io import _hdf_io_h5py as IOH
from maia.io.cgns_io_tree import load_tree_from_filter
from maia.io.cgns_io_tree import  create_tree_hdf_filter 
from maia.io.cgns_io_tree import add_distribution_info
import warnings 

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
      file_links = maia.io.read_links(out_file)
      assert file_links == [links[0]]
    else:
      t = maia.io.read_tree(out_file)
      assert (PT.get_value(PT.get_node_from_name(t,"CoordinateX")) == [0.,1.,2.,3.]).all()
      t = maia.io.read_tree(out_file, legacy=True)
      file_links = maia.io.read_links(out_file, legacy=True)
      
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

@pytest_parallel.mark.parallel(3)
def test_write_trees(comm):
  rank=comm.Get_rank()
  trees = {
        0: maia.factory.generate_dist_block(4, "TRI_3", comm),
        1: maia.factory.generate_dist_block(2, "TRI_3", comm),
        2: maia.factory.generate_dist_block(3, "TRI_3", comm)
    }
  tmp_dir = TU.create_collective_tmp_dir(comm)
  tmp_file = os.path.join(tmp_dir, f'test_rank_{comm.Get_rank()}.cgns')
  if rank in trees:
        maia.io.write_trees(trees[rank], tmp_file, comm)
  legacy_file = os.path.join(tmp_dir, f'legacy_test_rank_{rank}.cgns')
  with pytest.warns(DeprecationWarning, match=".*"):  
    maia.io.write_trees(trees[rank], legacy_file, comm, legacy=True)
  expected_legacy_filename = f"{legacy_file.rstrip('.cgns')}_{rank}.cgns"
  assert os.path.exists(expected_legacy_filename), f"The file legacy {expected_legacy_filename} is not created"
  


@pytest_parallel.mark.parallel(2)
def test_fill_size_tree(comm): 
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  dist_tree = maia.io.cgns_io_tree.load_size_tree(filename, comm)
  assert dist_tree is not None
  add_distribution_info(dist_tree, comm)
  hdf_filter = create_tree_hdf_filter(dist_tree) 
  assert hdf_filter is not None
  #PT.print_tree(dist_tree)
  size_nodes = PT.get_nodes_from_name(dist_tree, '*#Size')
  assert size_nodes is not None
  #PT.print_node(size_nodes, any, any)
  hdf_filter=create_tree_hdf_filter(dist_tree)
  hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')} 
  maia.io.cgns_io_tree.fill_size_tree(dist_tree, filename, comm, False)
  assert len(size_nodes) == 4, "4 nodes added again" 
  with warnings.catch_warnings(record=True) as w:
    maia.io.cgns_io_tree.fill_size_tree(dist_tree, filename, comm, True)
    
@pytest_parallel.mark.parallel(2)  
def test_load_size_tree(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  dist_tree = maia.io.cgns_io_tree.load_size_tree(filename, comm, False)
  with warnings.catch_warnings(record=True) as w:
    maia.io.cgns_io_tree._hdf_io.load_size_tree(filename, comm)
      
@pytest_parallel.mark.parallel(2)  
def test_load_partial(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  dist_tree = maia.io.cgns_io_tree.load_size_tree(filename, comm, False)
  #PT.print_tree(dist_tree)
  add_distribution_info(dist_tree, comm)
  hdf_filter = create_tree_hdf_filter(dist_tree) 
  hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}
  #print(hdf_filter)
  assert hdf_filter is not None
  #maia.io.cgns_io_tree.load_partial(filename, dist_tree, hdf_filter,comm)
 
@pytest_parallel.mark.parallel(2)  
def test_load_size_tree(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
  first_dist_tree = maia.io.cgns_io_tree.load_size_tree(filename, comm, False)
  second_dist_tree = maia.io.cgns_io_tree.load_size_tree(filename, comm, True)

# @pytest_parallel.mark.parallel(3)
# def test_write_trees(comm):
#   tree = maia.factory.generate_dist_block(4, "TRI_3", comm)
#   tmp_dir = TU.create_collective_tmp_dir(comm)
#   tmp_file = os.path.join(tmp_dir, f'write_tree.cgns')
#   links=[]
#   for zone_path in maia.pytree.predicates_to_paths(tree, 'CGNSBase_t/Zone_t'):
#     print(zone_path)
#     _links = [link for link in links if link[:].startswith(zone_path)]
#     PT.print_tree(tree)
  # print('#####################LINKS##############################')
  # print(_links)
  # maia.io.write_tree(tree, tmp_file, _links)
  # maia.io.write_tree(tree, tmp_file, links=[], legacy=True)
        
@pytest.mark.parallel(2)
def test_load_from_filter(comm):
    filename = str(TU.sample_mesh_dir / 'only_coords.hdf')
    dist_tree = maia.io.cgns_io_tree.load_size_tree(filename, comm, False)
    add_distribution_info(dist_tree, comm)
    hdf_filter = create_tree_hdf_filter(dist_tree)
    hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}
    hdf_filter_with_func = {key: value for (key, value) in hdf_filter.items() if not isinstance(value, (list, tuple))} 
    if hdf_filter_with_func:
      maia.io.cgns_io_tree.load_tree_from_filter(filename, dist_tree, comm, hdf_filter)
      assert dist_tree is not None
    print("###################")
    print(hdf_filter)
