import pytest
import pytest_parallel

import numpy as np
import shutil
import subprocess
from pathlib import Path
from h5py    import h5f, h5g

import maia.pytree as PT

from maia.pytree.yaml import parse_yaml_cgns
from maia.utils.test_utils import sample_mesh_dir


from maia.io.hdf import _hdf_cgns as HCG

sample_tree = """
  Base CGNSBase_t [2,2]:
    ZoneU Zone_t [[6, 0, 0]]:
      ZoneType ZoneType_t "Unstructured":
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [1., 2., 3., 4., 5., 6.]:
        CoordinateY DataArray_t R8 [-1., -2., -3., -4., -5., -6.]:
    ZoneS Zone_t [[2, 1, 0], [2, 1, 0]]:
      ZoneType ZoneType_t 'Structured':
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [[1., 2.], [3., 4.]]:
        CoordinateY DataArray_t R8 [[-1., -1.], [0., 0.]]:
  """

@pytest.fixture
def ref_hdf_file():
  return str(sample_mesh_dir / 'only_coords.hdf')

@pytest.fixture
def tmp_hdf_file(tmp_path, ref_hdf_file):
  """ Copy the ref hdf in a temporary dir for tests modifying it """
  shutil.copy(ref_hdf_file, tmp_path)
  return str(tmp_path / Path('only_coords.hdf'))

def get_subprocess_stdout(cmd):
  """ Run a command line using subprocess.run and return the stdout as a list of lines """
  output = subprocess.run(cmd.split(), capture_output=True)
  return [s.strip() for s in output.stdout.decode().split('\n')]

def check_h5ls_output(output, key, expected_value):
  # Some hdf version include a line break in h5ls output
  idx = output.index(key) #Will raise if not in list
  if output[idx+2] == "Data:":    # Line break after data
    assert output[idx+3] == expected_value
  else: # Everything on the same line
    assert output[idx+2].split(':')[1].strip() == expected_value

class Test_AttributeRW:
  attr_rw = HCG.AttributeRW()
  assert attr_rw.buff_S33.size == 1 and attr_rw.buff_S33.dtype == 'S33'
  assert attr_rw.buff_flag1.size == 1 and attr_rw.buff_flag1.dtype == np.int32
  
  def test_read(self, ref_hdf_file):
    # Read
    fid = h5f.open(bytes(ref_hdf_file, 'utf-8'), h5f.ACC_RDONLY)
    gid = HCG.open_from_path(fid, 'Base/ZoneU/GridCoordinates')
    assert self.attr_rw.read_str_33(gid, b'name') == 'GridCoordinates'
    assert self.attr_rw.read_bytes_3(gid, b'type') == b'MT'

  def test_write(self, tmp_hdf_file):
    fid = h5f.open(bytes(tmp_hdf_file, 'utf-8'), h5f.ACC_RDWR)
    gid = HCG.open_from_path(fid, 'Base/ZoneU')
    gid = h5g.create(gid, 'ZoneTypeNew'.encode())

    self.attr_rw.write_str_33(gid, b'some_attribute', 'AttrValue')
    self.attr_rw.write_flag(gid)
    gid.close()
    fid.close()

    out = get_subprocess_stdout(f"h5ls -gvd {tmp_hdf_file}/Base/ZoneU/ZoneTypeNew")
    check_h5ls_output(out, "Attribute: some_attribute scalar", '"AttrValue"')
    check_h5ls_output(out, "Attribute: flags {1}", '1')

def test_add_root_attribute(tmp_hdf_file):
  fid = h5f.open(bytes(tmp_hdf_file, 'utf-8'), h5f.ACC_RDWR)
  gid = HCG.open_from_path(fid, 'Base/ZoneU')
  gid = h5g.create(gid, 'FakeRoot'.encode())
  HCG.add_root_attributes(gid)
  gid.close()
  fid.close()

  out = get_subprocess_stdout(f"h5ls -gvd {tmp_hdf_file}/Base/ZoneU/FakeRoot")
  check_h5ls_output(out, "Attribute: label scalar", '"Root Node of HDF5 File"')

  cmd = ["h5ls", "-gvd", f"{tmp_hdf_file}/Base/ZoneU/FakeRoot/ hdf5version"]
  assert subprocess.run(cmd, capture_output=True).returncode == 0
  cmd = ["h5ls", "-gvd", f"{tmp_hdf_file}/Base/ZoneU/FakeRoot/ format"]
  assert subprocess.run(cmd, capture_output=True).returncode == 0

def test_is_combinated():
  assert HCG.is_combinated([[0], [1], [100], [1], [0], [1], [100], [1], [100], [0]]) == False
  assert HCG.is_combinated([[0,0], [1,1], [1,10], [1,1], [0,0], [1,1], [1,10], [1,1], [1,100], [0]]) == False
  assert HCG.is_combinated([[0], [1], [10], [1], [[0, 0, 0], [1, 1, 1], [10, 1, 1], [1, 1, 1]], [10, 2, 5], [0]]) == True

def test_group_by():
  for i, elts in enumerate(HCG.group_by(['a','b','c', 'd','e','f'], 3)):
    if i == 0:
      assert elts == ('a', 'b', 'c')
    if i == 1:
      assert elts == ('d', 'e', 'f')

def test_open_from_path(ref_hdf_file):
  fid = h5f.open(bytes(ref_hdf_file, 'utf-8'), h5f.ACC_RDONLY)
  # If this not raise, open is OK
  gid = HCG.open_from_path(fid, 'Base/ZoneS/ZoneType')
  gid = HCG.open_from_path(fid, 'Base')

def test_load_data(ref_hdf_file):
  fid = h5f.open(bytes(ref_hdf_file, 'utf-8'), h5f.ACC_RDONLY)
  gid = HCG.open_from_path(fid, 'Base/ZoneU')
  data = HCG.load_data(gid)
  assert np.array_equal(data, [[6,0,0]]) and data.dtype == np.int32 and data.flags.f_contiguous

  gid = HCG.open_from_path(fid, 'Base/ZoneU/GridCoordinates/CoordinateX')
  data = HCG.load_data(gid)
  assert np.array_equal(data, [1,2,3,4,5,6]) and data.dtype == np.float64

  gid = HCG.open_from_path(fid, 'Base/ZoneS/GridCoordinates/CoordinateX')
  data = HCG.load_data(gid)
  assert np.array_equal(data, [[1,2], [3,4]]) and data.dtype == np.float64 and np.isfortran(data)

def test_write_data(tmp_hdf_file):
  fid = h5f.open(bytes(tmp_hdf_file, 'utf-8'), h5f.ACC_RDWR)
  gid = HCG.open_from_path(fid, 'Base/ZoneU/GridCoordinates')
  gid = h5g.create(gid, 'CoordinateZ'.encode())
  HCG.write_data(gid, np.array([0,0,0, 1,1,1], np.float64))
  gid.close()
  fid.close()

  out = get_subprocess_stdout(f"h5ls -vd {tmp_hdf_file}/Base/ZoneU/GridCoordinates/CoordinateZ")
  idx = out.index("Type:      native double") #Will raise if not in list
  idx = out.index("Data:") #Will raise if not in list
  if out[idx+1].startswith('('):
    out[idx+1] = out[idx+1][4:] #Some hdf version include (0) before data : remote it
  assert out[idx+1] == '0, 0, 0, 1, 1, 1'

def test_write_link(tmp_hdf_file):
  fid = h5f.open(bytes(tmp_hdf_file, 'utf-8'), h5f.ACC_RDWR)
  gid = HCG.open_from_path(fid, 'Base/ZoneU/GridCoordinates')
  HCG.write_link(gid, 'CoordinateZ', 'this/hdf/file.hdf', 'this/node')
  gid.close()
  fid.close()
  out = get_subprocess_stdout(f"h5ls -vdg {tmp_hdf_file}/Base/ZoneU/GridCoordinates/CoordinateZ")
  check_h5ls_output(out, "Attribute: type scalar", '"LK"')
  out = get_subprocess_stdout(f"h5ls -vd {tmp_hdf_file}/Base/ZoneU/GridCoordinates/CoordinateZ")
  idx = out.index("\\ file                   Dataset {18/18}")
  idx = out.index("\\ path                   Dataset {10/10}")
  for l in out:
    if "\\ link" in l:
      assert "External Link {this/hdf/file.hdf//this/node}" in l
      break
  else:
    assert False

def test_load_data_partial(ref_hdf_file):
  fid = h5f.open(bytes(ref_hdf_file, 'utf-8'), h5f.ACC_RDONLY)

  gid = HCG.open_from_path(fid, 'Base/ZoneU/GridCoordinates/CoordinateX')
  data = HCG.load_data_partial(gid, [[0], [1], [3], [1], [3], [1], [3], [1], [6], [1]])
  assert np.array_equal(data, [4,5,6]) and data.dtype == np.float64

  gid = HCG.open_from_path(fid, 'Base/ZoneS/GridCoordinates/CoordinateX')
  # Simple
  data = HCG.load_data_partial(gid, [[0,0], [1,1], [1,2], [1,1], 
                                     [1,0], [1,1], [1,2], [1,1], [2,2], [0]])
  assert np.array_equal(data, [[3,4]]) and data.dtype == np.float64 and data.flags.f_contiguous

  # Multiple
  data = HCG.load_data_partial(gid, [[0], [1], [3], [1], 
                                     [[1,0], [1,1], [1,2], [1,1], [0,0],[1,1],[1,1],[1,1]], [2,2], [0]])
  assert np.array_equal(data, [1,3,4]) and data.dtype == np.float64

@pytest.mark.parametrize('combinated', [False, True])
def test_write_data_partial(tmp_hdf_file, combinated):
  fid = h5f.open(bytes(tmp_hdf_file, 'utf-8'), h5f.ACC_RDWR)
  gid = HCG.open_from_path(fid, 'Base/ZoneU/GridCoordinates')
  gid = h5g.create(gid, 'CoordinateZ'.encode())
  if combinated:
    filter = [[0], [1], [4], [1], [[0], [1], [3], [1], [5], [1], [1], [1]], [6], [1]]
    array = np.array([-10,-20,-30,-100], np.float64)
    indexes = np.array([0,1,2,5])
  else:
    filter = [[0], [1], [2], [1], [3], [1], [2], [1], [6], [1]]
    array = np.array([-1,-2], np.float64)
    indexes = np.array([3,4])
  HCG.write_data_partial(gid, array, filter)
  gid.close()
  fid.close()

  out = get_subprocess_stdout(f"h5ls -vd {tmp_hdf_file}/Base/ZoneU/GridCoordinates/CoordinateZ")
  idx = out.index("Data:") #Will raise if not in list
  if out[idx+1].startswith('('):
    out[idx+1] = out[idx+1][4:] #Some hdf version include (0) before data : remote it
  written_array = np.array([float(x) for x in out[idx+1].split(',')])
  assert np.allclose(written_array[indexes], array)

@pytest.mark.parametrize('partial', [True,False])
def test_load_node_partial(partial, ref_hdf_file):
  fid = h5f.open(bytes(ref_hdf_file, 'utf-8'), h5f.ACC_RDONLY)
  gid = HCG.open_from_path(fid, 'Base/ZoneU/GridCoordinates')

  parent = ['ZoneU', None, [], 'Zone_t']
  ancestors_stack = (['Base', 'ZoneU'], ['CGNSBase_t', 'Zone_t'])

  if partial:
    # Load only one array
    HCG._load_node_partial(gid, parent, lambda N,L : N[-1] != 'CoordinateY', ancestors_stack)
    yt = """
    ZoneU Zone_t:
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [1,2,3,4,5,6]:
        CoordinateY DataArray_t:
        CoordinateY#Size DataArray_t I8 [6]:
    """
  else:
    HCG._load_node_partial(gid, parent, lambda N,L : True, ancestors_stack)
    yt = """
    ZoneU Zone_t:
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 [1,2,3,4,5,6]:
        CoordinateY DataArray_t R8 [-1,-2,-3,-4,-5,-6]:
    """

  assert PT.is_same_tree(parent, parse_yaml_cgns.to_node(yt))

def test_write_node_partial(tmp_hdf_file):
  tree = parse_yaml_cgns.to_cgns_tree(sample_tree)
  node = PT.get_node_from_path(tree, 'Base/ZoneU/GridCoordinates')
  ancestors_stack = (['Base', 'ZoneU', 'GridCoordinates'], ['CGNSBase_t', 'Zone_t', 'GridCoordinates_t'])

  fid = h5f.open(bytes(tmp_hdf_file, 'utf-8'), h5f.ACC_RDWR)
  gid = HCG.open_from_path(fid, 'Base/ZoneU')

  gid.unlink(b'GridCoordinates') # Remove before writting
  HCG._write_node_partial(gid, node, lambda N,L: N[-1] != 'CoordinateY', ancestors_stack)

  gid.close()
  fid.close()

  # Check node
  out = get_subprocess_stdout(f"h5ls -vgd {tmp_hdf_file}/Base/ZoneU/GridCoordinates")
  check_h5ls_output(out, "Attribute: type scalar", '"MT"')
  check_h5ls_output(out, "Attribute: label scalar", '"GridCoordinates_t"')

  #Check children
  for coord in ['X', 'Y']:
    out = get_subprocess_stdout(f"h5ls -vgd {tmp_hdf_file}/Base/ZoneU/GridCoordinates/Coordinate{coord}")
    check_h5ls_output(out, "Attribute: type scalar", '"R8"')

    out = get_subprocess_stdout(f"h5ls -r {tmp_hdf_file}/Base/ZoneU/GridCoordinates/Coordinate{coord}")
    if coord == 'X':
      assert 'data' in out[0]
    else:
      assert out == ['']

@pytest.mark.parametrize('partial', [True,False])
def test_load_tree_partial(partial, ref_hdf_file):
  if partial:
    tree = HCG.load_tree_partial(ref_hdf_file, lambda N,L : N[-1] != 'CoordinateY')
    yt = """
    Base CGNSBase_t [2,2]:
      ZoneU Zone_t [[6, 0, 0]]:
        ZoneType ZoneType_t "Unstructured":
        GridCoordinates GridCoordinates_t:
          CoordinateX DataArray_t R8 [1., 2., 3., 4., 5., 6.]:
          CoordinateY DataArray_t:
          CoordinateY#Size DataArray_t I8 [6]:
      ZoneS Zone_t [[2, 1, 0], [2, 1, 0]]:
        ZoneType ZoneType_t 'Structured':
        GridCoordinates GridCoordinates_t:
          CoordinateX DataArray_t R8 [[1., 2.], [3., 4.]]:
          CoordinateY DataArray_t:
          CoordinateY#Size DataArray_t I8 [2,2]:
    """
  else:
    tree = HCG.load_tree_partial(ref_hdf_file, lambda N,L : True)
    yt = sample_tree
  assert PT.is_same_tree(tree, parse_yaml_cgns.to_cgns_tree(yt))

def test_write_tree_partial(tmp_path, ref_hdf_file):
  tree = parse_yaml_cgns.to_cgns_tree(sample_tree)
  outfile = str(tmp_path / Path('only_coords.hdf'))
  HCG.write_tree_partial(tree, outfile, lambda N,L : True)
  cmd = ["h5diff", f"{ref_hdf_file}", f"{outfile}", "Base"] #hdf5version dataset can vary
  assert subprocess.run(cmd).returncode == 0
