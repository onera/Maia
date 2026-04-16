import os
import warnings
import numpy as np
import pytest
import pytest_parallel
from mpi4py import MPI
import maia
import maia.pytree as PT
import maia.io.cgns_io_tree as IOT
import maia.utils.test_utils as TU

from maia import npy_pdm_gnum_dtype as pdm_dtype
dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

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
  tree = maia.factory.generate_dist_block(4, "TRI_3", comm)
  tmp_dir = TU.create_collective_tmp_dir(comm)

  tmp_file = os.path.join(tmp_dir, f'TEST/test.cgns')
  maia.io.write_trees(tree, tmp_file, comm)

  comm.barrier()
  for i in range(comm.Get_size()):
    assert os.path.exists(os.path.join(tmp_dir, f'TEST/test_{i}.cgns'))

  TU.rm_collective_dir(tmp_dir, comm)


@pytest_parallel.mark.parallel(2)
def test_fill_size_tree(comm):
  filename = str(TU.sample_mesh_dir / 'only_coords.hdf')

  cx_u = [[1.,2.,3.], [4.,5.,6.]][comm.rank]
  cy_u = [[-1.,-2.,-3.], [-4.,-5.,-6.]][comm.rank]
  cx_s = [[1.0, 3.0],[2.,4.]][comm.rank]
  cy_s = [[-1.0, 0.0],[-1.,0.]][comm.rank]
  distri_vtx_u = [[0,3,6], [3,6,6]][comm.rank]
  distri_cell_u = [[0,0,0], [0,0,0]][comm.rank]
  distri_vtx_s = [[0,2,4], [2,4,4]][comm.rank]
  distri_cell_s = [[0,1,1], [1,1,1]][comm.rank]

  expected = PT.yaml.to_cgns_tree(f"""
  Base CGNSBase_t I4 [2, 2]:
    ZoneU Zone_t I4 [[6, 0, 0]]:
      ZoneType ZoneType_t 'Unstructured':
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 {cx_u}:
        CoordinateY DataArray_t R8 {cy_u}:
      :CGNS#Distribution UserDefinedData_t:
        Vertex DataArray_t {dtype}  {distri_vtx_u}:
        Cell DataArray_t {dtype} {distri_cell_u}:
    ZoneS Zone_t I4 [[2, 1, 0], [2, 1, 0]]:
      ZoneType ZoneType_t 'Structured':
      GridCoordinates GridCoordinates_t:
        CoordinateX DataArray_t R8 {cx_s}:
        CoordinateY DataArray_t R8 {cy_s}:
      :CGNS#Distribution UserDefinedData_t:
        Vertex DataArray_t {dtype} {distri_vtx_s}:
        Cell DataArray_t {dtype} {distri_cell_s}:
  """)

  dist_tree = IOT.file_to_dist_tree(filename, comm)
  assert PT.is_same_tree(expected, dist_tree)


@pytest_parallel.mark.parallel(2)
def test_load_size_tree(comm):
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

  size_tree = IOT.load_size_tree(filename, comm)
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


@pytest_parallel.mark.parallel(2)
def test_load_from_filter(comm):

  # Create U/NGON tree for test
  tmp_dir = TU.create_collective_tmp_dir(comm)
  filename = os.path.join(tmp_dir, 'tree.cgns')

  if comm.rank == 0:
    dist_tree = maia.factory.generate_dist_block([4,2,2], 'S', MPI.COMM_SELF)
    maia.algo.dist.convert_s_to_ngon(dist_tree, MPI.COMM_SELF)
    maia.io.dist_tree_to_file(dist_tree, filename, MPI.COMM_SELF)

  comm.barrier()

  # Prepare tested function arguments
  size_tree = maia.io.cgns_io_tree.load_size_tree(filename, comm)
  IOT.add_distribution_info(size_tree, comm)
  hdf_filter = IOT.create_tree_hdf_filter(size_tree)
  hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}

  # Test function
  IOT.load_tree_from_filter(filename, size_tree, comm, hdf_filter)
  ngon_node = PT.Zone.NGonNode(PT.get_all_Zone_t(size_tree)[0])

  if comm.rank==0:
    ngon_expected = PT.yaml.to_node("""
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
  elif comm.rank==1:
    ngon_expected = PT.yaml.to_node("""
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

  assert PT.is_same_node(ngon_node, ngon_expected)

  TU.rm_collective_dir(tmp_dir, comm)


@pytest.fixture()
def incomplete_zsr_file(comm):
  yt = PT.yaml.to_cgns_tree("""
    Base CGNSBase_t [2,2]:
      ZoneU Zone_t [[6, 0, 0]]:
        ZoneType ZoneType_t "Unstructured":
        ZSR_AIRFOIL ZoneSubRegion_t:
          GridLocation GridLocation_t "FaceCenter":
          Density DataArray_t R8 [10., 11., 12.]:
  """)
  tmp_dir = TU.create_collective_tmp_dir(comm)
  tmp_file = tmp_dir/'yt.cgns'
  if comm.rank == 0:
    maia.io.write_tree(yt, tmp_file)
  comm.barrier() # wait for rank 0 to finish writing
  return tmp_file

@pytest.fixture()
def incomplete_zsr_dist_tree(comm):
  if comm.rank == 0:
    vertex_dist = '[0, 3, 6]'
    zsr_dist = '[0, 2, 3]'
    zsr_density = '[10., 11.]'
  elif comm.rank == 1:
    vertex_dist = '[3, 6, 6]'
    zsr_dist = '[2, 3, 3]'
    zsr_density = '[12.]'
  dist_yt = PT.yaml.to_cgns_tree(f"""
    Base CGNSBase_t [2,2]:
      ZoneU Zone_t [[6, 0, 0]]:
        ZoneType ZoneType_t "Unstructured":
        :CGNS#Distribution UserDefinedData_t:
          Vertex DataArray_t {dtype} {vertex_dist}:
          Cell DataArray_t {dtype} [0, 0, 0]:
        ZSR_AIRFOIL ZoneSubRegion_t:
          GridLocation GridLocation_t "FaceCenter":
          Density DataArray_t R8 {zsr_density}:
          :CGNS#Distribution UserDefinedData_t:
            Index DataArray_t {dtype} {zsr_dist}:
  """)
  return dist_yt

@pytest_parallel.mark.parallel(2)
def test_read_incomplete_zsr(comm, incomplete_zsr_file, incomplete_zsr_dist_tree):
  dist_tree = maia.io.file_to_dist_tree(incomplete_zsr_file, comm)

  assert PT.is_same_tree(dist_tree, incomplete_zsr_dist_tree)
  TU.rm_collective_dir(incomplete_zsr_file.parent, comm)

@pytest_parallel.mark.parallel(2)
def test_write_incomplete_zsr(comm, incomplete_zsr_dist_tree):
  tmp_dir = TU.create_collective_tmp_dir(comm)

  maia.io.dist_tree_to_file(incomplete_zsr_dist_tree, tmp_dir/'test_write.cgns', comm)

  # We suppose reading is OK (is tested above)
  dist_tree_from_write = maia.io.file_to_dist_tree(tmp_dir/'test_write.cgns', comm)
  assert PT.is_same_tree(dist_tree_from_write, incomplete_zsr_dist_tree)
  TU.rm_collective_dir(tmp_dir, comm)


@pytest.fixture()
def incomplete_fs_file(comm):
  yt = PT.yaml.to_cgns_tree("""
    Base CGNSBase_t [2,2]:
      ZoneU Zone_t [[6, 3, 0]]:
        ZoneType ZoneType_t "Unstructured":
        FS_CC_Complete FlowSolution_t:
          GridLocation GridLocation_t "CellCenter":
          Density DataArray_t R8 [10., 11., 12.]:
        FS_CC_Partial FlowSolution_t:
          GridLocation GridLocation_t "CellCenter":
          Density DataArray_t R8 [100., 101.]:
        FS_FC FlowSolution_t:
          GridLocation GridLocation_t "FaceCenter":
          Density DataArray_t R8 [0., 1., 2., 3., 5.]:
  """)
  tmp_dir = TU.create_collective_tmp_dir(comm)
  tmp_file = tmp_dir/'incomplete_fs.cgns'
  if comm.rank == 0:
    maia.io.write_tree(yt, tmp_file)
  comm.barrier() # wait for rank 0 to finish writing
  return tmp_file

@pytest.fixture()
def incomplete_fs_dist_tree(comm):
  if comm.rank == 0:
    vtx_dist           = '[0, 3, 6]'
    cell_dist          = '[0, 2, 3]'
    fs_cc_partial_dist = '[0, 1, 2]'
    fs_fc_dist         = '[0, 3, 5]'
    fs_cc_complete = '[10., 11.]'
    fs_cc_partial  = '[100.]'
    fs_fc          = '[0., 1., 2.]'
  elif comm.rank == 1:
    vtx_dist           = '[3, 6, 6]'
    cell_dist          = '[2, 3, 3]'
    fs_cc_partial_dist = '[1, 2, 2]'
    fs_fc_dist         = '[3, 5, 5]'
    fs_cc_complete = '[12.]'
    fs_cc_partial  = '[101.]'
    fs_fc          = '[3., 5.]'
  dist_yt = PT.yaml.to_cgns_tree(f"""
    Base CGNSBase_t [2,2]:
      ZoneU Zone_t [[6, 3, 0]]:
        ZoneType ZoneType_t "Unstructured":
        :CGNS#Distribution UserDefinedData_t:
          Vertex DataArray_t {dtype} {vtx_dist}:
          Cell   DataArray_t {dtype} {cell_dist}:
        FS_CC_Complete FlowSolution_t:
          GridLocation GridLocation_t "CellCenter":
          Density DataArray_t R8 {fs_cc_complete}:
        FS_CC_Partial FlowSolution_t:
          GridLocation GridLocation_t "CellCenter":
          Density DataArray_t R8 {fs_cc_partial}:
          :CGNS#Distribution UserDefinedData_t:
            Index DataArray_t {dtype} {fs_cc_partial_dist}:
        FS_FC FlowSolution_t:
          GridLocation GridLocation_t "FaceCenter":
          Density DataArray_t R8 {fs_fc}:
          :CGNS#Distribution UserDefinedData_t:
            Index DataArray_t {dtype} {fs_fc_dist}:
  """)
  return dist_yt

@pytest_parallel.mark.parallel(2)
def test_read_incomplete_fs(comm, incomplete_fs_file, incomplete_fs_dist_tree):
  dist_tree = maia.io.file_to_dist_tree(incomplete_fs_file, comm)

  assert PT.is_same_tree(dist_tree, incomplete_fs_dist_tree)
  TU.rm_collective_dir(incomplete_fs_file.parent, comm)


@pytest_parallel.mark.parallel(1)
def test_ptcl_dist_tree_to_file_1proc(comm):
    yt = """
Base CGNSBase_t I4 [3, 3]:
  ParticleZone ParticleZone_t I4 [3]:
    ParticleCoordinates ParticleCoordinates_t:
      CoordinateX DataArray_t R8 [0., 1., 2.]:
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t I4 [0, 3, 3]:
"""

    dist_tree = PT.yaml.to_cgns_tree(yt)

    tmp_dir = TU.create_collective_tmp_dir(comm)
    out_file = os.path.join(tmp_dir, 'yt.cgns')
    maia.io.dist_tree_to_file(dist_tree, out_file, comm)

    t = maia.io.read_tree(out_file)
    assert (PT.get_value(PT.get_node_from_name(t, "CoordinateX")) == [0., 1., 2.]).all()
    TU.rm_collective_dir(tmp_dir, comm)

@pytest_parallel.mark.parallel(2)
def test_ptcl_dist_tree_to_file_2procs(comm):
    rank = comm.Get_rank()
    if rank == 0:
        yt = """
CGNSTree CGNSTree_t:
  Base CGNSBase_t I4 [3, 3]:
    ParticleZone ParticleZone_t 7:
      :CGNS#Distribution UserDefinedData_t:
        Vertex DataArray_t I8 [0, 3, 7]:
      ParticleCoordinates ParticleCoordinates_t:
        CoordinateX DataArray_t R8 [0.0, 3.0, 6.0]:
        CoordinateY DataArray_t R8 [1.0, 4.0, 7.0]:
        CoordinateZ DataArray_t R8 [2.0, 5.0, 8.0]:
      ParticleSolution ParticleSolution_t:
        Identifier DataArray_t I8 [0, 3, 6]:
"""
    else:
        yt = """
CGNSTree CGNSTree_t:
  Base CGNSBase_t I4 [3, 3]:
    ParticleZone ParticleZone_t 7:
      :CGNS#Distribution UserDefinedData_t:
        Vertex DataArray_t I8 [3, 7, 7]:
      ParticleCoordinates ParticleCoordinates_t:
        CoordinateX DataArray_t R8 [9.0, 12.0, 15.0, 18.0]:
        CoordinateY DataArray_t R8 [10.0, 13.0, 16.0, 19.0]:
        CoordinateZ DataArray_t R8 [11.0, 14.0, 17.0, 20.0]:
      ParticleSolution ParticleSolution_t:
        Identifier DataArray_t I8 [9, 12, 15, 18]:
"""

    dist_tree = PT.yaml.to_cgns_tree(yt)

    tmp_dir = TU.create_collective_tmp_dir(comm)
    out_file = os.path.join(tmp_dir, 'yt.cgns')
    maia.io.dist_tree_to_file(dist_tree, out_file, comm)

    t = maia.io.file_to_dist_tree(out_file, comm)
    coordX = PT.get_value(PT.get_node_from_name(t, "CoordinateX"))
    identifier = PT.get_value(PT.get_node_from_name(t, "Identifier"))
    if rank == 0:
        assert (coordX == [0., 3., 6., 9.]).all()
        assert (identifier == [0, 3, 6, 9]).all()
    else:
        assert (coordX == [12., 15., 18.]).all()
        assert (identifier == [12, 15, 18]).all()

    TU.rm_collective_dir(tmp_dir, comm)

@pytest_parallel.mark.parallel(1)
def test_dist_tree_to_file_long_names(comm):
  yt = f"""
Base CGNSBase_t I4 [3, 3]:
  ZoneWithALongLongLoooooongNameThatIsLongerThan32 Zone_t I4 [[1, 0, 0]]:
    ZoneType ZoneType_t 'Unstructured':
    :CGNS#Distribution UserDefinedData_t:
      Vertex DataArray_t {dtype} [0, 1, 1]:
      Cell DataArray_t {dtype} [0, 0, 0]:
    FlowSolution FlowSolution_t:
      AVeryLongFieldNameWithLotsOfDetailsAboutTurbulentDensityRootMeanSquareResidual1 DataArray_t R8 [0.]:
      Density DataArray_t R8 [0.]:
      RSDTurbulentDissipationRateDensityRMS DataArray_t R8 [0.]:
      AnotherVeryLongFieldNameWithLotsOfDetailsAboutTurbulentDensityRootMeanSquareResidual DataArray_t R8 [0.]:
      AVeryLongFieldNameWithLotsOfDetailsAboutTurbulentDensityRootMeanSquareResidual2 DataArray_t R8 [0.]:
"""

  dist_tree = PT.yaml.to_cgns_tree(yt)

  tmp_dir = TU.create_collective_tmp_dir(comm)

  IOT.dist_tree_to_file(dist_tree, tmp_dir/'yt.cgns', comm)
  loaded_dist_tree = IOT.file_to_dist_tree(tmp_dir/'yt.cgns', comm)

  assert PT.is_same_tree(loaded_dist_tree, dist_tree)

  TU.rm_collective_dir(tmp_dir, comm)

def test_long_links(tmp_path, comm):
  tree = PT.yaml.to_cgns_tree(f"""
  Base CGNSBase_t I4 [3, 3]:
    ZoneWithALongNameThatIsLongerThan32 Zone_t I4 [[1, 0, 0]]:
      ZoneType ZoneType_t 'Unstructured':
      FlowSolution FlowSolution_t:
        Density DataArray_t R8 [0.]:
        AnotherVeryLongFieldNameWithLotsOfDetails DataArray_t R8 [0.]:
  """)
  fname = str(tmp_path / 'tree.cgns')
  
  # Intermediate long name
  links = [['.', 'other/file.cgns', 'other/link', 'Base/ZoneWithALongNameThatIsLongerThan32/FlowSolution']]
  maia.io.write_tree(tree, fname, links)
  r_links = maia.io.read_links(fname)
  assert r_links == links

  # Terminal long name
  links = [['.', 'other/file.cgns', 'other/link',
            'Base/ZoneWithALongNameThatIsLongerThan32/FlowSolution/AnotherVeryLongFieldNameWithLotsOfDetails']]
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    dtree = PT.deep_copy(tree)
  distri = PT.yaml.to_node(f"""
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {dtype} [0, 1, 1]:
    Cell DataArray_t {dtype} [0, 0, 0]:
  """)
  PT.add_child(PT.find_node_from_label(dtree, 'Zone_t'), distri)

  maia.io.dist_tree_to_file(dtree, fname, comm, links)
  r_links = maia.io.read_links(fname)
  assert r_links == links

  # Terminal long name, implicit
  links = [['.', 'other/file.cgns', 'other/link',
            'Base/ZoneWithALongNameThatIsLongerThan32/FlowSolution/ImplicitLongFieldNameWithLotsOfDetails']]
  maia.io.write_tree(tree, fname, links)
  r_links = maia.io.read_links(fname)
  assert r_links == links