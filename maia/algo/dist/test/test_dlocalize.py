import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia
from   maia           import npy_pdm_gnum_dtype as pdm_dtype
from   maia.utils     import test_utils as TU

from maia.algo.dist import localize as LOC

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ['S', 'TRI_3'])
@pytest.mark.parametrize("location", ['Vertex', 'CellCenter'])
def test_get_point_cloud(elt_kind, location, comm):
  n_vtx = [3,3] if elt_kind == 'S' else 3
  tree = maia.factory.generate_dist_block(n_vtx, elt_kind, comm)

  pt_cloud = LOC.get_point_cloud(PT.get_all_Zone_t(tree)[0], comm, location)

  if location == 'Vertex':
    expected_lngn = [np.array([1,2,3,4,5]), np.array([6,7,8,9])][comm.rank]
    expected_coor = [np.array([0,0,0, .5,0,0, 1,0,0,   0,.5,0, .5,.5,0]),
                      np.array([1,.5,0,   0,1,0, .5,1,0, 1,1,0])][comm.rank]
  else:
    if elt_kind == 'S':
      expected_lngn = [np.array([1,2]), np.array([3,4])][comm.rank]
      expected_coor = [np.array([.25,.25,0, .75,.25,0]), np.array([.25,.75,0, .75,.75,0])][comm.rank]
    else:
      expected_lngn = [np.array([1,2,3,4]), np.array([5,6,7,8])][comm.rank]
      expected_coor = [np.array([1,1,0, 2,2,0, 4,1,0, 5,2,0]) / 6.,
                       np.array([1,4,0, 2,5,0, 4,4,0, 5,5,0]) / 6.][comm.rank]

  assert np.array_equal(pt_cloud[0], expected_coor)
  assert np.array_equal(pt_cloud[1], expected_lngn)

@pytest_parallel.mark.parallel(1)
def test_get_point_cloud_from_ctn(comm):
  tree = maia.factory.generate_dist_block(11, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  maia.algo.compute_elements_center(zone, 'CellCenter', comm)
  ctn = PT.get_node_from_name(zone, 'Geometry_3d')
  for child in PT.get_children(ctn):
    PT.set_name(child, PT.get_name(child).replace('Center', 'Coordinate'))

  pt_cloud = LOC.get_point_cloud(zone, comm, 'Geometry_3d')
  assert (pt_cloud[1] == np.arange(1, PT.Zone.n_cell(zone)+1)).all()

  with pytest.raises(RuntimeError):
    LOC.get_point_cloud(zone, comm, 'MissingNode')

@pytest_parallel.mark.parallel(2)
def test_minimal_partitioning_poly3d(comm):
  tree = maia.factory.generate_dist_block([3,2,2], 'HEXA_8', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  part_data = LOC.minimal_partitioning(PT.get_all_Zone_t(tree)[0], comm)

  assert (part_data[5] == [[1], [2]][comm.rank]).all() # Cell gnum
  assert (part_data[7] == [[1,2,4,5,7,8,10,11], [2,3,5,6,8,9,11,12]][comm.rank]).all() # vtx gnum
  assert (part_data[6] == [[1,3,4,6,7,10],   # NB : face numbering is not regular, because of elt -> ng
                           [2,5,6,8,9,11]][comm.rank]).all() # Face gnum
  assert (part_data[0] == [0,6]).all() # Cell face idx
  assert (part_data[1] == [[1,2,3,4,5,6], [-3,1,2,4,5,6]][comm.rank]).all() # Cell face
  assert (part_data[2] == [0,4,8,12,16,20,24]).all() # Face vtx idx
  expected_face_vtx = np.array([3,4,2,1, 2,6,5,1, 5,7,3,1, 4,8,6,2, 7,8,4,3, 8,7,5,6])
  if comm.rank == 1:
   expected_face_vtx[8:12] = [3,7,5,1] # Face vtx is almost same for 2 ranks except for shared face
  assert (part_data[3] == expected_face_vtx).all()
  assert (part_data[4] == [[0,0,0, .5,0,0, 0,1,0, .5,1,0, 0,0,1, .5,0,1, 0,1,1, .5,1,1], 
                           [.5,0,0, 1,0,0, .5,1,0, 1,1,0, .5,0,1, 1,0,1, .5,1,1, 1,1,1]][comm.rank]).all() # Coords

@pytest_parallel.mark.parallel(2)
def test_minimal_partitioning_S2d(comm):
  tree = maia.factory.generate_dist_block([4,3], 'S', comm)
  part_data = LOC.minimal_partitioning(PT.get_all_Zone_t(tree)[0], comm)

  assert (part_data[3] == [[1,2,3], [4,5,6]][comm.rank]).all() # Cell gnum
  assert (part_data[4] == [[1,2,3,4,5,6,7,8], [5,6,7,8,9,10,11,12]][comm.rank]).all() # Vtx gnum
  assert (part_data[2] == [[0,0,0, 1/3,0,0, 2/3,0,0, 1,0,0,  0,.5,0, 1/3,.5,0, 2/3,.5,0, 1,.5,0], # Coords
                           [0,.5,0, 1/3,.5,0, 2/3,.5,0, 1,.5,0,  0,1,0, 1/3,1,0, 2/3,1,0, 1,1,0]][comm.rank]).all()
  assert (part_data[0] == [0,4,8,12]).all() # Cell vtx idx
  assert (part_data[1] == [1,2,6,5, 2,3,7,6, 3,4,8,7]).all() # Cell vtx

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("src_kind", ['S', 'Poly'])
@pytest.mark.parametrize("tgt_kind", ['S', 'QUAD_4'])
def test_localize_2d(src_kind, tgt_kind, comm):
  n_vtx_src = [6,6]   if src_kind == 'S' else 6
  n_vtx_tgt = [21,21] if tgt_kind == 'S' else 21
  _src_kind = 'QUAD_4' if src_kind == 'Poly' else 'S'
  src_tree = maia.factory.generate_dist_block(n_vtx_src, _src_kind, comm)
  tgt_tree = maia.factory.generate_dist_block(n_vtx_tgt, tgt_kind,  comm)
  if src_kind == 'Poly':
    maia.algo.dist.convert_elements_to_ngon(src_tree, comm)
  
  maia.algo.localize_points(src_tree, tgt_tree, 'CellCenter', comm)
  
  zone = PT.get_node_from_label(tgt_tree, 'Zone_t')
  cell_distri = MT.get_distribution(zone, 'Cell')[1]

  expected_f = ((np.arange(400) // 4 )  % 5) + np.repeat([1,6,11,16,21], 80) # Regular pattern
  expected = expected_f[cell_distri[0] : cell_distri[1]]

  assert (PT.get_node_from_path(zone, 'Localization/SrcId')[1] == expected).all()
  assert (PT.get_node_from_path(zone, 'Localization/DomId')[1] == 1).all()

  assert PT.get_value(PT.get_node_from_path(zone, 'Localization/DomainList')) == "Base/zone"


@pytest_parallel.mark.parallel(2)
def test_localize_mdom(comm):
  yaml_path = TU.mesh_dir / 'S_twoblocks.yaml'
  tree = maia.io.file_to_dist_tree(yaml_path, comm)

  maia.io.dist_tree_to_file(tree, 'src.cgns', comm)

  src_parts  = LOC._collect_source(PT.get_all_Zone_t(tree), comm)

  # In this test we create the points to localize (clouds) by hand
  if comm.rank == 0:
    tgt_clouds = (np.array([3.2,5.4,3.8,  22.2,6.5,3.3,  17, 0.5, 0.5]),  # Big, small, outside
                  np.array([3,2,1], pdm_dtype))
  elif comm.rank == 1:
    tgt_clouds = (np.array([], float), np.array([], pdm_dtype))

  result, result_inv = LOC._mdom_mesh_location(src_parts, [tgt_clouds], comm, reverse=True)

  result = result[0] # Extract 1st point cloud
  result_inv_large = result_inv[0] # Extract inv. results for each block
  result_inv_small = result_inv[1]
  if comm.rank == 0:
    assert (result['domain'] == [1, 2]).all()
    assert (result['location'] == [468, 8]).all()
    assert (result['location_shifted'] == [468 + 0, 8 + 16*8*6]).all()
    assert (result['located_ids'] == [0,1]).all()
    assert (result['unlocated_ids'] == [2]).all()

  # Point in big zone is in cell 468, which in on rank 1 (local id = 468 - 384 - 1 = 83)
  pts_inside = result_inv_large['points_gnum']
  expected_cnt = np.zeros_like(pts_inside.counts)
  if comm.rank == 0:
    assert np.array_equal(pts_inside.counts, expected_cnt)
    assert pts_inside.dsize == 0
  if comm.rank == 1:
    expected_cnt[83] = 1
    assert np.array_equal(pts_inside.counts, expected_cnt)
    assert np.array_equal(pts_inside.values, [3])

  # Point in small zone is in cell 8, which in on rank 0 (local id = 8 - 1 = 7)
  pts_inside = result_inv_small['points_gnum']
  expected_cnt = np.zeros_like(pts_inside.counts)
  if comm.rank == 0:
    expected_cnt[7] = 1
    assert np.array_equal(pts_inside.counts, expected_cnt)
    assert np.array_equal(pts_inside.values, [2])
  if comm.rank == 1:
    assert np.array_equal(pts_inside.counts, expected_cnt)
    assert pts_inside.dsize == 0