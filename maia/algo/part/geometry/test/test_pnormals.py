import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT

from maia.algo.part.geometry import normals as GEO

@pytest.mark.parametrize('unitary', [False, True])
@pytest.mark.parametrize('elt_kind', ['TRI_3', 'Poly'])
@pytest_parallel.mark.parallel(3)
def test_compute_edge_normal2d(elt_kind, unitary, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  if elt_kind != 'TRI_3':
    maia.algo.dist.convert_elements_to_ngon(tree, comm)

  ptree = maia.factory.partition_dist_tree(tree, comm)

  PT.rm_nodes_from_name(ptree, 'CoordinateZ')
  zone = PT.get_all_Zone_t(ptree)[0]
  
  edge_normal = GEO.compute_edge_normal(zone, unitary)

  if elt_kind == 'TRI_3': # Only external edges are computed
    coef = 0.5 if not unitary else 1
    if comm.Get_rank() == 0:
      expected_edge_normal = coef * np.array([0,-1, 0,-1, -1,0])
    elif comm.Get_rank() == 1:
      expected_edge_normal = coef * np.array([0,1, 1,0, 1,0])
    elif comm.Get_rank() == 2:
      expected_edge_normal = coef * np.array([0,1, -1,0])
  elif elt_kind == 'Poly':
    ce = 1            if unitary else 0.5
    ci = 1/np.sqrt(2) if unitary else 0.5 
    if comm.Get_rank() == 0:
      expected_edge_normal = np.array([0,-ce,  -ce,0,  0,-ce,  ci,ci,  ce,0,  ci,ci,  0,ce])
    elif comm.Get_rank() == 1:
      expected_edge_normal = np.array([-ci,-ci,  ce,0,  0,ce,  -ce,0,  ci,ci,  ce,0,  0,ce])
    elif comm.Get_rank() == 2:
      expected_edge_normal = np.array([0,-ce,  -ce,0,  ci,ci,  ce,0,  0,ce])

  assert (edge_normal == expected_edge_normal).all()

@pytest.mark.parametrize('elt_kind', ['S', 'BAR_2'])
@pytest_parallel.mark.parallel(2)
def test_compute_edge_normal_1d(elt_kind, comm):
  # NB this test compute edge normal even if CellDim is 1, because PhyDim is 2
  tree = maia.factory.generate_dist_block([5], elt_kind, comm, origin=[0,0], length=[(2,1)])
  # Make a "roof" shape to have different normals
  cy = PT.get_np_value(PT.find_node_from_name(tree, 'CoordinateY'))
  if comm.rank == 1:
    cy[:] = [.25, 0]

  ptree = maia.factory.partition_dist_tree(tree, comm)
  PT.rm_nodes_from_name(ptree, 'CoordinateZ') # Bug in BAR_2 : CZ is created
  zone = PT.get_all_Zone_t(ptree)[0]
  edge_normal = GEO._compute_elements_normal(zone)

  if comm.rank == 0:
    assert (edge_normal == np.array([ 0.25,-0.5,   0.25,-0.5])).all()
  elif comm.rank == 1:
    assert (edge_normal == np.array([-0.25,-0.5,  -0.25,-0.5])).all()

@pytest.mark.parametrize('unitary', [False, True])
@pytest_parallel.mark.parallel(2)
def test_compute_face_normal3d_ngon(unitary, comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]
  
  GEO.compute_elements_normal(zone, unitary)
  # Back to dist tree for // independant comparison
  maia.transfer.part_tree_to_dist_tree_all(tree, ptree, comm)

  # All face area are 0.25
  coef = 0.25 if not unitary else 1
  if comm.rank == 0:
    expected_x = coef*np.array([0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1], float)
    expected_y = coef*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], float)
    expected_z = coef*np.array([-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,0,0,0,0,0,0], float)
  elif comm.rank == 1:
    expected_x = coef*np.array([-1,-1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0], float)
    expected_y = coef*np.array([0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1], float)
    expected_z = coef*np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], float)

  prefix = 'UnitNormal' if unitary else 'Normal'
  assert np.array_equal(PT.get_np_value(PT.find_node_from_name(tree, prefix+'X')), expected_x)
  assert np.array_equal(PT.get_np_value(PT.find_node_from_name(tree, prefix+'Y')), expected_y)
  assert np.array_equal(PT.get_np_value(PT.find_node_from_name(tree, prefix+'Z')), expected_z)

@pytest.mark.parametrize('unitary', [False, True])
@pytest_parallel.mark.parallel(1)
def test_compute_face_normal3d_elt(unitary, comm):
  tree = maia.factory.generate_dist_block([3,3,2], 'HEXA_8', comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]
  
  # Computed only for boundary faces
  face_normal = GEO.compute_face_normal(zone, unitary)

  # All face area are 0.25
  c1 = 0.25 if not unitary else 1
  c2 = 0.50 if not unitary else 1

  expected_face_normal =  np.array([ 0, 0,-c1,   0, 0,-c1,  0, 0,-c1,  0, 0,-c1,
                                     0, 0, c1,   0, 0, c1,  0, 0, c1,  0, 0, c1,
                                    -c2, 0, 0,  -c2, 0, 0,  c2, 0, 0,  c2, 0, 0,
                                     0,-c2, 0,   0,-c2, 0,  0, c2, 0,  0, c2, 0])

  assert (face_normal == expected_face_normal).all()

@pytest.mark.parametrize('elt_kind', ['Poly', 'Standard', 'S'])
@pytest_parallel.mark.parallel(2)
def test_compute_face_normal2d(elt_kind, comm):
  # NB this test compute face normal even if CellDim is 2, because PhyDim is 3
  tree = maia.factory.generate_dist_block([3,3], 'S', comm, length=[2,1])
  # Make a "roof" shape to have different normals
  cz = PT.get_np_value(PT.find_node_from_name(tree, 'CoordinateZ'))
  if comm.rank == 0:
    cz[[1,4]] += 1
  else:
    cz[[2]] += 1

  if elt_kind != 'S':
    maia.algo.dist.convert_s_to_u(tree, elt_kind, comm)
  if elt_kind == 'Poly':
    maia.algo.edge_pe_to_ngon(tree, comm) # Needed for partitioning

  ptree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(ptree)[0]

  GEO.compute_elements_normal(zone)
  # NB : split is mesh dependant (apparently along Oy axis in all cases,
  # but partition affectation is switched) so we go back to distributed
  # mesh to compare to expected result
  maia.transfer.part_tree_to_dist_tree_all(tree, ptree, comm)

  assert (PT.find_node_from_name(tree, 'NormalX')[1] == [-0.5, 0.5]).all()
  assert (PT.find_node_from_name(tree, 'NormalY')[1] == [0., 0.]).all()
  assert (PT.find_node_from_name(tree, 'NormalZ')[1] == [0.5, 0.5]).all()

@pytest.mark.parametrize('cell_dim', [2,3])
@pytest_parallel.mark.parallel(1)
def test_compute_elements_normal_face_placement(cell_dim, comm):
  # NB : in this test we just check the placement, not the values
  n_vtx = [4,4,4] if cell_dim == 3 else [4,4]
  base_tree = maia.factory.generate_dist_block(n_vtx, 'S', comm)
  
  # > S 
  tree = PT.deep_copy(base_tree)
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]
  GEO.compute_elements_normal(zone)

  if cell_dim == 3:
    assert PT.get_node_from_name(zone, 'Geometry_2d') is None
    for dir in ['I', 'J', 'K']:
      container = PT.find_node_from_name(zone, f'Geometry_2d_{dir}')
      assert PT.Subset.GridLocation(container) == f'{dir}FaceCenter'
      #assert PT.get_child_from_name(container, 'PointRange') is not None
      assert PT.get_child_from_name(container, 'NormalZ') is not None
  else:
    container = PT.find_node_from_name(zone, f'Geometry_2d')
    assert PT.Subset.GridLocation(container) == 'CellCenter'
    assert PT.get_child_from_name(container, 'PointRange') is None
    assert PT.get_child_from_name(container, 'NormalZ') is not None
  
  # > U
  expt_loc = 'FaceCenter' if cell_dim == 3 else 'CellCenter'
  for cnt in ['Poly', 'Standard']:
    tree = PT.deep_copy(base_tree)
    maia.algo.dist.convert_s_to_u(tree, cnt, comm)
    if cell_dim == 2:
      maia.algo.edge_pe_to_ngon(tree, comm) # Needed for partitioning

    tree = maia.factory.partition_dist_tree(tree, comm)
    zone = PT.get_all_Zone_t(tree)[0]
    GEO.compute_elements_normal(zone, unitary=True)

    container = PT.find_node_from_name(zone, 'Geometry_2d')
    assert PT.Subset.GridLocation(container) == expt_loc
    assert (PT.get_child_from_name(container, 'PointList') is not None) == (cell_dim == 3)
    assert PT.get_child_from_name(container, 'UnitNormalZ') is not None

@pytest.mark.parametrize('cell_dim', [1,2])
@pytest_parallel.mark.parallel(1)
def test_compute_elements_normal_edge_placement(cell_dim, comm):
  # NB : in this test we just check the placement, not the values
  
  # > S 
  n_vtx = [5,5] if cell_dim == 2 else [5]
  tree = maia.factory.generate_dist_block(n_vtx, 'S', comm)
  tree = maia.factory.partition_dist_tree(tree, comm)
  PT.rm_nodes_from_name(tree, 'CoordinateZ')
  zone = PT.get_all_Zone_t(tree)[0]
  GEO.compute_elements_normal(zone, unitary=True)

  if cell_dim == 2:
    assert PT.get_node_from_name(zone, 'Geometry_1d') is None
    for dir in ['I', 'J']:
      container = PT.find_node_from_name(zone, f'Geometry_1d_{dir}')
      assert PT.Subset.GridLocation(container) == f'{dir}EdgeCenter'
      #assert PT.get_child_from_name(container, 'PointRange') is not None
      assert PT.get_child_from_name(container, 'UnitNormalY') is not None
      assert PT.get_child_from_name(container, 'UnitNormalZ') is None
  else:
    container = PT.find_node_from_name(zone, f'Geometry_1d')
    assert PT.Subset.GridLocation(container) == 'CellCenter'
    assert PT.get_child_from_name(container, 'PointRange') is None
    assert PT.get_child_from_name(container, 'UnitNormalY') is not None
    assert PT.get_child_from_name(container, 'UnitNormalZ') is None

  # > U
  expt_loc = 'EdgeCenter' if cell_dim == 2 else 'CellCenter'
  # > Std elements
  for cnt in ['Poly', 'Standard']:
    if cnt == 'Poly' and cell_dim == 1:
      continue #This case does not exists

    elt_kind = 'QUAD_4' if cell_dim == 2 else 'BAR_2'
    tree = maia.factory.generate_dist_block(5, elt_kind, comm)
    if cnt == 'Poly':
      maia.algo.dist.convert_elements_to_ngon(tree, comm)
    tree = maia.factory.partition_dist_tree(tree, comm)
    PT.rm_nodes_from_name(tree, 'CoordinateZ')

    zone = PT.get_all_Zone_t(tree)[0]
    maia.algo.compute_elements_normal(tree)

    container = PT.find_node_from_name(zone, 'Geometry_1d')
    assert PT.Subset.GridLocation(container) == expt_loc
    assert (PT.get_child_from_name(container, 'PointList') is not None) == (cell_dim == 2)
    assert PT.get_child_from_name(container, 'NormalY') is not None
    assert PT.get_child_from_name(container, 'NormalZ') is None