import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT

from maia.algo.part.geometry import centers

def to_expected_cyl(expected_cart):
  x = expected_cart[0::3]
  y = expected_cart[1::3]
  z = expected_cart[2::3]
  expected_cyl = np.empty_like(expected_cart)
  expected_cyl[0::3] = np.sqrt(x**2 + y**2) #R
  expected_cyl[1::3] = np.arctan2(y,x)      #O
  expected_cyl[2::3] = z                    #Z
  return expected_cyl


def as_partitioned(tree):
  #On partitions, element are supposed to be I4
  for zone in PT.get_all_Zone_t(tree):
    for elt_node in PT.iter_children_from_label(zone, 'Elements_t'):
      for name in ['ElementRange', 'ElementConnectivity', 'ParentElements', 'ElementStartOffset']:
        node = PT.get_child_from_name(elt_node, name)
        if node is not None:
          node[1] = node[1].astype(np.int32)
    for pl_node in PT.iter_nodes_from_name(zone, 'PointList'):
      pl_node[1] = pl_node[1].astype(np.int32)
  PT.rm_nodes_from_name(tree, ':CGNS#Distribution')

#region Cell center ------------------------------------------------------------

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("elt_kind", ['NFACE_n', 'HEXA_8', 'S'])
@pytest.mark.parametrize("cylindric", [False, True])
def test_compute_cell_center_u(elt_kind, cylindric, comm):
  #Test U
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  as_partitioned(tree)
  zone = PT.get_all_Zone_t(tree)[0]

  if elt_kind == 'S':
    for coord in PT.iter_children_from_predicates(zone, 'GridCoordinates_t/DataArray_t'):
      coord[1] = coord[1].reshape((3,3,3), order='F')

  if cylindric:
    maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))

  cell_center = centers.compute_cell_center(zone)

  if cylindric:
    expected = np.array([0.35355339, 0.78539816, 0.25, 
                        0.79056942, 0.32175055, 0.25, 
                        0.79056942, 1.24904577, 0.25, 
                        1.06066017, 0.78539816, 0.25, 
                        0.35355339, 0.78539816, 0.75, 
                        0.79056942, 0.32175055, 0.75, 
                        0.79056942, 1.24904577, 0.75, 
                        1.06066017, 0.78539816, 0.75])
  else:
    expected = np.array([0.25, 0.25, 0.25, 
                         0.75, 0.25, 0.25, 
                         0.25, 0.75, 0.25, 
                         0.75, 0.75, 0.25, 
                         0.25, 0.25, 0.75, 
                         0.75, 0.25, 0.75, 
                         0.25, 0.75, 0.75, 
                         0.75, 0.75, 0.75])
 
  assert np.allclose(cell_center, expected)

@pytest_parallel.mark.parallel(1)
class Test_compute_cell_center_filtered:

  def run_test(self, cell_indices_l, expected_l):
    zone = PT.get_all_Zone_t(self.tree)[0]
    for cell_indices, expec in zip(cell_indices_l, expected_l):
      cell_indices = np.array(cell_indices)
      cell_center = centers.compute_cell_center(zone, cell_indices)
      assert np.allclose(cell_center,np.array(expec))

  def test_ngon(self, comm):
    self.tree = maia.factory.generate_dist_block(3, 'Poly', comm)
    as_partitioned(self.tree)

    cell_indices_l = ([[44,37]], [[38,37]], [[]])
    expected_l     = ([0.75,0.75,0.75,0.25,0.25,0.25], [0.75,0.25,0.25,0.25,0.25,0.25], [])
    self.run_test(cell_indices_l, expected_l)

  def test_elts(self, comm):
    self.tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
    as_partitioned(self.tree)

    cell_indices_l = ([[8,1]], [[2,1]], [[]])
    expected_l     = ([0.75,0.75,0.75,0.25,0.25,0.25], [0.75,0.25,0.25,0.25,0.25,0.25], [])
    self.run_test(cell_indices_l, expected_l)

  def test_S(self, comm):
    dtree = maia.factory.generate_dist_block(3, 'S', comm)
    self.tree = maia.factory.partition_dist_tree(dtree, comm)

    cell_indices_l = ([[2,1],[2,1],[2,1]], [[2,1],[1,1],[1,1]], [[],[],[]])
    expected_l     = ([0.75,0.75,0.75,0.25,0.25,0.25], [0.75,0.25,0.25,0.25,0.25,0.25], [])
    self.run_test(cell_indices_l, expected_l)
    del(self.tree)

#endregion Cell center ---------------------------------------------------------

#region Face center ------------------------------------------------------------
@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("cylindric", [False, True])
@pytest.mark.parametrize("filtering", [False, True])
def test_compute_face_center_3d_u_ngon(comm, cylindric, filtering):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindric:
    maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))

  expected = np.array([
      0.25,0.25,0.  ,   0.75,0.25,0.  ,   0.25,0.75,0.  ,   0.75,0.75,0.  ,
      0.25,0.25,0.5 ,   0.75,0.25,0.5 ,   0.25,0.75,0.5 ,   0.75,0.75,0.5 ,
      0.25,0.25,1.  ,   0.75,0.25,1.  ,   0.25,0.75,1.  ,   0.75,0.75,1.  ,
      0.  ,0.25,0.25,   0.  ,0.75,0.25,   0.  ,0.25,0.75,   0.  ,0.75,0.75,
      0.5 ,0.25,0.25,   0.5 ,0.75,0.25,   0.5 ,0.25,0.75,   0.5 ,0.75,0.75,
      1.  ,0.25,0.25,   1.  ,0.75,0.25,   1.  ,0.25,0.75,   1.  ,.75 ,.75 ,
      0.25,0.  ,0.25,   0.25,0.  ,0.75,   0.75,0.  ,0.25,   0.75,0.  ,0.75,
      0.25,0.5 ,0.25,   0.25,0.5 ,0.75,   0.75,0.5 ,0.25,   0.75,0.5 ,0.75,
      0.25,1.  ,0.25,   0.25,1.  ,0.75,   0.75,1.  ,0.25,   0.75,1.  ,0.75,])
  
  if cylindric:
    expected = to_expected_cyl(expected)

  if filtering:
    face_ind_l = ([[1]], [[1,2,3]], [[1,3,5,7,9,11]], [[11,12,14]])
    for face_ind in face_ind_l:
      face_ind = np.array(face_ind)
      out = centers.compute_face_center(zone, face_ind)
      for d in range(3):
        assert np.allclose(out[d::3], expected[d::3][face_ind[0]-1])

  else:
    out = centers.compute_face_center(zone)
    assert np.allclose(out, expected)



@pytest.mark.skipif(not maia.pdm_has_ptscotch, reason="Require PTScotch")
@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("filtering", [False, True])
def test_compute_face_center_3d_u_elts(comm, filtering):
  tree = maia.factory.generate_dist_block(2, 'HEXA_8', comm)
  maia.algo.dist.reorder_elt_sections_from_dim(tree) # Put face first, for easier face idx selection
  
  part_tree = maia.factory.partition_dist_tree(tree, comm, graph_part_tool='ptscotch')
  # Nb : metis is not robust if n_cell == 1
  zone = PT.get_all_Zone_t(part_tree)[0]

  expected = np.array([0.5,0.5,0., 0.5,0.5,1., 0.,0.5,0.5,
                       1.,0.5,0.5, 0.5,0.,0.5, 0.5,1.,0.5])

  if filtering:
    face_ind_l = ([[1]], [[1,2,3]], [[5,4,3]], [[5,2,1]])
    for face_ind in face_ind_l:
      face_ind = np.array(face_ind)
      out = centers.compute_face_center(zone, face_ind)
      for d in range(3):
        assert np.allclose(out[d::3], expected[d::3][face_ind[0]-1])
  else:
    assert np.allclose(centers.compute_face_center(zone), expected)


@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("cylindric", [False, True])
@pytest.mark.parametrize("filtering", [False, True])
def test_compute_face_center_3d_s(comm, cylindric, filtering):
  tree = maia.factory.generate_dist_block(3, 'Structured', comm)
  # Reput coords as partitioned tree
  tree = maia.factory.partition_dist_tree(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindric:
    maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))

  expected = np.array([
     0.  ,0.25,0.25, 0.5 ,0.25,0.25, 1.  ,0.25,0.25,
     0.  ,0.75,0.25, 0.5 ,0.75,0.25, 1.  ,0.75,0.25, 
     0.  ,0.25,0.75, 0.5 ,0.25,0.75, 1.  ,0.25,0.75,
     0.  ,0.75,0.75, 0.5 ,0.75,0.75, 1.  ,0.75,0.75, # End of IFaces
     0.25,0.  ,0.25, 0.75,0.  ,0.25, 0.25,0.5 ,0.25,
     0.75,0.5 ,0.25, 0.25,1.  ,0.25, 0.75,1.  ,0.25, 
     0.25,0.  ,0.75, 0.75,0.  ,0.75, 0.25,0.5 ,0.75,
     0.75,0.5 ,0.75, 0.25,1.  ,0.75, 0.75,1.  ,0.75, # End of JFaces
     0.25,0.25,0.  , 0.75,0.25,0.  , 0.25,0.75,0.  ,
     0.75,0.75,0.  , 0.25,0.25,0.5 , 0.75,0.25,0.5 ,
     0.25,0.75,0.5 , 0.75,0.75,0.5 , 0.25,0.25,1.  ,
     0.75,0.25,1.  , 0.25,0.75,1.  , 0.75,0.75,1.  ]) # End of KFaces

  if cylindric:
    expected = to_expected_cyl(expected)

  if filtering:
    face_ind_l = ([[1,2,3,1,3], [1,1,1,1,2], [1,1,1,2,2]], [[2,2,2,2],[1,3,2,1],[2,1,1,1]], [[], [], []]) # PL of indices to compute
    face_dir_l = ('I', 'J', 'I')                                                                          # FaceDirection to compute
    raw_ind_l  = ([[1,2,3,7,12]], [[20,18,16,14]], [[]])                                                  # Corresponding glob idx for check
    for face_ind, dir, raw_ind in zip(face_ind_l, face_dir_l, raw_ind_l):
      face_ind = np.array(face_ind)
      out = centers.compute_face_center(zone, face_ind, f"{dir}FaceCenter")
      if face_ind.size == 0:
        assert len(out) == 0
      else:
        for d in range(3):
          assert np.allclose(out[d::3], expected[d::3][np.array(raw_ind)-1])

    else:
      out = centers.compute_face_center(zone)
      assert np.allclose(out, expected)

  # 2D cases

@pytest.mark.parametrize("phydim", [3,2])
@pytest.mark.parametrize("filtering", [False, True])
@pytest_parallel.mark.parallel(1)
def test_compute_face_center_2d_elt(phydim, filtering, comm):
  dslice_tree = maia.factory.generate_dist_block(4, "QUAD_4", comm)
  pslice_tree = maia.factory.partition_dist_tree(dslice_tree, comm)
  zone = PT.get_all_Zone_t(pslice_tree)[0]

  if phydim == 2:
    PT.rm_nodes_from_name(zone, 'CoordinateZ')

  expected = np.array([1/6, 1/6, 0.,   0.5, 1/6, 0.,   5/6, 1/6, 0.,
                       1/6, 0.5, 0.,   0.5, 0.5, 0.,   5/6, 0.5, 0.,
                       1/6, 5/6, 0.,   0.5, 5/6, 0.,   5/6, 5/6, 0.])
  
  face_ind_l = ([[1]], [[1,2,3]], [[1,3,5,7]], [[6,7,5]])
  if filtering:
    for face_ind in face_ind_l:
      face_ind = np.array(face_ind)
      out = centers.compute_face_center(zone,face_ind)
      for d in range(3):
        assert np.allclose(out[d::3], expected[d::3][face_ind[0]-1])
  else:
    assert np.allclose(centers.compute_face_center(zone), expected)


@pytest.mark.parametrize("phydim", [3,2])
@pytest.mark.parametrize("filtering", [False, True])
def test_compute_face_center_2d_s(phydim, filtering, comm):
  if phydim == 3:
    tree = maia.factory.generate_dist_block([3,3,1], 'Structured', comm)
  elif phydim == 2:
    tree = maia.factory.generate_dist_block([3,3], 'Structured', comm, origin=[0., 0.])
  zone = PT.get_all_Zone_t(tree)[0]

  # As partitioned
  for dir in ['X', 'Y']:
    node = PT.get_node_from_name(zone, f'Coordinate{dir}')
    node[1] = node[1].reshape((3,3), order='F')
  
  if phydim == 3:
    # Change Z for test
    node = PT.get_node_from_name(zone, 'CoordinateZ')
    node[1] = np.array([[0,0.1,0], [0,0,0], [0,0.2,0.2]], order='F')
    expected = np.array([0.25,0.25,0.025  , 0.75 ,0.25,0.05,  0.25, 0.75, 0.025,   0.75, 0.75, 0.1])
  elif phydim == 2:
    expected = np.array([0.25,0.25,0,  0.75 ,0.25,0,  0.25, 0.75,0,  0.75, 0.75,0])

  if filtering:
    face_ind_l = ([[1],[1]], [[1,2,2],[1,1,2]], [[1,1,2],[2,1,1]], [[],[]]) # PL of indices to compute
    raw_ind_l  = ([1], [1,2,4], [3,1,2])                                    # Corresponding glob idx for check
    for face_ind, raw_ind in zip(face_ind_l, raw_ind_l):
      face_ind = np.array(face_ind)
      out = centers.compute_face_center(zone, face_ind)
      if face_ind.size == 0:
        assert len(out) == 0
      else:
        for d in range(3):
          assert np.array_equal(out[d::3], expected[d::3][np.array(raw_ind)-1])
  else:
    assert np.array_equal(centers.compute_face_center(zone), expected)

def test_compute_face_center_2d_s_cyl(comm):
  # Without CZ, in cyl coords (move by hand, fct does not manage z == 0 ...)
  tree = maia.factory.generate_dist_block([3,3], 'Structured', comm, origin=[0., 0.])
  cx = PT.get_node_from_name(tree, 'CoordinateX')
  cy = PT.get_node_from_name(tree, 'CoordinateY')
  r     = np.sqrt(cx[1]**2  + cy[1]**2)
  theta = np.arctan2(cy[1], cx[1])
  cx[1] = r.reshape((3,3), order='F')
  cy[1] = theta.reshape((3,3), order='F')
  cx[0] = 'CoordinateR'
  cy[0] = 'CoordinateTheta'
  zone = PT.get_all_Zone_t(tree)[0]
  expected = np.array([0.35355339,0.78539816,0,  0.79056942,0.32175055,0,  0.79056942,1.24904577,0,  1.06066017,0.785398160,0])
  assert np.allclose(centers.compute_face_center(zone), expected, atol=1e-6)

#endregion Face center ---------------------------------------------------------

#region Edge center ------------------------------------------------------------

@pytest.mark.parametrize("cyl", [False, True])
@pytest.mark.parametrize("phydim", [3,2])
@pytest.mark.parametrize("filtering", [False, True])
@pytest_parallel.mark.parallel(1)
def test_compute_edge_center_2d_elt(cyl, phydim, filtering, comm):
  tree = maia.factory.generate_dist_block(3, 'QUAD_4', comm)
  as_partitioned(tree)
  zone = PT.get_all_Zone_t(tree)[0]

  if cyl:
    maia.algo.cartesian_to_cylindrical(tree, axis=(0,0,1))
  if phydim == 2:
    PT.rm_nodes_from_name(zone, 'CoordinateZ')

  if cyl:
    expected = np.array([0.25,0.,0.,         0.75,0.,0.,        1.030776,1.325818,0.,  1.25,0.927295,0.,
                          0.25, 1.570796,0.,  0.75,1.570796,0.,  1.030776,0.244979,0.,  1.25,0.643501,0.])
  else:
    expected = np.array([0.25,0.,0., 0.75,0.,0., 0.25,1.,0., 0.75,1.,0.,
                          0.,0.25,0., 0.,0.75,0., 1.,0.25,0., 1.,0.75,0.])

  if filtering:
    edge_indices_l = ([[5]], [[7,6,5]], [[]])
    expected_l     = ([0.25,0.,0.], [0.25,1.,0., 0.75,0.,0., 0.25,0.,0.], [])
    if cyl:
      expected_l = [to_expected_cyl(np.array(t)) for t in expected_l]
  
    for edge_indices, expec in zip(edge_indices_l, expected_l):
      edge_indices = np.array(edge_indices)
      edge_center = centers.compute_edge_center(zone, edge_indices)
      assert np.allclose(edge_center, np.array(expec))
  else:
    assert np.allclose(centers.compute_edge_center(zone), expected, atol=1e-4)

def test_compute_edge_center_u_ngon_3d(comm):
  tree = maia.factory.generate_dist_block(3, 'NFACE_n', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  with pytest.raises(NotImplementedError):
    centers.compute_edge_center(zone)

def test_compute_edge_center_s(comm):
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  with pytest.raises(NotImplementedError):
    centers.compute_edge_center(zone)

#endregion Edge center ---------------------------------------------------------