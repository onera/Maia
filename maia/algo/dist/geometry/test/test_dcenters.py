import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia

from maia.utils import pr_utils

from maia.algo.dist.geometry import centers as GEO

def to_expected_cyl(expected_cart):
  x = expected_cart[0::3]
  y = expected_cart[1::3]
  z = expected_cart[2::3]
  expected_cyl = np.empty_like(expected_cart)
  expected_cyl[0::3] = np.sqrt(x**2 + y**2) #R
  expected_cyl[1::3] = np.arctan2(y,x)      #O
  expected_cyl[2::3] = z                    #Z
  return expected_cyl

#region edge_center ------------------------------------------------------------
@pytest_parallel.mark.parallel(3)
def test_compute_edge_center_3d_ngon(comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  with pytest.raises(NotImplementedError):
    GEO.compute_edge_center(zone, comm)

@pytest_parallel.mark.parallel(3)
def test_compute_edge_center_3d_s(comm):
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  zone = PT.get_all_Zone_t(tree)[0]
  with pytest.raises(NotImplementedError):
    GEO.compute_edge_center(zone, comm)

@pytest_parallel.mark.parallel(3)
def test_compute_edge_center_3d_elts(comm):
  tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  edge_center = GEO.compute_edge_center(zone, comm)
  assert edge_center.size == 0 # No edges in this mesh (only cell + faces) --> return []

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False,True])
class Test_compute_edge_center_2d_u_elts:

  def setup_zone(self, cylindrical, comm):
    tree = maia.factory.generate_dist_block(5, 'TRI_3', comm)
    if cylindrical:
      maia.algo.cartesian_to_cylindrical(tree, (0,0,1))
    zone = PT.get_all_Zone_t(tree)[0]
    return zone

  def test_basic(self, cylindrical,comm):
    zone = self.setup_zone(cylindrical, comm)
    edge_center = GEO.compute_edge_center(zone, comm)

    expec = [
      [0.125, 0.   , 0.   ,   0.375, 0.   , 0.   , 
      0.625, 0.   , 0.   ,   0.875, 0.   , 0.   , 
      0.125, 1.   , 0.   ,   0.375, 1.   , 0.   ],
      [0.625, 1.   , 0.   ,   0.875, 1.   , 0.   ,
      0.   , 0.125, 0.   ,   0.   , 0.375, 0.   ,
      0.   , 0.625, 0.   ],
      [0.   , 0.875, 0.   ,   1.   , 0.125, 0.   ,
      1.   , 0.375, 0.   ,   1.   , 0.625, 0.   ,
      1.   , 0.875, 0.   ],
    ]
    expected_edge_center = np.array(expec[comm.rank])

    if cylindrical:
      expected_edge_center = to_expected_cyl(expected_edge_center)

    assert np.allclose(edge_center, expected_edge_center)

  def test_filter(self, cylindrical, comm):
    zone = self.setup_zone(cylindrical, comm)

    bc_path_l = ("ZoneBC/Xmin/PointList", "ZoneBC/Xmax/PointList", "ZoneBC/Ymin/PointList", "ZoneBC/Ymax/PointList")
    edge_indices_l = [PT.get_node_from_path(zone, bc_path)[1] for bc_path in bc_path_l]

    if comm.rank == 0:
      expected_l = ([0.   , 0.125, 0., 0.   , 0.375, 0.], [1.   , 0.125, 0., 1.   , 0.375, 0.],
                    [0.125, 0.   , 0., 0.375, 0.   , 0.], [0.125, 1.   , 0., 0.375, 1.   , 0.])
    elif comm.rank == 1:
      expected_l = ([0., 0.625, 0.], [1., 0.625, 0.], [0.625, 0., 0.], [0.625, 1., 0.])
    elif comm.rank == 2:
      expected_l = ([0., 0.875, 0.], [1., 0.875, 0.], [0.875, 0., 0.], [0.875, 1., 0.])

    for edge_indices, expec in zip(edge_indices_l, expected_l):

      edge_indices = np.array(edge_indices)
      edge_center = GEO.compute_edge_center(zone, comm, edge_indices=edge_indices)

      expected_edge_center = np.array(expec)

      if cylindrical:
        expected_edge_center = to_expected_cyl(expected_edge_center)

      assert np.allclose(edge_center, expected_edge_center)
#endregion edge_center ---------------------------------------------------------

#region face_center ------------------------------------------------------------

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("elt_kind", ['S', 'Poly', 'HEXA_8'])
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center3d(cylindrical, elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  if elt_kind == 'Poly':
    expected = [
        [0.25, 0.25, 0. ,  0.75, 0.25, 0. ,  0.25, 0.75, 0. ,  0.75, 0.75, 0. , 
          0.25, 0.25, 0.5,  0.75, 0.25, 0.5,  0.25, 0.75, 0.5,  0.75, 0.75, 0.5, 
          0.25, 0.25, 1. ,  0.75, 0.25, 1. ,  0.25, 0.75, 1. ,  0.75, 0.75, 1. ],
        [0. , 0.25, 0.25,  0. , 0.75, 0.25,  0.,  0.25, 0.75,  0. , 0.75, 0.75,
          0.5, 0.25, 0.25,  0.5, 0.75, 0.25,  0.5, 0.25, 0.75,  0.5, 0.75, 0.75,
          1. , 0.25, 0.25,  1. , 0.75, 0.25,  1.,  0.25, 0.75,  1. , 0.75, 0.75],
        [0.25, 0. , 0.25,  0.25, 0. , 0.75,  0.75, 0. , 0.25,  0.75, 0. , 0.75, 
          0.25, 0.5, 0.25,  0.25, 0.5, 0.75,  0.75, 0.5, 0.25,  0.75, 0.5, 0.75,
          0.25, 1. , 0.25,  0.25, 1. , 0.75,  0.75, 1. , 0.25,  0.75, 1. , 0.75]
    ][comm.rank]
  elif elt_kind == 'HEXA_8':
    expected = [
      [0.25, 0.25, 0.  ,   0.75, 0.25, 0.  ,   0.25, 0.75, 0.  ,   0.75, 0.75, 0.  ,
      0.25, 0.25, 1.  ,   0.75, 0.25, 1.  ,   0.25, 0.75, 1.  ,   0.75, 0.75, 1.  ],
      [0.  , 0.25, 0.25,   0.  , 0.75, 0.25,   0.  , 0.25, 0.75,   0.  , 0.75, 0.75,
      1.  , 0.25, 0.25,   1.  , 0.75, 0.25,   1.  , 0.25, 0.75,   1.  , 0.75, 0.75],
      [0.25, 0.  , 0.25,   0.25, 0.  , 0.75,   0.75, 0.  , 0.25,   0.75, 0.  , 0.75,
      0.25, 1.  , 0.25,   0.25, 1.  , 0.75,   0.75, 1.  , 0.25,   0.75, 1.  , 0.75],
    ][comm.rank]
  elif elt_kind == 'S':
    expected = [
      [0.  , 0.25, 0.25,   0.5 , 0.25, 0.25,   1.  , 0.25, 0.25,   0.  , 0.75, 0.25,
      0.5 , 0.75, 0.25,   1.  , 0.75, 0.25,   0.  , 0.25, 0.75,   0.5 , 0.25, 0.75,
      1.  , 0.25, 0.75,   0.  , 0.75, 0.75,   0.5 , 0.75, 0.75,   1.  , 0.75, 0.75],
      [0.25, 0.  , 0.25,   0.75, 0.  , 0.25,   0.25, 0.5 , 0.25,   0.75, 0.5 , 0.25,
      0.25, 1.  , 0.25,   0.75, 1.  , 0.25,   0.25, 0.  , 0.75,   0.75, 0.  , 0.75,
      0.25, 0.5 , 0.75,   0.75, 0.5 , 0.75,   0.25, 1.  , 0.75,   0.75, 1.  , 0.75],
      [0.25, 0.25, 0.  ,   0.75, 0.25, 0.  ,   0.25, 0.75, 0.  ,   0.75, 0.75, 0.  ,
      0.25, 0.25, 0.5 ,   0.75, 0.25, 0.5 ,   0.25, 0.75, 0.5 ,   0.75, 0.75, 0.5 ,
      0.25, 0.25, 1.  ,   0.75, 0.25, 1.  ,   0.25, 0.75, 1.  ,   0.75, 0.75, 1.  ]
    ][comm.rank]

  expected_face_center = np.array(expected)
  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  face_center = GEO.compute_face_center(zone, comm)
  assert np.allclose(face_center, expected_face_center)


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False,True])
def test_compute_face_center3d_u_ngon_filtered(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  bc_path_l = ("ZoneBC/Ymin/PointList", "ZoneBC/Xmax/PointList", "ZoneBC/Zmin/PointList")
  face_ind_l = [PT.get_value(PT.get_node_from_path(zone,bc_path)) for bc_path in bc_path_l]
  if comm.rank == 0:
    expected_l = ([], [], [0.25, 0.25, 0., 0.75, 0.25, 0., 0.25, 0.75, 0., 0.75, 0.75, 0.])
  elif comm.rank == 1:
    expected_l = ([], [1., 0.25, 0.25, 1., 0.75, 0.25, 1., 0.25, 0.75, 1., 0.75, 0.75], [])
  elif comm.rank == 2:
    expected_l = ([0.25, 0., 0.25, 0.25, 0., 0.75, 0.75, 0., 0.25, 0.75, 0., 0.75], [], [])

  for face_ind, expec in zip(face_ind_l, expected_l):
    face_center = GEO.compute_face_center(zone, comm, face_ind)

    expected_face_center = np.array(expec)
    if cylindrical:
      expected_face_center = to_expected_cyl(expected_face_center)

    assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False,True])
def test_compute_face_center3d_u_elts_filtered(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  bc_path_l = ("ZoneBC/Xmin/PointList", "ZoneBC/Ymax/PointList", "ZoneBC/Zmin/PointList")
  face_ind_l = [PT.get_value(PT.get_node_from_path(zone,bc_path)) for bc_path in bc_path_l]
  if comm.rank == 0:
    expected_l = ([0.,0.25,0.25,0.,0.75,0.25], [0.25,1.,0.25,0.25,1.,0.75], [0.25,0.25,0.,0.75,0.25,0.])
  elif comm.rank == 1:
    expected_l = ([0.,0.25,0.75], [0.75,1.,0.25], [0.25,0.75,0.])
  elif comm.rank == 2:
    expected_l = ([0.,0.75,0.75], [0.75,1.,0.75], [0.75,0.75,0.])

  for face_ind, expec in zip(face_ind_l, expected_l):
    face_center = GEO.compute_face_center(zone, comm, face_ind)

    expected_face_center = np.array(expec)
    if cylindrical:
      expected_face_center = to_expected_cyl(expected_face_center)

    assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(1)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center3d_s_filtered(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  bc_path_l = ("ZoneBC/Xmin/PointRange", "ZoneBC/Xmax/PointRange", "ZoneBC/Ymin/PointRange", "ZoneBC/Zmax/PointRange")
  bc_dir    = ('I', 'I', 'J', 'K')
  face_ind_l = list()
  for bc_path, dir in zip(bc_path_l, bc_dir):
    # Recompute face pointlist on the fly
    face_pr = PT.get_value(PT.get_node_from_path(zone,bc_path))
    face_pr = np.array([f if di==dir else np.clip(f,0,2) for f,di in zip(face_pr, 'IJK')])
    face_ind_l.append(pr_utils.unroll_pr(face_pr))

  expected_l = [
    [0.  , 0.25, 0.25, 0.  , 0.75, 0.25, 0.  , 0.25, 0.75, 0.  , 0.75, 0.75],
    [1.  , 0.25, 0.25, 1.  , 0.75, 0.25, 1.  , 0.25, 0.75, 1.  , 0.75, 0.75], 
    [0.25, 0.  , 0.25, 0.75, 0.  , 0.25, 0.25, 0.  , 0.75, 0.75, 0.  , 0.75],
    [0.25, 0.25, 1.  , 0.75, 0.25, 1.  , 0.25, 0.75, 1.  , 0.75, 0.75, 1.  ]]  
  
  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  for face_ind, dir, expec in zip(face_ind_l, bc_dir, expected_l):
    face_center = GEO.compute_face_center(zone, comm, face_ind, f'{dir}FaceCenter')

    expected_face_center = np.array(expec)
    if cylindrical:
      expected_face_center = to_expected_cyl(expected_face_center)

    assert np.allclose(face_center, expected_face_center)


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("filtering", [False, True])
@pytest.mark.parametrize("elt_kind", ['TRI_3', 'NGON_n'])
def test_compute_face_center2d_u(cylindrical, filtering, elt_kind, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  if elt_kind == 'NGON_n':
    maia.algo.dist.convert_elements_to_ngon(tree, comm)
  maia.algo.dist.reorder_elt_sections_from_dim(tree, reverse=True)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  if filtering:
    if comm.rank == 0:
      face_indices_l = ([[1]], [[7,8]], [[]], [[4,6]], [[]])
      expected_l = ([1/6,1/6,0.], [4/6,4/6,0.,5/6,5/6,0], [], [5/6,2/6,0,2/6,5/6,0], [])
    elif comm.rank == 1:
      face_indices_l = ([[3]], [[1,2]], [[4,6]], [[]], [[]])
      expected_l = ([4/6,1/6,0], [1/6,1/6,0.,2/6,2/6,0.], [5/6,2/6,0,2/6,5/6,0], [], [])
    for face_indices, expected in zip(face_indices_l, expected_l):
      _face_indices = np.array(face_indices, dtype=int)
      
      face_center = GEO.compute_face_center(zone, comm, _face_indices)

      expected_face_center = np.array(expected)
      if cylindrical:
        expected_face_center = to_expected_cyl(expected_face_center)

      assert np.allclose(face_center, expected_face_center)

  else:

    face_center = GEO.compute_face_center(zone, comm)

    if cylindrical:
      if comm.Get_rank() == 0:
        assert np.allclose(face_center, np.array([0.23570226,0.78539816,0,  0.47140452,0.78539816,0,  0.68718427,0.24497866,0,  0.89752747,0.38050638,0]))
      if comm.Get_rank() == 1:
        assert np.allclose(face_center, np.array([0.68718427,1.32581766,0,  0.89752747,1.19028995,0,  0.94280904,0.78539816,0,  1.17851130,0.78539816,0]))
    else:
      if comm.Get_rank() == 0:
        assert (face_center == np.array([1.,1,0, 2,2,0, 4,1,0, 5,2,0]) / 6.).all()
      if comm.Get_rank() == 1:
        assert (face_center == np.array([1.,4,0, 2,5,0, 4,4,0, 5,5,0]) / 6.).all()


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("filtering", [False, True])
def test_compute_face_center2d_s(cylindrical, filtering, comm):
  tree = maia.factory.generate_dist_block([3,3,1], 'Structured', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  if filtering:
    if comm.rank == 0:
      face_point_list = np.array([[1,2,1,2],[1,1,2,2]])
      expected = np.array([0.25, 0.25, 0.  , 0.75, 0.25, 0.  , 0.25, 0.75, 0.  , 0.75, 0.75, 0.  ])
    elif comm.rank == 1:
      face_point_list = np.array([[2],[1]])
      expected = np.array([0.75, 0.25, 0.  ])
  else:
    face_point_list = None
    if comm.rank == 0:
      expected = np.array([0.25, 0.25, 0.  , 0.75, 0.25, 0.  ])
    elif comm.rank == 1:
      expected = np.array([0.25, 0.75, 0.  , 0.75, 0.75, 0.  ])

  face_center = GEO.compute_face_center(zone, comm, face_point_list)
    
  if cylindrical:
    expected = to_expected_cyl(expected)
  assert np.allclose(face_center, expected)

#endregion face_center ---------------------------------------------------------

#region cell_center ------------------------------------------------------------
@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ["S", "Poly", "HEXA_8"])
@pytest.mark.parametrize("cylindrical", [False, True])
class Test_compute_cell_center:

  def setup_tree(self, cylindrical, elt_kind, comm):
    tree = maia.factory.generate_dist_block(3, elt_kind, comm)
    if elt_kind == 'Poly':
      maia.algo.pe_to_nface(tree, comm)
    zone = PT.get_all_Zone_t(tree)[0]
    if cylindrical:
      maia.algo.cartesian_to_cylindrical(tree, (0,0,1))
    return zone

  def test_simple(self, elt_kind, cylindrical, comm):
    zone = self.setup_tree(cylindrical, elt_kind, comm)
    
    cell_center = GEO.compute_cell_center(zone, comm)
    
    if cylindrical:
        from math import pi
        expt_cell_center = [
          np.array([0.35355339, pi/4, 0.25,  0.79056942,0.32175055,0.25,  0.79056942,1.24904577,0.25,  1.06066017,pi/4,0.25]),
          np.array([0.35355339, pi/4, 0.75,  0.79056942,0.32175055,0.75,  0.79056942,1.24904577,0.75,  1.06066017,pi/4,0.75])
        ][comm.Get_rank()]
    else:
        expt_cell_center = [
          np.array([0.25,0.25,0.25, 0.75,0.25,0.25, 0.25,0.75,0.25, 0.75,0.75,0.25]),
          np.array([0.25,0.25,0.75, 0.75,0.25,0.75, 0.25,0.75,0.75, 0.75,0.75,0.75])
        ][comm.Get_rank()]

    assert np.allclose(expt_cell_center, cell_center)

  def test_filter(self, elt_kind, cylindrical, comm):
    zone = self.setup_tree(cylindrical, elt_kind, comm)
    
    if comm.rank == 0:
      if elt_kind == 'Poly':
        cell_indices_l = ([[37]], [[37,38,41,42]], [[37,38]], [[]], [[]])
      elif elt_kind == 'HEXA_8':
        cell_indices_l = ([[1]], [[1,2,5,6]], [[1,2]], [[]], [[]])
      elif elt_kind == 'S':
        cell_indices_l = ([[1],[1],[1]], [[1,2,1,2],[1,1,1,1],[1,1,2,2]], [[1,2],[1,1],[1,1]], [[],[],[]], [[],[],[]])
      expected_l = ([0.25,0.25,0.25], [0.25,0.25,0.25, 0.75,0.25,0.25, 0.25,0.25,0.75, 0.75,0.25,0.75], [0.25,0.25,0.25, 0.75,0.25,0.25], [], [])
    elif comm.rank == 1:
      if elt_kind == 'Poly':
        cell_indices_l = ( [[38]], [[39,40,43,44]], [[]], [[37,38]], [[]])
      elif elt_kind == 'HEXA_8':
        cell_indices_l = ([[2]], [[3,4,7,8]], [[]], [[1,2]], [[]])
      elif elt_kind == 'S':
        cell_indices_l = ([[2],[1],[1]], [[1,2,1,2],[2,2,2,2],[1,1,2,2]], [[],[],[]], [[1,2],[1,1],[1,1]], [[],[],[]])
      expected_l = ([0.75,0.25,0.25], [0.25,0.75,0.25, 0.75,0.75,0.25, 0.25,0.75,0.75, 0.75,0.75,0.75], [], [0.25,0.25,0.25, 0.75,0.25,0.25], [])

    for cell_indices, expected in zip(cell_indices_l, expected_l):
      _cell_indices = np.array(cell_indices, dtype=int)
      
      cell_center = GEO.compute_cell_center(zone, comm, _cell_indices)

      expected_cell_center = np.array(expected)
      if cylindrical:
        expected_cell_center = to_expected_cyl(expected_cell_center)

      assert np.allclose(cell_center,expected_cell_center)
#endregion cell_center ------------------------------------------------------------