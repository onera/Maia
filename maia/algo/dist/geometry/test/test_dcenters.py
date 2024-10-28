import pytest
import pytest_parallel
import numpy as np

import maia.pytree        as PT
import maia

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


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center3d_u_ngon(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm)

  if comm.Get_rank() == 0:
    expected_face_center = np.array([
        0.25, 0.25, 0. ,  0.75, 0.25, 0. ,  0.25, 0.75, 0. ,  0.75, 0.75, 0. , 
        0.25, 0.25, 0.5,  0.75, 0.25, 0.5,  0.25, 0.75, 0.5,  0.75, 0.75, 0.5, 
        0.25, 0.25, 1. ,  0.75, 0.25, 1. ,  0.25, 0.75, 1. ,  0.75, 0.75, 1. ,
    ])
  elif comm.Get_rank() == 1:
    expected_face_center = np.array([
        0. , 0.25, 0.25,  0. , 0.75, 0.25,  0.,  0.25, 0.75,  0. , 0.75, 0.75,
        0.5, 0.25, 0.25,  0.5, 0.75, 0.25,  0.5, 0.25, 0.75,  0.5, 0.75, 0.75,
        1. , 0.25, 0.25,  1. , 0.75, 0.25,  1.,  0.25, 0.75,  1. , 0.75, 0.75,
    ])
  if comm.Get_rank() == 2:
    expected_face_center = np.array([
        0.25, 0. , 0.25,  0.25, 0. , 0.75,  0.75, 0. , 0.25,  0.75, 0. , 0.75, 
        0.25, 0.5, 0.25,  0.25, 0.5, 0.75,  0.75, 0.5, 0.25,  0.75, 0.5, 0.75,
        0.25, 1. , 0.25,  0.25, 1. , 0.75,  0.75, 1. , 0.25,  0.75, 1. , 0.75,
    ])

  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False,True])
def test_compute_face_center3d_u_elts(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm)
  expec = [
    [0.25, 0.25, 0.  ,   0.75, 0.25, 0.  ,   0.25, 0.75, 0.  ,   0.75, 0.75, 0.  ,
     0.25, 0.25, 1.  ,   0.75, 0.25, 1.  ,   0.25, 0.75, 1.  ,   0.75, 0.75, 1.  ],
    [0.  , 0.25, 0.25,   0.  , 0.75, 0.25,   0.  , 0.25, 0.75,   0.  , 0.75, 0.75,
     1.  , 0.25, 0.25,   1.  , 0.75, 0.25,   1.  , 0.25, 0.75,   1.  , 0.75, 0.75],
    [0.25, 0.  , 0.25,   0.25, 0.  , 0.75,   0.75, 0.  , 0.25,   0.75, 0.  , 0.75,
     0.25, 1.  , 0.25,   0.25, 1.  , 0.75,   0.75, 1.  , 0.25,   0.75, 1.  , 0.75],
  ]

  expected_face_center = np.array(expec[comm.rank])

  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False,True])
def test_compute_face_center3d_s(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  # logger.warn(comm.rank, face_ind)
  face_center = GEO.compute_face_center(zone, comm)
  expec = [
    [0.  , 0.25, 0.25,   0.5 , 0.25, 0.25,   1.  , 0.25, 0.25,   0.  , 0.75, 0.25,
     0.5 , 0.75, 0.25,   1.  , 0.75, 0.25,   0.  , 0.25, 0.75,   0.5 , 0.25, 0.75,
     1.  , 0.25, 0.75,   0.  , 0.75, 0.75,   0.5 , 0.75, 0.75,   1.  , 0.75, 0.75],
    [0.25, 0.  , 0.25,   0.75, 0.  , 0.25,   0.25, 0.5 , 0.25,   0.75, 0.5 , 0.25,
     0.25, 1.  , 0.25,   0.75, 1.  , 0.25,   0.25, 0.  , 0.75,   0.75, 0.  , 0.75,
     0.25, 0.5 , 0.75,   0.75, 0.5 , 0.75,   0.25, 1.  , 0.75,   0.75, 1.  , 0.75],
    [0.25, 0.25, 0.  ,   0.75, 0.25, 0.  ,   0.25, 0.75, 0.  ,   0.75, 0.75, 0.  ,
     0.25, 0.25, 0.5 ,   0.75, 0.25, 0.5 ,   0.25, 0.75, 0.5 ,   0.75, 0.75, 0.5 ,
     0.25, 0.25, 1.  ,   0.75, 0.25, 1.  ,   0.25, 0.75, 1.  ,   0.75, 0.75, 1.  ]
  ]
  expected_face_center = np.array(expec[comm.rank])

  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)


@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("bc_path,expec", [
  ("ZoneBC/Ymin/PointList", [[], [], [0.25, 0., 0.25, 0.25, 0., 0.75, 0.75, 0., 0.25, 0.75, 0., 0.75]]),
  ("ZoneBC/Xmax/PointList", [[], [1., 0.25, 0.25, 1., 0.75, 0.25, 1., 0.25, 0.75, 1., 0.75, 0.75], []]),
  ("ZoneBC/Zmin/PointList", [[0.25, 0.25, 0., 0.75, 0.25, 0., 0.25, 0.75, 0., 0.75, 0.75, 0.], [], []]),
  ])
def test_compute_face_center3d_u_ngon_filtered(cylindrical, comm, bc_path, expec):
  tree = maia.factory.generate_dist_block(3, 'Poly', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  face_ind = PT.get_value(PT.get_node_from_path(zone,bc_path))
  
  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  comm.barrier()
  face_center = GEO.compute_face_center(zone, comm, face_ind)

  expected_face_center = np.array(expec[comm.rank])
  
  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False,True])
@pytest.mark.parametrize("bc_path,expec", [
  ("ZoneBC/Xmin/PointList", [[0.,0.25,0.25,0.,0.75,0.25],[0.,0.25,0.75],[0.,0.75,0.75]]),
  ("ZoneBC/Ymax/PointList", [[0.25,1.,0.25,0.25,1.,0.75],[0.75,1.,0.25],[0.75,1.,0.75]]),
  ("ZoneBC/Zmin/PointList", [[0.25,0.25,0.,0.75,0.25,0.,],[0.25,0.75,0.],[0.75,0.75,0.]]),
  ])
def test_compute_face_center3d_u_elts_filtered(cylindrical, comm, bc_path,expec):
  tree = maia.factory.generate_dist_block(3, 'HEXA_8', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  face_ind = PT.get_value(PT.get_node_from_path(zone,bc_path))

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm, face_ind)
  # return
  expected_face_center = np.array(expec[comm.rank])

  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

# @pytest_parallel.mark.parallel(1)
@pytest_parallel.mark.parallel(3)
@pytest.mark.parametrize("cylindrical", [False,
                                        #  True
                                         ])
@pytest.mark.parametrize("bc_path,d,expec", [
  ("ZoneBC/Xmin/PointRange","I", [
    [0.  , 0.25, 0.25, 0.  , 0.75, 0.25, 0.  , 0.25, 0.75, 0.  , 0.75, 0.75]
  ]*3),
  ("ZoneBC/Xmax/PointRange","I", [
    [1.  , 0.25, 0.25, 1.  , 0.75, 0.25, 1.  , 0.25, 0.75, 1.  , 0.75, 0.75]
  ]*3),
  ("ZoneBC/Ymin/PointRange","J", [
    [0.25, 0.  , 0.25, 0.75, 0.  , 0.25, 0.25, 0.  , 0.75, 0.75, 0.  , 0.75]
  ]*3),
  ("ZoneBC/Ymax/PointRange","J", [
    [0.25, 1.  , 0.25, 0.75, 1.  , 0.25, 0.25, 1.  , 0.75, 0.75, 1.  , 0.75]
  ]*3),
  ("ZoneBC/Zmin/PointRange","K", [
    [0.25, 0.25, 0.  , 0.75, 0.25, 0.  , 0.25, 0.75, 0.  , 0.75, 0.75, 0.  ]
  ]*3),
  ("ZoneBC/Zmax/PointRange","K", [
    [0.25, 0.25, 1.  , 0.75, 0.25, 1.  , 0.25, 0.75, 1.  , 0.75, 0.75, 1.  ]
  ]*3),
  ])
def test_compute_face_center3d_s_filtered(cylindrical, comm, bc_path, d, expec):
  tree = maia.factory.generate_dist_block(3, 'S', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  face_ind = PT.get_value(PT.get_node_from_path(zone,bc_path))
  # vertex -> face
  face_ind = np.array([f if di==d else np.clip(f,0,2) for f,di in zip(face_ind,'IJK')])
  n_vtx  = PT.Zone.VertexSize(zone)
  from maia.utils import pr_utils
  point_list = pr_utils.compute_pointList_from_pointRanges([face_ind],
                                                           n_vtx,f"{d}FaceCenter")
  
  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm, point_list)
  expected_face_center = np.array(expec[comm.rank])

  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center2d_u_ngon(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

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
def test_compute_face_center2d_u_elts(cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  expec = [
    [1/6, 1/6, 0. , 1/3, 1/3, 0. , 2/3, 1/6, 0. , 5/6, 1/3,  0.],
    [1/6, 2/3, 0. , 1/3, 5/6, 0. , 2/3, 2/3, 0. , 5/6, 5/6,  0.]
  ]

  face_center = GEO.compute_face_center(zone, comm)

  expected_face_center = np.array(expec[comm.rank])
  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_face_center2d_s(cylindrical, comm):
  tree = maia.factory.generate_dist_block([3,3,1], 'Structured', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm)
  expec = [
    [0.25, 0.25, 0.  , 0.75, 0.25, 0.  ],
    [0.25, 0.75, 0.  , 0.75, 0.75, 0.  ],
  ]
  expected_face_center = np.array(expec[comm.rank])
  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest.mark.parametrize("cylindrical", [False,True])
@pytest.mark.parametrize("pl,expec", [
  ([[1],[3]],[[1/6,1/6,0.],[2/3,1/6,0]]),
  ([[7,8],[1,2]],[[2/3,2/3,0.,5/6,5/6,0],[1/6,1/6,0.,1/3,1/3,0.]]),
  ([[],[4,6]],[[],[5/6,1/3,0,1/3,5/6,0]]),
  ([[4,6],[]],[[5/6,1/3,0,1/3,5/6,0],[]]),
  ([[],[]],[[],[]]),
])
@pytest_parallel.mark.parallel(2)
def test_compute_face_center2d_u_ngon_filtered(cylindrical, pl, expec, comm):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  maia.algo.dist.convert_elements_to_ngon(tree, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  face_ind = pl[comm.rank]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm, face_indices=face_ind)

  expected_face_center = np.array(expec[comm.rank])
  
  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cylindrical", [False,True])
@pytest.mark.parametrize("pl,expec", [
  ([[1],[3]],[[1/6,1/6,0.],[2/3,1/6,0]]),
  ([[7,8],[1,2]],[[2/3,2/3,0.,5/6,5/6,0],[1/6,1/6,0.,1/3,1/3,0.]]),
  ([[],[4,6]],[[],[5/6,1/3,0,1/3,5/6,0]]),
  ([[4,6],[]],[[5/6,1/3,0,1/3,5/6,0],[]]),
  ([[],[]],[[],[]]),
])
def test_compute_face_center2d_u_elts_filtered(cylindrical, comm, pl, expec):
  tree = maia.factory.generate_dist_block(3, 'TRI_3', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  face_ind = pl[comm.rank]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm, face_indices=face_ind)

  expected_face_center = np.array(expec[comm.rank])
  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("pr,expec", [
  ([[[1,2],[1,2]],[[2,2],[1,1]]],[[0.25, 0.25, 0.  , 0.75, 0.25, 0.  , 0.25, 0.75, 0.  , 0.75, 0.75, 0.  ],[0.75, 0.25, 0.  ]]),
  ([[[1,3],[1,1]],[[1,3],[1,1]]],[[0.25, 0.25, 0.  , 0.75, 0.25, 0.  , 0.25, 0.75, 0.  ]]*2),
  ])
def test_compute_face_center2d_s_filtered(cylindrical, comm, pr, expec):
  tree = maia.factory.generate_dist_block([3,3,1], 'Structured', comm)
  zone = PT.get_all_Zone_t(tree)[0]

  # vertex -> face
  n_vtx  = PT.Zone.VertexSize(zone)
  from maia.utils import pr_utils
  point_list = pr_utils.compute_pointList_from_pointRanges([np.array(pr[comm.rank])],
                                                           n_vtx,"CellCenter")

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))

  face_center = GEO.compute_face_center(zone, comm, point_list)
  expected_face_center = np.array(expec[comm.rank])

  if cylindrical:
    expected_face_center = to_expected_cyl(expected_face_center)

  assert np.allclose(face_center, expected_face_center)

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("elt_kind", ["S", "NFACE_n", "Poly"])
@pytest.mark.parametrize("cylindrical", [False, True])
def test_compute_cell_center(elt_kind, cylindrical, comm):
  tree = maia.factory.generate_dist_block(3, elt_kind, comm)
  zone = PT.get_all_Zone_t(tree)[0]

  if cylindrical:
    maia.algo.cartesian_to_cylindrical(tree, (0,0,1))
  
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
