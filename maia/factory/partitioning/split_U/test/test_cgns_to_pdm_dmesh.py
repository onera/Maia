from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Pypdm.Pypdm as PDM

import maia
import maia.pytree as PT
from   maia.pytree.yaml   import parse_yaml_cgns
from maia.factory.partitioning.split_U import cgns_to_pdm_dmesh as CTP

from maia import npy_pdm_gnum_dtype
dtype = 'I4' if npy_pdm_gnum_dtype == np.int32 else 'I8'

@mark_mpi_test(2)
def test_split_point_list_by_dim(sub_comm):
  if sub_comm.Get_rank() == 0:
    pl1 = np.array([[1,5,3]], np.int32)
    pl2 = np.array([[14,13]], np.int32)
  elif sub_comm.Get_rank() == 1:
    pl1 =  np.array([[2,4,6]], np.int32)
    pl2 =  np.empty((1,0),     np.int32)

  pl_list = [pl1,pl2]
  range_by_dim = [[0,0], [1,10], [11,20], [21,30]]
  splitted_bc = CTP._split_point_list_by_dim(pl_list, range_by_dim, sub_comm)
  assert splitted_bc == [[], [pl1], [pl2], []]

@mark_mpi_test(2)
def test_cgns_dist_zone_to_pdm_dmesh_vtx(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[12,0,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0., 0.5, 1.,  0. , 0.5, 1. ]:
    CoordinateY DataArray_t R8 [0., 0.,  0.,  0.5, 0.5, 0.5]:
    CoordinateZ DataArray_t R8 [0., 0.,  0.,  0. , 0. , 0. ]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t [0,6,12]:
    Cell DataArray_t [0,0,0]:
  """
  elif sub_comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t [[12,0,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0., 0.5, 1., 0., 0.5, 1.]:
    CoordinateY DataArray_t R8 [1.,  1., 1., 0.,  0., 0.]:
    CoordinateZ DataArray_t R8 [0.,  0., 0., 1.,  1., 1.]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t [6,12,12]:
    Cell DataArray_t [0,0,0]:
  """
  dist_zone = parse_yaml_cgns.to_node(dt)
  dmesh = CTP.cgns_dist_zone_to_pdm_dmesh_vtx(dist_zone, sub_comm)
  #No getters for dmesh so we can not check data
  assert PT.get_child_from_name(dist_zone, ':CGNS#MultiPart') is not None

@mark_mpi_test(3)
def test_cgns_dist_zone_to_pdm_dmesh(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0., 0.5, 1.,  0. , 0.5, 1. ]:
    CoordinateY DataArray_t R8 [0., 0.,  0.,  0.5, 0.5, 0.5]:
    CoordinateZ DataArray_t R8 [0., 0.,  0.,  0. , 0. , 0. ]:
  NGonElements Elements_t [22,0]:
    ElementRange IndexRange_t [1,20]:
    ElementConnectivity DataArray_t:
      I4 : [10, 13, 4, 1,  2, 5, 14, 11,  3, 6, 15, 12,  13, 16, 7, 4,
            5, 8, 17, 14,  6, 9, 18, 15,  1, 2, 11, 10]
    ParentElements DataArray_t I4 [[21,0],[21,22],[22,0],[23,0],[23,24],[24,0],[21,0]]:
    ElementStartOffset DataArray_t [0,4,8,12,16,20,24,28]:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t [0,7,20]:
      ElementConnectivity DataArray_t [0,28,80]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t [0,6,18]:
    Cell DataArray_t [0,2,4]:
  """
  elif sub_comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0., 0.5, 1., 0., 0.5, 1.]:
    CoordinateY DataArray_t R8 [1.,  1., 1., 0.,  0., 0.]:
    CoordinateZ DataArray_t R8 [0.,  0., 0., 1.,  1., 1.]:
  NGonElements Elements_t [22,0]:
    ElementRange IndexRange_t [1,20]:
    ElementConnectivity DataArray_t:
      I4 : [2, 3, 12, 11,  4, 5, 14, 13,  5, 6, 15, 14,  16, 17, 8, 7,
            17, 18, 9, 8,  4, 5,  2,  1,  5, 6, 3,   2]
    ParentElements DataArray_t I4 [[22,0],[23,21],[24,22],[23,0],[24,0],[21,0],[22,0]]:
    ElementStartOffset DataArray_t [28,32,36,40,44,48,52,56]:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t [7,14,20]:
      ElementConnectivity DataArray_t [28,56,80]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t [6,12,18]:
    Cell DataArray_t [2,3,4]:
  """
  elif sub_comm.Get_rank() == 2:
    dt = """
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0.,  0.5, 1.,  0.,  0.5, 1.]:
    CoordinateY DataArray_t R8 [0.5, 0.5, 0.5, 1.,  1.,  1.]:
    CoordinateZ DataArray_t R8 [1.,  1.,  1.,  1.,  1. , 1.]:
  NGonElements Elements_t [22,0]:
    ElementRange IndexRange_t [1,20]:
    ElementConnectivity DataArray_t:
      I4 : [7,  8,  5,  4,  8,  9,  6,  5,  10, 11, 14, 13, 11, 12, 15, 14,
            13, 14, 17, 16, 14, 15, 18, 17]
    ParentElements DataArray_t I4 [[23,0],[24,0],[21,0],[22,0],[23,0],[24,0]]:
    ElementStartOffset DataArray_t [56,60,64,68,72,76,80]:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t [14,20,20]:
      ElementConnectivity DataArray_t [56,80,80]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t [12,18,18]:
    Cell DataArray_t [3,4,4]:
  """

  dist_zone = parse_yaml_cgns.to_node(dt)

  dmesh = CTP.cgns_dist_zone_to_pdm_dmesh(dist_zone, sub_comm)
  #No getters for dmesh so we can not check data
  assert PT.get_child_from_name(dist_zone, ':CGNS#MultiPart') is not None

@mark_mpi_test(2)
def test_cgns_dist_zone_to_pdm_dmesh_poly2d(sub_comm):
  dist_tree = maia.factory.generate_dist_block(5, "QUAD_4", sub_comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, sub_comm)
  dist_zone = PT.get_all_Zone_t(dist_tree)[0]

  dmesh_nodal = CTP.cgns_dist_zone_to_pdm_dmesh_poly2d(dist_zone, sub_comm)
  dims = PDM.dmesh_nodal_get_g_dims(dmesh_nodal)
  assert dims['n_cell_abs'] == 0
  assert dims['n_face_abs'] == 16
  assert dims['n_vtx_abs'] == 25
  assert PT.get_child_from_name(dist_zone, ':CGNS#MultiPart') is not None
  

@mark_mpi_test(3)
def test_cgns_dist_zone_to_pdm_dmesh_nodal(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = f"""
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0., 0.5, 1.,  0. , 0.5, 1. ]:
    CoordinateY DataArray_t R8 [0., 0.,  0.,  0.5, 0.5, 0.5]:
    CoordinateZ DataArray_t R8 [0., 0.,  0.,  0. , 0. , 0. ]:
  Hexa Elements_t [17,0]:
    ElementRange IndexRange_t [1,4]:
    ElementConnectivity DataArray_t {dtype} [1,2,5,4,10,11,14,13,  2,3,6,5,11,12,15,14]:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t {dtype} [0,2,4]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {dtype} [0,6,18]:
    Cell DataArray_t {dtype} [0,2,4]:
  """
    expected_dnface = 7
    expected_facecell = [1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 4, 0]
  elif sub_comm.Get_rank() == 1:
    dt = f"""
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0., 0.5, 1., 0., 0.5, 1.]:
    CoordinateY DataArray_t R8 [1.,  1., 1., 0.,  0., 0.]:
    CoordinateZ DataArray_t R8 [0.,  0., 0., 1.,  1., 1.]:
  Hexa Elements_t [17,0]:
    ElementRange IndexRange_t [1,4]:
    ElementConnectivity DataArray_t {dtype} [4,5,8,7,13,14,17,16]:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t {dtype} [2,3,4]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {dtype} [6,12,18]:
    Cell DataArray_t {dtype} [2,3,4]:
  """
    expected_dnface = 5
    expected_facecell = [2, 1, 1, 3, 2, 0, 2, 4, 3, 0]
  elif sub_comm.Get_rank() == 2:
    dt = f"""
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t R8 [0.,  0.5, 1.,  0.,  0.5, 1.]:
    CoordinateY DataArray_t R8 [0.5, 0.5, 0.5, 1.,  1.,  1.]:
    CoordinateZ DataArray_t R8 [1.,  1.,  1.,  1.,  1. , 1.]:
  Hexa Elements_t [17,0]:
    ElementRange IndexRange_t [1,4]:
    ElementConnectivity DataArray_t {dtype} [5,6,9,8,14,15,18,17]:
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t {dtype} [3,4,4]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {dtype} [12,18,18]:
    Cell DataArray_t {dtype} [3,4,4]:
  """
    expected_dnface = 8
    expected_facecell = [3, 4, 1, 0, 3, 0, 4, 0, 2, 0, 4, 0, 3, 0, 4, 0]

  dist_zone = parse_yaml_cgns.to_node(dt)

  dmeshnodal = CTP.cgns_dist_zone_to_pdm_dmesh_nodal(dist_zone, sub_comm)

  assert PT.get_child_from_name(dist_zone, ':CGNS#MultiPart') is not None
  dims = PDM.dmesh_nodal_get_g_dims(dmeshnodal)
  assert dims['n_cell_abs'] == 4
  assert dims['n_vtx_abs'] == 18
