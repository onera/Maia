from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import Pypdm.Pypdm as PDM

from maia import npy_pdm_gnum_dtype
from   maia.utils        import parse_yaml_cgns
from maia.partitioning.split_U import cgns_to_pdm_dmesh_nodal as CTP

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
  

@mark_mpi_test(3)
def test_cgns_dist_zone_to_pdm_dmesh_nodal(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = f"""
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t [0., 0.5, 1.,  0. , 0.5, 1. ]:
    CoordinateY DataArray_t [0., 0.,  0.,  0.5, 0.5, 0.5]:
    CoordinateZ DataArray_t [0., 0.,  0.,  0. , 0. , 0. ]:
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
    CoordinateX DataArray_t [0., 0.5, 1., 0., 0.5, 1.]:
    CoordinateY DataArray_t [1.,  1., 1., 0.,  0., 0.]:
    CoordinateZ DataArray_t [0.,  0., 0., 1.,  1., 1.]:
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
    CoordinateX DataArray_t [0.,  0.5, 1.,  0.,  0.5, 1.]:
    CoordinateY DataArray_t [0.5, 0.5, 0.5, 1.,  1.,  1.]:
    CoordinateZ DataArray_t [1.,  1.,  1.,  1.,  1. , 1.]:
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

  assert I.getNodeFromName(dist_zone, ':CGNS#MultiPart') is not None
  dims = PDM.dmesh_nodal_get_g_dims(dmeshnodal)
  assert dims['n_cell_abs'] == 4
  assert dims['n_vtx_abs'] == 18
