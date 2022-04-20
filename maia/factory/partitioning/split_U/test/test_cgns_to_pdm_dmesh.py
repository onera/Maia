from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import Pypdm.Pypdm as PDM

from   maia.utils.yaml   import parse_yaml_cgns
from maia.factory.partitioning.split_U import cgns_to_pdm_dmesh as CTP

@mark_mpi_test(3)
def test_cgns_dist_zone_to_pdm_dmesh(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t [0., 0.5, 1.,  0. , 0.5, 1. ]:
    CoordinateY DataArray_t [0., 0.,  0.,  0.5, 0.5, 0.5]:
    CoordinateZ DataArray_t [0., 0.,  0.,  0. , 0. , 0. ]:
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
    CoordinateX DataArray_t [0., 0.5, 1., 0., 0.5, 1.]:
    CoordinateY DataArray_t [1.,  1., 1., 0.,  0., 0.]:
    CoordinateZ DataArray_t [0.,  0., 0., 1.,  1., 1.]:
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
    CoordinateX DataArray_t [0.,  0.5, 1.,  0.,  0.5, 1.]:
    CoordinateY DataArray_t [0.5, 0.5, 0.5, 1.,  1.,  1.]:
    CoordinateZ DataArray_t [1.,  1.,  1.,  1.,  1. , 1.]:
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
  PE = I.getNodeFromName(dist_zone, 'ParentElements')

  dmesh = CTP.cgns_dist_zone_to_pdm_dmesh(dist_zone, sub_comm)
  #No getters for dmesh so we can not check data
  assert I.getNodeFromName(dist_zone, ':CGNS#MultiPart') is not None

