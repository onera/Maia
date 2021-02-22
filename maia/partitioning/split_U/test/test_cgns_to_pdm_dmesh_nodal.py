from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I
import Pypdm.Pypdm as PDM

from   maia.utils        import parse_yaml_cgns
from maia.partitioning.split_U import cgns_to_pdm_dmesh_nodal as CTP

@mark_mpi_test(3)
def test_cgns_dist_zone_to_pdm_dmesh(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t [0., 0.5, 1.,  0. , 0.5, 1. ]:
    CoordinateY DataArray_t [0., 0.,  0.,  0.5, 0.5, 0.5]:
    CoordinateZ DataArray_t [0., 0.,  0.,  0. , 0. , 0. ]:
  Hexa Elements_t [17,0]:
    ElementConnectivity DataArray_t:
      I4 : [1,2,5,4,10,11,14,13,  2,3,6,5,11,12,15,14]
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t [0,2,4]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t [0,6,18]:
    Cell DataArray_t [0,2,4]:
  """
    expected_dnface = 7
    expected_facecell = [1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 4, 0]
  elif sub_comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t [0., 0.5, 1., 0., 0.5, 1.]:
    CoordinateY DataArray_t [1.,  1., 1., 0.,  0., 0.]:
    CoordinateZ DataArray_t [0.,  0., 0., 1.,  1., 1.]:
  Hexa Elements_t [17,0]:
    ElementConnectivity DataArray_t:
      I4 : [4,5,8,7,13,14,17,16]
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t [2,3,4]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t [6,12,18]:
    Cell DataArray_t [2,3,4]:
  """
    expected_dnface = 5
    expected_facecell = [2, 1, 1, 3, 2, 0, 2, 4, 3, 0]
  elif sub_comm.Get_rank() == 2:
    dt = """
ZoneU Zone_t [[18,6,0]]:
  GridCoordinates GridCoordinates_t:
    CoordinateX DataArray_t [0.,  0.5, 1.,  0.,  0.5, 1.]:
    CoordinateY DataArray_t [0.5, 0.5, 0.5, 1.,  1.,  1.]:
    CoordinateZ DataArray_t [1.,  1.,  1.,  1.,  1. , 1.]:
  Hexa Elements_t [17,0]:
    ElementConnectivity DataArray_t:
      I4 : [5,6,9,8,14,15,18,17]
    :CGNS#Distribution UserDefinedData_t:
      Element DataArray_t [3,4,4]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t [12,18,18]:
    Cell DataArray_t [3,4,4]:
  """
    expected_dnface = 8
    expected_facecell = [3, 4, 1, 0, 3, 0, 4, 0, 2, 0, 4, 0, 3, 0, 4, 0]

  dist_tree = parse_yaml_cgns.to_complete_pytree(dt)
  dist_zone  = I.getZones(dist_tree)[0]

  dmeshnodal = CTP.cgns_dist_zone_to_pdm_dmesh_nodal(dist_zone, sub_comm)

  assert I.getNodeFromName(dist_zone, ':CGNS#MultiPart') is not None
  dmeshnodal.compute()
  face_cell = dmeshnodal.get_face_cell()
  assert face_cell['n_face'] == 20
  assert face_cell['dn_face'] == expected_dnface
  assert (face_cell['np_dface_cell'] == expected_facecell).all()
