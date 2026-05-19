import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import interpolation_cons as ITP

minimal_tri = """
  zone Zone_t [[6, 5, 0]]:
    ZoneType ZoneType_t "Unstructured":
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [0, 0, 0, 1, 1, 0.5]:
      CoordinateY DataArray_t R8 [1, 0.5, 0, 0, 1, 0.5]:
      CoordinateZ DataArray_t R8 [0, 0, 0, 0, 0, 0]:
    TRI Elements_t [5, 0]:
      ElementRange IndexRange_t [1, 5]:
      ElementConnectivity DataArray_t [1,2,6, 2,3,6, 3,4,6, 4,5,6, 5,1,6]:
    Geometry_0d DiscreteData_t:
      DualVol24 DataArray_t R8 [3, 2, 3, 4, 4, 8]:
    Geometry_2d DiscreteData_t:
      GridLocation GridLocation_t "CellCenter":
      Measure DataArray_t R8 [0.125, 0.125, 0.25, 0.25, 0.25]:
"""


@pytest_parallel.mark.parallel(2)
def test_vtx2cell(comm):
  
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      field DataArray_t R8 [4, 5, 7, 10, 8, 3]:
  """)
  tree = maia.factory.full_to_dist_tree(ftree, comm)
  cell_distri = MT.Zone.cell_distribution(PT.find_node_from_label(tree, 'Zone_t'))

  vtx_field = PT.get_np_value(PT.find_node_from_name(tree, 'field'))
  dual_vol = PT.get_np_value(PT.find_node_from_name(tree, 'DualVol24')) / 24
  it = ITP.VertexToCell(PT.get_all_Zone_t(tree), comm)
  cell_field = it._exchange_fields({'field' : [vtx_field]}, True)
  
  expected_cell_val = np.array([12, 15, 20, 21, 15]) / 3
  assert len(cell_field) == 1 and len(vals := cell_field['field']) == 1
  assert np.allclose(vals[0], expected_cell_val[cell_distri[0]:cell_distri[1]])

  # From integrated
  vtx_field *= dual_vol
  cell_vol = PT.get_np_value(PT.find_node_from_name(tree, 'Measure'))
  cell_field = it._exchange_fields({'field': [vtx_field]}, False)['field'][0]
  assert np.allclose(cell_field/cell_vol, expected_cell_val[cell_distri[0]:cell_distri[1]])

  # Check conservativity (on integrated var)
  vtx_sum = comm.allreduce(vtx_field.sum())
  cell_sum = comm.allreduce(cell_field.sum())
  assert abs(vtx_sum - cell_sum) < 1E-12

@pytest_parallel.mark.parallel(2)
def test_cell2vtx(comm):
  
  ftree = PT.yaml.to_cgns_tree(minimal_tri + """
    Sol FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      field DataArray_t R8 [2, 6, 3, 8, 5]:
  """)
  tree = maia.factory.full_to_dist_tree(ftree, comm)
  vtx_distri = MT.Zone.vtx_distribution(PT.find_node_from_label(tree, 'Zone_t'))

  cell_field = PT.get_np_value(PT.find_node_from_name(tree, 'field'))

  it = ITP.CellToVertex(PT.get_all_Zone_t(tree), comm)
  vtx_field = it._exchange_fields({'field' : [cell_field]}, False)

  expected_vtx_val = np.array([7, 8, 9, 11, 13, 24]) / 3
  assert len(vtx_field) == 1 and len(vals := vtx_field['field']) == 1
  assert np.allclose(vals[0], expected_vtx_val[vtx_distri[0]:vtx_distri[1]])

  # From conservative
  vol = PT.get_np_value(PT.find_node_from_name(tree, 'Measure'))
  dual_vol = PT.get_np_value(PT.find_node_from_name(tree, 'DualVol24')) / 24
  cell_field_cons = cell_field / vol
  vtx_field_cons = it._exchange_fields({'field' : [cell_field_cons]}, True)['field'][0]
  assert np.allclose(vtx_field_cons*dual_vol, expected_vtx_val[vtx_distri[0]:vtx_distri[1]])
  
  # Check conservativity (on integrated var)
  _vtx_field = vtx_field['field'][0]
  vtx_sum = comm.allreduce(_vtx_field.sum())
  cell_sum = comm.allreduce(cell_field.sum())
  assert abs(vtx_sum - cell_sum) < 1E-12