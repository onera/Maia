import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import numpy      as np
import Converter.Internal as I

from maia import npy_pdm_gnum_dtype as pdm_dtype
import maia.sids.sids     as SIDS
import maia.tree_exchange.part_to_dist.data_exchange as PTB
from   maia.utils        import parse_yaml_cgns

dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

@mark_mpi_test(3)
def test_discover_partitioned_fields(sub_comm):
  dist_zone  = I.newZone('Zone')
  part_zones = [I.newZone('Zone.P{0}.N0'.format(sub_comm.Get_rank()))]
  if sub_comm.Get_rank() == 0:
    p_sol = I.newFlowSolution('NewSol1', 'CellCenter', part_zones[0])
    I.newDataArray('NewField1', parent=p_sol)
    I.newDataArray('NewField2', parent=p_sol)
  if sub_comm.Get_rank() == 2:
    part_zones.append(I.newZone('Zone.P{0}.N1'.format(sub_comm.Get_rank())))
    p_sol = I.newFlowSolution('NewSol2', 'Vertex', part_zones[1])
    I.newDataArray('NewField3', parent=p_sol)
    I.newDataArray('NewField4', parent=p_sol)
  p_sol = I.newFlowSolution('NewSol3', 'Vertex', part_zones[0])
  I.newDataArray('NewField5', parent=p_sol)

  PTB.discover_partitioned_fields(dist_zone, part_zones, sub_comm)

  assert [I.getName(sol) for sol in I.getNodesFromType1(dist_zone, 'FlowSolution_t')] \
      == ['NewSol1', 'NewSol3', 'NewSol2']
  assert [SIDS.GridLocation(sol) for sol in I.getNodesFromType1(dist_zone, 'FlowSolution_t')] \
      == ['CellCenter', 'Vertex', 'Vertex']
  assert I.getNodeFromPath(dist_zone, 'NewSol2/NewField4') is not None

  with pytest.raises(AssertionError):
    p_sol = I.newFlowSolution('NewSolFail', 'FaceCenter', part_zones[0])
    I.newPointList(parent=p_sol)
    PTB.discover_partitioned_fields(dist_zone, part_zones, sub_comm)

@mark_mpi_test(2)
def test_dist_to_part(sub_comm):
  part_data = dict()
  expected_dist_data = dict()
  if sub_comm.Get_rank() == 0:
    partial_distri = np.array([0, 5, 10], dtype=pdm_dtype)
    ln_to_gn_list = [np.array([2,4,6,10], dtype=pdm_dtype)]
    part_data["field"] = [np.array([2., 4., 6., 1000.])]
    expected_dist_data["field"] = np.array([1., 2., 3., 4., 5.])
  else:
    partial_distri = np.array([5, 10, 10], dtype=pdm_dtype)
    ln_to_gn_list = [np.array([9,7,5,3,1], dtype=pdm_dtype),
                     np.array([8], dtype=pdm_dtype),
                     np.array([1], dtype=pdm_dtype)]
    part_data["field"] = [np.array([9., 7., 5., 3., 1.]), np.array([8.]), np.array([1.])]
    expected_dist_data["field"] = np.array([6., 7., 8., 9., 1000.])

  dist_data = PTB.part_to_dist(partial_distri, part_data, ln_to_gn_list, sub_comm)
  assert dist_data["field"].dtype == np.float64
  assert (dist_data["field"] == expected_dist_data["field"]).all()

@mark_mpi_test(2)
def test_part_flowsol_to_dist_flowsol(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[2]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {0} [0,1,3]:
    field1 DataArray_t [10]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [0,3,6]:
  """.format(dtype)
    pt = """
  ZoneU.P0.N0 Zone_t [[3,0,0]]:
    FlowSolWithPL FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[1]]:
      field1 DataArray_t [-10]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [2]:
    NewFlowSol FlowSolution_t:
      GridLocation GridLocation_t "Vertex":
      field2 DataArray_t R8 [0,0,0]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [3,4,1]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[6,4]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {0} [1,3,3]:
    field1 DataArray_t [20,30]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,6]:
  """.format(dtype)
    pt = """
  ZoneU.P1.N0 Zone_t [[3,0,0]]:
    FlowSolWithPL FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[12,15]]:
      field1 DataArray_t [-20,-30]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [3,1]:
    NewFlowSol FlowSolution_t:
      GridLocation GridLocation_t "Vertex":
      field2 DataArray_t R8 [1,1,1]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,6,2]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_complete_pytree(dt)
  part_tree = parse_yaml_cgns.to_complete_pytree(pt)

  dist_zone  = I.getZones(dist_tree)[0]
  part_zones = I.getZones(part_tree)
  PTB.part_flowsol_to_dist_flowsol(dist_zone, part_zones, sub_comm)

  assert I.getNodeFromPath(dist_zone, 'FlowSolWithPL/field1')[1].dtype == np.int
  assert I.getNodeFromPath(dist_zone, 'NewFlowSol/field2')[1].dtype == np.float64
  if sub_comm.Get_rank () == 0:
    assert (I.getNodeFromPath(dist_zone, 'FlowSolWithPL/field1')[1] == [-30]).all()
    assert (I.getNodeFromPath(dist_zone, 'NewFlowSol/field2')[1] == [0,1,0]).all()
  if sub_comm.Get_rank () == 1:
    assert (I.getNodeFromPath(dist_zone, 'FlowSolWithPL/field1')[1] == [-10, -20]).all()
    assert (I.getNodeFromPath(dist_zone, 'NewFlowSol/field2')[1] == [0,1,1]).all()
