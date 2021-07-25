import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import numpy      as np
import Converter.Internal as I
import maia.sids.Internal_ext as IE

from maia import npy_pdm_gnum_dtype as pdm_dtype
import maia.sids.sids     as SIDS
import maia.tree_exchange.part_to_dist.data_exchange as PTB
from   maia.utils        import parse_yaml_cgns

dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

@mark_mpi_test(3)
class Test__discover_wrapper:
  def test_sol_without_pl(self, sub_comm):
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

    PTB._discover_wrapper(dist_zone, part_zones, 'FlowSolution_t', 'FlowSolution_t/DataArray_t', sub_comm)

    assert [I.getName(sol) for sol in I.getNodesFromType1(dist_zone, 'FlowSolution_t')] \
        == ['NewSol1', 'NewSol3', 'NewSol2']
    assert [SIDS.GridLocation(sol) for sol in I.getNodesFromType1(dist_zone, 'FlowSolution_t')] \
        == ['CellCenter', 'Vertex', 'Vertex']
    assert I.getNodeFromPath(dist_zone, 'NewSol2/NewField4') is not None

  def test_sol_with_pl(self, sub_comm):
    dt = """
  Zone Zone_t:
    Hexa Elements_t [17,0]:
      ElementRange IndexRange_t [1,50]:
    """
    if sub_comm.Get_rank() == 0:
      pt = """
    Zone.P0.N0 Zone_t:
      Hexa Elements_t [17,0]:
        ElementRange IndexRange_t [1,5]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {0} [29,42,19,18,41]:
      FS DiscreteData_t:
        GridLocation GridLocation_t "CellCenter":
        PointList IndexArray_t [[4,2]]:
        field DataArray_t:
      """.format(dtype)
    if sub_comm.Get_rank() == 1:
      pt = ""
    if sub_comm.Get_rank() == 2:
      pt = """
    Zone.P2.N0 Zone_t:
    Zone.P2.N1 Zone_t:
      Hexa Elements_t [17,0]:
        ElementRange IndexRange_t [1,4]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {0} [8,13,18,25]:
      FS DiscreteData_t:
        GridLocation GridLocation_t "CellCenter":
        PointList IndexArray_t [[1,4,3]]:
        field DataArray_t:
      """.format(dtype)
    dist_zone  = parse_yaml_cgns.to_node(dt)
    part_zones = parse_yaml_cgns.to_nodes(pt)

    PTB._discover_wrapper(dist_zone, part_zones, \
        'DiscreteData_t', 'DiscreteData_t/DataArray_t', sub_comm)

    fs = I.getNodeFromName(dist_zone, 'FS')
    assert I.getType(fs) == 'DiscreteData_t'
    dist_pl     = I.getNodeFromPath(fs, 'PointList')[1]
    dist_distri = I.getVal(IE.getDistribution(fs, 'Index'))
    assert dist_distri.dtype == pdm_dtype

    if sub_comm.Get_rank() == 0:
      assert (dist_distri == [0,2,4]).all()
      assert (dist_pl     == [8,18] ).all()
    elif sub_comm.Get_rank() == 1:
      assert (dist_distri == [2,3,4]).all()
      assert (dist_pl     == [25]   ).all()
    elif sub_comm.Get_rank() == 2:
      assert (dist_distri == [3,4,4]).all()
      assert (dist_pl     == [42]   ).all()

  def test_dataset(self, sub_comm):
    dt = """
  Zone Zone_t:
    ZBC ZoneBC_t:
      bc1 BC_t:
      bc2 BC_t:
    """
    if sub_comm.Get_rank() == 0:
      pt = """
    Zone.P0.N0 Zone_t:
      ZBC ZoneBC_t:
        bc1 BC_t:
          BCDS BCDataSet_t 'BCInflow':
            BCData BCData_t:
              newField1 DataArray_t:
              newField2 DataArray_t:
      """
    elif sub_comm.Get_rank() == 1:
      pt = """
    Zone.P1.N0 Zone_t:
      """
    if sub_comm.Get_rank() == 2:
      pt = """
    Zone.P2.N0 Zone_t:
      ZBC ZoneBC_t:
        bc1 BC_t:
          BCDS2 BCDataSet_t 'BCWall':
            BCData BCData_t:
              newField3 DataArray_t:
    Zone.P2.N1 Zone_t:
      ZBC ZoneBC_t:
        bc2 BC_t:
          BCDS BCDataSet_t 'Null':
            BCData BCData_t:
              newField4 DataArray_t:
      """
    dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
    part_tree = parse_yaml_cgns.to_cgns_tree(pt)

    bc_ds_path = 'ZoneBC_t/BC_t/BCDataSet_t'
    PTB._discover_wrapper(I.getZones(dist_tree)[0], I.getZones(part_tree), \
        bc_ds_path, bc_ds_path+'/BCData_t/DataArray_t', sub_comm)

    assert [I.getValue(bcds) for bcds in I.getNodesFromType(dist_tree, 'BCDataSet_t')] \
        == ['BCInflow', 'BCWall', 'Null']
    assert [I.getName(field) for field in I.getNodesFromType(dist_tree, 'DataArray_t')] \
        == ['newField1', 'newField2', 'newField3', 'newField4']

@mark_mpi_test(2)
def test_part_to_dist(sub_comm):
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
def test_part_coords_to_dist_coords(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [0,3,6]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t:
    CY DataArray_t:
  """.format(dtype)
    pt = """
  ZoneU.P0.N0 Zone_t [[3,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [1,6]:
    GridCoordinates GridCoordinates_t:
      CX DataArray_t [1,6]:
      CY DataArray_t [2,1]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,6]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t:
    CY DataArray_t:
  """.format(dtype)
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,2]:
    GridCoordinates GridCoordinates_t:
      CX DataArray_t [5,2]:
      CY DataArray_t [1,2]:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [3,4]:
    GridCoordinates GridCoordinates_t:
      CX DataArray_t [3,4]:
      CY DataArray_t [2,1]:
  """.format(dtype)

  dist_zone  = parse_yaml_cgns.to_node(dt)
  part_zones = parse_yaml_cgns.to_nodes(pt)

  PTB.part_coords_to_dist_coords(dist_zone, part_zones, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CX')[1] == [1,6]).all()
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CY')[1] == [2,1]).all()
  elif sub_comm.Get_rank() == 1:
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CX')[1] == [5,2]).all()
    assert (I.getNodeFromPath(part_zones[0], 'GridCoordinates/CY')[1] == [1,2]).all()
    assert (I.getNodeFromPath(part_zones[1], 'GridCoordinates/CX')[1] == [3,4]).all()
    assert (I.getNodeFromPath(part_zones[1], 'GridCoordinates/CY')[1] == [2,1]).all()

@mark_mpi_test(2)
def test_part_sol_to_dist_sol(sub_comm):
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
    NewFlowSol DiscreteData_t:
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
    NewFlowSol DiscreteData_t:
      GridLocation GridLocation_t "Vertex":
      field2 DataArray_t R8 [1,1,1]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,6,2]:
  """.format(dtype)

  dist_zone  = parse_yaml_cgns.to_node(dt)
  part_zones = parse_yaml_cgns.to_nodes(pt)

  PTB.part_sol_to_dist_sol(dist_zone, part_zones, sub_comm)

  assert I.getNodeFromPath(dist_zone, 'FlowSolWithPL/field1')[1].dtype == np.int32
  assert I.getNodeFromPath(dist_zone, 'NewFlowSol/field2')[1].dtype == np.float64
  if sub_comm.Get_rank () == 0:
    assert (I.getNodeFromPath(dist_zone, 'FlowSolWithPL/field1')[1] == [-30]).all()
    assert (I.getNodeFromPath(dist_zone, 'NewFlowSol/field2')[1] == [0,1,0]).all()
  if sub_comm.Get_rank () == 1:
    assert (I.getNodeFromPath(dist_zone, 'FlowSolWithPL/field1')[1] == [-10, -20]).all()
    assert (I.getNodeFromPath(dist_zone, 'NewFlowSol/field2')[1] == [0,1,1]).all()

@mark_mpi_test(2)
def test_part_subregion_to_dist_subregion(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[18, 22]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,2,6]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[6]]:
    field DataArray_t I4 [[42]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [0,1,2]:
  LinkedZSR ZoneSubRegion_t:
    GridLocation GridLocation_t "FaceCenter":
    GridConnectivityRegionName Descriptor_t "GC":
    field DataArray_t:
  """.format(dtype)
    pt = """
  ZoneU.P0.N0 Zone_t [[3,0,0]]:
    ZGC ZoneGridConnectivity_t:
      GC GridConnectivity_t:
        PointList IndexArray_t [[1, 12, 21]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [2,5,1]:
    LinkedZSR ZoneSubRegion_t:
      GridLocation GridLocation_t "FaceCenter":
      GridConnectivityRegionName Descriptor_t "GC":
      field DataArray_t R8 [[200,500,100]]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[13, 39, 41, 9]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [2,6,6]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[2]]:
    field DataArray_t I4 [[24]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [1,2,2]:
  LinkedZSR ZoneSubRegion_t:
    GridLocation GridLocation_t "FaceCenter":
    GridConnectivityRegionName Descriptor_t "GC":
    field DataArray_t:
  """.format(dtype)
    pt = """
  ZoneU.P1.N0 Zone_t [[3,0,0]]:
    ZGC ZoneGridConnectivity_t:
      GC GridConnectivity_t:
        PointList IndexArray_t [[1, 29, 108]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [6,3,4]:
    ZSRWithPL ZoneSubRegion_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[1,2]]:
      field DataArray_t I4 [[84,48]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [1,2]:
    LinkedZSR ZoneSubRegion_t:
      GridLocation GridLocation_t "FaceCenter":
      GridConnectivityRegionName Descriptor_t "GC":
      field DataArray_t R8 [[600,300,400]]:
  """.format(dtype)

  dist_zone  = parse_yaml_cgns.to_node(dt)
  part_zones = parse_yaml_cgns.to_nodes(pt)

  PTB.part_subregion_to_dist_subregion(dist_zone, part_zones, sub_comm)

  assert I.getNodeFromPath(dist_zone, 'ZSRWithPL/field')[1].dtype == np.int32
  assert I.getNodeFromPath(dist_zone, 'LinkedZSR/field')[1].dtype == np.float64
  if sub_comm.Get_rank () == 0:
    assert (I.getNodeFromPath(dist_zone, 'ZSRWithPL/field')[1] == [84]).all()
    assert (I.getNodeFromPath(dist_zone, 'LinkedZSR/field')[1] == [100,200]).all()
  if sub_comm.Get_rank () == 1:
    assert (I.getNodeFromPath(dist_zone, 'ZSRWithPL/field')[1] == [48]).all()
    assert (I.getNodeFromPath(dist_zone, 'LinkedZSR/field')[1] == [300,400,500,600]).all()

@mark_mpi_test(2)
def test_part_dataset_to_dist_dataset(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[18, 22]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,2,6]:
      BCDSWithoutPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t:
      BCDSWithPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t R8 [[100]]:
        PointList IndexArray_t [[10]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {0} [0,1,1]:
  """.format(dtype)
    pt = """
  ZoneU.P0.N0 Zone_t:
    ZBC ZoneBC_t:
      BC BC_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 12]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [2,5]:
        BCDSWithoutPL BCDataSet_t:
          DirichletData BCData_t:
            field DataArray_t [[2,2]]:
    """.format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[13, 39, 41, 9]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [2,6,6]:
      BCDSWithoutPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t:
      BCDSWithPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t R8 [[]]:
        PointList IndexArray_t [[]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {0} [1,1,1]:
  """.format(dtype)
    pt = """
  ZoneU.P1.N0 Zone_t:
    ZBC ZoneBC_t:
      BC BC_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1,29,108,21]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [6,3,4,1]:
        BCDSWithPL BCDataSet_t:
          DirichletData BCData_t:
            field DataArray_t R8 [[200.]]:
          PointList IndexArray_t [[108]]:
          :CGNS#GlobalNumbering UserDefinedData_t:
            Index DataArray_t {0} [1]:
        BCDSWithoutPL BCDataSet_t:
          DirichletData BCData_t:
            field DataArray_t [[1,4,3,1]]:
  """.format(dtype)

  dist_zone  = parse_yaml_cgns.to_node(dt)
  part_zones = parse_yaml_cgns.to_nodes(pt)

  PTB.part_dataset_to_dist_dataset(dist_zone, part_zones, sub_comm)

  assert I.getNodeFromPath(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field')[1].dtype    == np.float64
  assert I.getNodeFromPath(dist_zone, 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1].dtype == np.int32
  if sub_comm.Get_rank () == 0:
    assert (I.getNodeFromPath(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field')[1] == [200.]).all()
    assert (I.getNodeFromPath(dist_zone, 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1] == [1,2]).all()
  if sub_comm.Get_rank () == 1:
    assert len(I.getNodeFromPath(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field')[1]) == 0
    assert (I.getNodeFromPath(dist_zone, 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1] == [4,3,2,1]).all()
