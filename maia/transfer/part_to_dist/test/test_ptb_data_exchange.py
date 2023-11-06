import pytest
import pytest_parallel
import numpy      as np

import maia.pytree      as PT
import maia.pytree.maia as MT


from maia import npy_pdm_gnum_dtype as pdm_dtype
from   maia.transfer import part_to_dist
import maia.transfer.part_to_dist.data_exchange as PTB
import maia.transfer.protocols as EP
from   maia.pytree.yaml   import parse_yaml_cgns

dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

@pytest_parallel.mark.parallel(2)
def test_lngn_to_distri(comm):
  if comm.rank == 0:
    lngn_list = []
    expt_distri = [0,2,4]
  if comm.rank == 1:
    lngn_list = [np.array([2,3,1,4], pdm_dtype)]
    expt_distri = [2,4,4]
  assert (PTB._lngn_to_distri(lngn_list, comm) == expt_distri).all()

  if comm.rank == 0:
    lngn_list = [np.empty(0, pdm_dtype), np.array([4,3,1,10], pdm_dtype)]
    expt_distri = [0,5,10]
  if comm.rank == 1:
    lngn_list = [np.array([2,3,1,1,5,4], pdm_dtype)]
    expt_distri = [5,10,10]
  assert (PTB._lngn_to_distri(lngn_list, comm) == expt_distri).all()


@pytest_parallel.mark.parallel(3)
class Test__discover_wrapper:
  def test_sol_without_pl(self, comm):
    dist_zone  = PT.new_Zone('Zone', type = 'Unstructured')
    part_zones = [PT.new_Zone('Zone.P{0}.N0'.format(comm.Get_rank()))]
    if comm.Get_rank() == 0:
      p_sol = PT.new_FlowSolution('NewSol1', loc='CellCenter', parent=part_zones[0])
      PT.new_DataArray('NewField1', None, parent=p_sol)
      PT.new_DataArray('NewField2', None, parent=p_sol)
    if comm.Get_rank() == 2:
      part_zones.append(PT.new_Zone('Zone.P{0}.N1'.format(comm.Get_rank())))
      p_sol = PT.new_FlowSolution('NewSol2', loc='Vertex', parent=part_zones[1])
      PT.new_DataArray('NewField3', None, parent=p_sol)
      PT.new_DataArray('NewField4', None, parent=p_sol)
    p_sol = PT.new_FlowSolution('NewSol3', loc='Vertex', parent=part_zones[0])
    PT.new_DataArray('NewField5', None, parent=p_sol)

    PTB._discover_wrapper(dist_zone, part_zones, 'FlowSolution_t', 'FlowSolution_t/DataArray_t', comm)

    assert PT.get_names(PT.get_children_from_label(dist_zone, 'FlowSolution_t')) == ['NewSol1', 'NewSol3', 'NewSol2']
    assert [PT.Subset.GridLocation(sol) for sol in PT.get_children_from_label(dist_zone, 'FlowSolution_t')] \
        == ['CellCenter', 'Vertex', 'Vertex']
    assert PT.get_node_from_path(dist_zone, 'NewSol2/NewField4') is not None

  def test_sol_with_pl(self, comm):
    dt = """
  Zone Zone_t:
    ZoneType ZoneType_t "Unstructured":
    Hexa Elements_t [17,0]:
      ElementRange IndexRange_t [1,50]:
    """
    if comm.Get_rank() == 0:
      pt = """
    Zone.P0.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      Hexa Elements_t [17,0]:
        ElementRange IndexRange_t [1,5]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Element DataArray_t {0} [29,42,19,18,41]:
      FS DiscreteData_t:
        GridLocation GridLocation_t "CellCenter":
        PointList IndexArray_t [[4,2]]:
        field DataArray_t:
      """.format(dtype)
    if comm.Get_rank() == 1:
      pt = ""
    if comm.Get_rank() == 2:
      pt = """
    Zone.P2.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
    Zone.P2.N1 Zone_t:
      ZoneType ZoneType_t "Unstructured":
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
        'DiscreteData_t', 'DiscreteData_t/DataArray_t', comm)

    fs = PT.get_child_from_name(dist_zone, 'FS')
    assert PT.get_label(fs) == 'DiscreteData_t'
    dist_pl     = PT.get_node_from_path(fs, 'PointList')[1]
    dist_distri = PT.get_value(MT.getDistribution(fs, 'Index'))
    assert dist_distri.dtype == pdm_dtype

    if comm.Get_rank() == 0:
      assert (dist_distri == [0,2,4]).all()
      assert (dist_pl     == [8,18] ).all()
    elif comm.Get_rank() == 1:
      assert (dist_distri == [2,3,4]).all()
      assert (dist_pl     == [25]   ).all()
    elif comm.Get_rank() == 2:
      assert (dist_distri == [3,4,4]).all()
      assert (dist_pl     == [42]   ).all()

  def test_zsr(self, comm):
    dt = """
  Zone Zone_t:
    ZoneType ZoneType_t "Unstructured":
    InitialZSR ZoneSubRegion_t:
      GridConnectivityRegionName Descriptor_t "match":
    """
    if comm.Get_rank() == 0:
      pt = """
    Zone.P0.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      CreatedZSR.0 ZoneSubRegion_t:
        GridConnectivityRegionName Descriptor_t "gc.0":
      """
    elif comm.Get_rank() == 1:
      pt = """
    Zone.P1.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      InitialZSR.0 ZoneSubRegion_t:
        GridConnectivityRegionName Descriptor_t "gc.0":
      """
    if comm.Get_rank() == 2:
      pt = """
    Zone.P2.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      CreatedZSR.0 ZoneSubRegion_t:
        GridConnectivityRegionName Descriptor_t "gc.0":
      CreatedZSR.1 ZoneSubRegion_t:
        GridConnectivityRegionName Descriptor_t "gc.1":
      """
    dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
    part_tree = parse_yaml_cgns.to_cgns_tree(pt)

    PTB._discover_wrapper(PT.get_all_Zone_t(dist_tree)[0], PT.get_all_Zone_t(part_tree), \
        'ZoneSubRegion_t', 'DataArray_t', comm)
    dist_zsr = PT.get_nodes_from_label(dist_tree, 'ZoneSubRegion_t')
    assert [PT.get_name(n) for n in dist_zsr] \
        == ['InitialZSR', 'CreatedZSR']
    assert [PT.get_value(PT.get_node_from_label(n, 'Descriptor_t')) for n in dist_zsr] \
        == ['match', 'gc']

  def test_dataset(self, comm):
    dt = """
  Zone Zone_t:
    ZoneType ZoneType_t "Unstructured":
    ZBC ZoneBC_t:
      bc1 BC_t:
      bc2 BC_t:
    """
    if comm.Get_rank() == 0:
      pt = """
    Zone.P0.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      ZBC ZoneBC_t:
        bc1 BC_t:
          BCDS BCDataSet_t 'BCInflow':
            BCData BCData_t:
              newField1 DataArray_t:
              newField2 DataArray_t:
      """
    elif comm.Get_rank() == 1:
      pt = """
    Zone.P1.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      """
    if comm.Get_rank() == 2:
      pt = """
    Zone.P2.N0 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      ZBC ZoneBC_t:
        bc1 BC_t:
          BCDS2 BCDataSet_t 'BCWall':
            BCData BCData_t:
              newField3 DataArray_t:
    Zone.P2.N1 Zone_t:
      ZoneType ZoneType_t "Unstructured":
      ZBC ZoneBC_t:
        bc2 BC_t:
          BCDS BCDataSet_t 'Null':
            BCData BCData_t:
              newField4 DataArray_t:
      """
    dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
    part_tree = parse_yaml_cgns.to_cgns_tree(pt)

    bc_ds_path = 'ZoneBC_t/BC_t/BCDataSet_t'
    PTB._discover_wrapper(PT.get_all_Zone_t(dist_tree)[0], PT.get_all_Zone_t(part_tree), \
        bc_ds_path, bc_ds_path+'/BCData_t/DataArray_t', comm)

    assert [PT.get_value(bcds) for bcds in PT.get_nodes_from_label(dist_tree, 'BCDataSet_t')] \
        == ['BCInflow', 'BCWall', 'Null']
    assert [PT.get_name(field) for field in PT.get_nodes_from_label(dist_tree, 'DataArray_t')] \
        == ['newField1', 'newField2', 'newField3', 'newField4']

@pytest_parallel.mark.parallel(2)
def test_part_coords_to_dist_coords(comm):
  if comm.Get_rank() == 0:
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
  elif comm.Get_rank() == 1:
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

  PTB.part_coords_to_dist_coords(dist_zone, part_zones, comm)

  if comm.Get_rank() == 0:
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CX')[1] == [1,6]).all()
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CY')[1] == [2,1]).all()
  elif comm.Get_rank() == 1:
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CX')[1] == [5,2]).all()
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CY')[1] == [1,2]).all()
    assert (PT.get_node_from_path(part_zones[1], 'GridCoordinates/CX')[1] == [3,4]).all()
    assert (PT.get_node_from_path(part_zones[1], 'GridCoordinates/CY')[1] == [2,1]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("filter", [False, True])
def test_part_sol_to_dist_sol(comm, filter):
  if comm.Get_rank() == 0:
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
      field3 DataArray_t R8 [0,0,0]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [3,4,1]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
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
      field3 DataArray_t R8 [-1,-1,-1]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,6,2]:
  """.format(dtype)

  dist_zone  = parse_yaml_cgns.to_node(dt)
  part_zones = parse_yaml_cgns.to_nodes(pt)

  if filter:
    part_to_dist.part_zones_to_dist_zone_only(dist_zone, part_zones, comm, {'DiscreteData_t' : ['NewFlowSol/field3']})
  else:
    PTB.part_sol_to_dist_sol(dist_zone, part_zones, comm)
    PTB.part_discdata_to_dist_discdata(dist_zone, part_zones, comm)

  if filter:
    assert PT.get_node_from_path(dist_zone, 'NewFlowSol/field2') is None
  else:
    assert PT.get_node_from_path(dist_zone, 'FlowSolWithPL/field1')[1].dtype == np.int32
  assert PT.get_node_from_path(dist_zone, 'NewFlowSol/field3')[1].dtype == np.float64
  if comm.Get_rank () == 0:
    if not filter:
      assert (PT.get_node_from_path(dist_zone, 'FlowSolWithPL/field1')[1] == [-30]).all()
    assert (PT.get_node_from_path(dist_zone, 'NewFlowSol/field3')[1] == [0,-1,0]).all()
  if comm.Get_rank () == 1:
    if not filter:
      assert (PT.get_node_from_path(dist_zone, 'FlowSolWithPL/field1')[1] == [-10, -20]).all()
    assert (PT.get_node_from_path(dist_zone, 'NewFlowSol/field3')[1] == [0,-1,-1]).all()

@pytest_parallel.mark.parallel(2)
def test_part_sol_to_dist_sol_with_reduce_func(comm):
  if comm.Get_rank() == 0:
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
      PointList IndexArray_t [[1,8,10]]:
      field1 DataArray_t [0,-10,-15]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [1,2,1]:
    NewFlowSol DiscreteData_t:
      GridLocation GridLocation_t "Vertex":
      field2 DataArray_t R8 [0,0,0]:
      field3 DataArray_t R8 [0,0,0]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [3,4,1]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
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
      field2 DataArray_t R8 [1,1,1,1]:
      field3 DataArray_t R8 [-1,-1,-1,-1]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,6,2,1]:
  """.format(dtype)

  dist_zone  = parse_yaml_cgns.to_node(dt)
  part_zones = parse_yaml_cgns.to_nodes(pt)

  PTB.part_sol_to_dist_sol(dist_zone, part_zones, comm,reduce_func=EP.reduce_sum)
  PTB.part_discdata_to_dist_discdata(dist_zone, part_zones, comm,reduce_func=EP.reduce_mean)

  assert PT.get_node_from_path(dist_zone, 'FlowSolWithPL/field1')[1].dtype == np.int64
  assert PT.get_node_from_path(dist_zone, 'NewFlowSol/field2')[1].dtype == np.float64
  assert PT.get_node_from_path(dist_zone, 'NewFlowSol/field3')[1].dtype == np.float64
  if comm.Get_rank() == 0:
    assert (PT.get_node_from_path(dist_zone, 'FlowSolWithPL/field1')[1] == [(0-15.-30.)]).all()
    assert (PT.get_node_from_path(dist_zone, 'NewFlowSol/field2')[1] == [(0.+1.)/2.,1,0]).all()
    assert (PT.get_node_from_path(dist_zone, 'NewFlowSol/field3')[1] == [(0.-1.)/2.,-1,0]).all()
  if comm.Get_rank() == 1:
    assert (PT.get_node_from_path(dist_zone, 'FlowSolWithPL/field1')[1] == [-10, -20]).all()
    assert (PT.get_node_from_path(dist_zone, 'NewFlowSol/field2')[1] == [0,1,1]).all()
    assert (PT.get_node_from_path(dist_zone, 'NewFlowSol/field3')[1] == [0,-1,-1]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("from_api", [False, True])
def test_part_subregion_to_dist_subregion(comm, from_api):
  if comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[19, 23, 11]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,3,6]:
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[18, 22]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,2,6]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[6]]:
    field DataArray_t I4 [42]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [0,1,2]:
  LinkedZSR ZoneSubRegion_t:
    GridLocation GridLocation_t "FaceCenter":
    GridConnectivityRegionName Descriptor_t "GC":
    field DataArray_t:
  """.format(dtype)
    pt = """
  ZoneU.P0.N0 Zone_t [[3,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[11, 13, 15, 12, 14, 16]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [1,3,5,2,4,6]:
    ZGC ZoneGridConnectivity_t:
      GC.0 GridConnectivity_t:
        PointList IndexArray_t [[1, 12, 21]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [2,5,1]:
    LinkedZSR.0 ZoneSubRegion_t:
      GridLocation GridLocation_t "FaceCenter":
      GridConnectivityRegionName Descriptor_t "GC.0":
      field DataArray_t R8 [200,500,100]:
    CreatedZSR ZoneSubRegion_t:
      GridLocation GridLocation_t "FaceCenter":
      BCRegionName Descriptor_t "BC":
      field DataArray_t R8 [111., 113., 115., 112., 114., 116.]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[40, 42, 10]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [3,6,6]:
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[13, 39, 41, 9]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [2,6,6]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[2]]:
    field DataArray_t I4 [24]:
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
      GC.0 GridConnectivity_t:
        PointList IndexArray_t [[1, 108]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [6,4]:
      GC.1 GridConnectivity_t:
        PointList IndexArray_t [[29]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [3]:
    ZSRWithPL ZoneSubRegion_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[1,2]]:
      field DataArray_t I4 [84,48]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [1,2]:
    LinkedZSR.0 ZoneSubRegion_t:
      GridLocation GridLocation_t "FaceCenter":
      GridConnectivityRegionName Descriptor_t "GC.0":
      field DataArray_t R8 [600,400]:
    LinkedZSR.1 ZoneSubRegion_t:
      GridLocation GridLocation_t "FaceCenter":
      GridConnectivityRegionName Descriptor_t "GC.1":
      field DataArray_t R8 [300]:
  """.format(dtype)

  dist_zone  = parse_yaml_cgns.to_node(dt)
  part_zones = parse_yaml_cgns.to_nodes(pt)

  if from_api:
    part_to_dist.part_zones_to_dist_zone_all(dist_zone, part_zones, comm)
  else:
    PTB.part_subregion_to_dist_subregion(dist_zone, part_zones, comm)

  assert PT.get_node_from_path(dist_zone, 'ZSRWithPL/field')[1].dtype == np.int32
  assert PT.get_node_from_path(dist_zone, 'LinkedZSR/field') [1].dtype == np.float64
  assert PT.get_node_from_path(dist_zone, 'CreatedZSR/field')[1].dtype == np.float64
  if comm.Get_rank () == 0:
    assert (PT.get_node_from_path(dist_zone, 'ZSRWithPL/field')[1] == [84]).all()
    assert (PT.get_node_from_path(dist_zone, 'LinkedZSR/field')[1] == [100,200]).all()
    assert (PT.get_node_from_path(dist_zone, 'CreatedZSR/field')[1] == [111., 112., 113.]).all()
  if comm.Get_rank () == 1:
    assert (PT.get_node_from_path(dist_zone, 'ZSRWithPL/field')[1] == [48]).all()
    assert (PT.get_node_from_path(dist_zone, 'LinkedZSR/field')[1] == [300,400,500,600]).all()
    assert (PT.get_node_from_path(dist_zone, 'CreatedZSR/field')[1] == [114., 115., 116.]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("from_api", [False, True])
def test_part_dataset_to_dist_dataset(comm, from_api):
  if comm.Get_rank() == 0:
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
          field DataArray_t R8 [100]:
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
            field DataArray_t [2,2]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
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
          field DataArray_t R8 []:
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
            field DataArray_t R8 [200.]:
          PointList IndexArray_t [[108]]:
          :CGNS#GlobalNumbering UserDefinedData_t:
            Index DataArray_t {0} [1]:
        BCDSWithoutPL BCDataSet_t:
          DirichletData BCData_t:
            field DataArray_t [1,4,3,1]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)
  dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
  part_zones = PT.get_all_Zone_t(part_tree)

  if from_api:
    part_to_dist.part_tree_to_dist_tree_only_labels(dist_tree, part_tree, ['BCDataSet_t'], comm)
  else:
    PTB.part_dataset_to_dist_dataset(dist_zone, part_zones, comm)

  assert PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field')[1].dtype    == np.float64
  assert PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1].dtype == np.int32
  if comm.Get_rank () == 0:
    assert (PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field')[1] == [200.]).all()
    assert (PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1] == [1,2]).all()
  if comm.Get_rank () == 1:
    assert len(PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field')[1]) == 0
    assert (PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1] == [4,3,2,1]).all()

@pytest_parallel.mark.parallel(2)
def test_part_dataset_to_dist_dataset_filter(comm):
  if comm.Get_rank() == 0:
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
          field DataArray_t [-1, -1]:
      BCDSWithPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t R8 [100]:
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
            field DataArray_t [2,2]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
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
          field DataArray_t [-1,-1,-1,-1]:
      BCDSWithPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t R8 []:
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
            field DataArray_t R8 [200.]:
            field2 DataArray_t R8 [200.]:
            field3 DataArray_t R8 [200.]:
          PointList IndexArray_t [[108]]:
          :CGNS#GlobalNumbering UserDefinedData_t:
            Index DataArray_t {0} [1]:
        BCDSWithoutPL BCDataSet_t:
          DirichletData BCData_t:
            field DataArray_t [1,4,3,1]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)
  dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
  part_zones = PT.get_all_Zone_t(part_tree)

  PTB.part_dataset_to_dist_dataset(dist_zone, part_zones, comm, \
      exclude=['*/BCDSWithPL/*/field', '*/BCDSWithPL/*/field2'])

  assert PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field')  is not None
  assert PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field2') is None
  assert PT.get_node_from_path(dist_zone, 'ZBC/BC/BCDSWithPL/DirichletData/field3') is not None
