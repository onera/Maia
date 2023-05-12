import pytest
import numpy      as np
import maia.pytree        as PT

import pytest_parallel

from   maia.pytree.yaml   import parse_yaml_cgns
from   maia.transfer import dist_to_part
import maia.transfer.dist_to_part.data_exchange as BTP
from maia import npy_pdm_gnum_dtype as pdm_dtype

dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

dt0 = """
ZoneU Zone_t [[6,0,0]]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t [1,2,3]:
    CY DataArray_t [2,2,2]:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[18, 22]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,2,6]:
      BCDSWithoutPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t [1,2]:
      BCDSWithPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t R8 [100]:
        PointList IndexArray_t [[10]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {0} [0,1,1]:
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t "GhostZone":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t I4 [1,3]:
      PointListDonor IndexArray_t I4 [102,104]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t I4 [0,2,3]:
  FlowSolution FlowSolution_t:
    GridLocation GridLocation_t "Vertex":
    field1 DataArray_t I4 [0,0,0]:
    field2 DataArray_t R8 [6.,5.,4.]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[2]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {0} [0,1,3]:
    field1 DataArray_t [10]:
  ZSRWithoutPL ZoneSubRegion_t:
    GridLocation GridLocation_t "FaceCenter":
    BCRegionName Descriptor_t "BC":
    field DataArray_t R8 [100,200]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[6]]:
    field DataArray_t I4 [42]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [0,1,2]:
  ZSRWithGC ZoneSubRegion_t:
    GridConnectivityRegionName Descriptor_t "GC":
    fieldGC DataArray_t R8 [1., 9.]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [0,3,6]:
ZoneS Zone_t [[2,0,0],[3,0,0],[1,0,0]]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t [1,2,3]:
    CY DataArray_t [2,2,2]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [0,3,6]:
""".format(dtype)

dt1 = """
ZoneU Zone_t [[6,0,0]]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t [4,5,6]:
    CY DataArray_t [1,1,1]:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[13, 39, 41, 9]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [2,6,6]:
      BCDSWithoutPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t [4,3,2,1]:
      BCDSWithPL BCDataSet_t:
        DirichletData BCData_t:
          field DataArray_t R8 []:
        PointList IndexArray_t [[]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {0} [1,1,1]:
  ZGC ZoneGridConnectivity_t:
    GC GridConnectivity_t "GhostZone":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t I4 [5]:
      PointListDonor IndexArray_t I4 [106]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t I4 [2,3,3]:
  FlowSolution FlowSolution_t:
    GridLocation GridLocation_t "Vertex":
    field1 DataArray_t I4 [1,1,1]:
    field2 DataArray_t R8 [3.,2.,1.]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[6,4]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {0} [1,3,3]:
    field1 DataArray_t [20,30]:
  ZSRWithoutPL ZoneSubRegion_t:
    GridLocation GridLocation_t "FaceCenter":
    BCRegionName Descriptor_t "BC":
    field DataArray_t R8 [300,400,500,600]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[2]]:
    field DataArray_t I4 [24]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [1,2,2]:
  ZSRWithGC ZoneSubRegion_t:
    GridConnectivityRegionName Descriptor_t "GC":
    fieldGC DataArray_t R8 [25.]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,6]:
ZoneS Zone_t [[2,0,0],[3,0,0],[1,0,0]]:
  GridCoordinates GridCoordinates_t:
    CX DataArray_t [4,5,6]:
    CY DataArray_t [1,1,1]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,6]:
""".format(dtype)


@pytest_parallel.mark.parallel(2)
def test_dist_coords_to_part_coords_U(comm):
  if comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [1,6]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,2]:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [3,4]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
  part_zones = PT.get_all_Zone_t(part_tree)
  BTP.dist_coords_to_part_coords(dist_zone, part_zones, comm)

  if comm.Get_rank() == 0:
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CX')[1] == [1,6]).all()
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CY')[1] == [2,1]).all()
  elif comm.Get_rank() == 1:
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CX')[1] == [5,2]).all()
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CY')[1] == [1,2]).all()
    assert (PT.get_node_from_path(part_zones[1], 'GridCoordinates/CX')[1] == [3,4]).all()
    assert (PT.get_node_from_path(part_zones[1], 'GridCoordinates/CY')[1] == [2,1]).all()

@pytest_parallel.mark.parallel(2)
def test_dist_coords_to_part_coords_S(comm):
  if comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneS.P0.N0 Zone_t [[2,0,0],[1,0,0],[1,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [1,2]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneS.P1.N0 Zone_t [[2,0,0],[2,0,0],[1,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,6,3,4]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = PT.get_all_Zone_t(dist_tree)[1]
  part_zones = PT.get_all_Zone_t(part_tree)
  BTP.dist_coords_to_part_coords(dist_zone, part_zones, comm)

  if comm.Get_rank() == 0:
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CX')[1] == \
        np.array([1,2]).reshape((2,1,1), order='F')).all()
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CY')[1] == \
        np.array([2,2]).reshape((2,1,1), order='F')).all()
  elif comm.Get_rank() == 1:
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CX')[1] == \
        np.array([5,6,3,4]).reshape((2,2,1),order='F')).all()
    assert (PT.get_node_from_path(part_zones[0], 'GridCoordinates/CY')[1] == \
        np.array([1,1,2,1]).reshape((2,2,1),order='F')).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("include", [["FlowSolution/field1"], ["FlowSolution/field*", "*/field2"], []])
def test_dist_sol_to_part_sol_allvtx(comm, include):
  if comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [1,6]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [5,2]:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [3,4]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
  part_zones = PT.get_all_Zone_t(part_tree)
  BTP.dist_sol_to_part_sol(dist_zone, part_zones, comm, include=include)

  should_have_field2 = include != ["FlowSolution/field1"]
  for zone in part_zones:
    assert PT.get_node_from_path(zone, 'FlowSolution/field1')[1].dtype == np.int32
    if should_have_field2:
      assert PT.get_node_from_path(zone, 'FlowSolution/field2')[1].dtype == np.float64
    else:
      assert PT.get_node_from_path(zone, 'FlowSolution/field2') is None
  if comm.Get_rank() == 0:
    assert (PT.get_node_from_path(part_zones[0], 'FlowSolution/field1')[1] == [0,1]).all()
    if should_have_field2:
      assert (PT.get_node_from_path(part_zones[0], 'FlowSolution/field2')[1] == [6,1]).all()
  elif comm.Get_rank() == 1:
    assert (PT.get_node_from_path(part_zones[0], 'FlowSolution/field1')[1] == [1,0]).all()
    assert (PT.get_node_from_path(part_zones[1], 'FlowSolution/field1')[1] == [0,1]).all()
    if should_have_field2:
      assert (PT.get_node_from_path(part_zones[0], 'FlowSolution/field2')[1] == [2,5]).all()
      assert (PT.get_node_from_path(part_zones[1], 'FlowSolution/field2')[1] == [4,3]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("exclude", [[], ["*/field1"]])
def test_dist_sol_to_part_sol_pl(comm, exclude):
  if comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    FlowSolWithPL FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[1]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [2]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    FlowSolWithPL FlowSolution_t:
      GridLocation GridLocation_t "CellCenter":
      PointList IndexArray_t [[12,15]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [3,1]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  PT.rm_nodes_from_name(dist_tree, 'FlowSolution') #Test only pl sol here
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
  part_zones = PT.get_all_Zone_t(part_tree)
  BTP.dist_sol_to_part_sol(dist_zone, part_zones, comm, exclude=exclude)

  if exclude == []:
    if comm.Get_rank() == 0:
      assert (PT.get_node_from_path(part_zones[0], 'FlowSolWithPL/field1')[1] == [20]).all()
    elif comm.Get_rank() == 1:
      assert PT.get_node_from_path(part_zones[0], 'FlowSolWithPL/field1') is None
      assert PT.get_node_from_path(part_zones[1], 'FlowSolWithPL/field1')[1].shape == (2,)
      assert (PT.get_node_from_path(part_zones[1], 'FlowSolWithPL/field1')[1] == [30,10]).all()
  else:
    assert PT.get_node_from_name(part_tree, 'field1') is None

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("from_api", [False, True])
def test_dist_dataset_to_part_dataset(comm, from_api):
  if comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 12, 21]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [2,5,1]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[1, 29, 108]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [6,3,4]:
        BCDSWithPL BCDataSet_t:
          PointList IndexArray_t [[108]]:
          :CGNS#GlobalNumbering UserDefinedData_t:
            Index DataArray_t {0} [1]:
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
  part_zones = PT.get_all_Zone_t(part_tree)
  if from_api:
    dist_to_part.dist_tree_to_part_tree_only_labels(dist_tree, part_tree, ["BCDataSet_t"], comm)
  else:
    BTP.dist_dataset_to_part_dataset(dist_zone, part_zones, comm)

  if comm.Get_rank() == 0:
    assert (PT.get_node_from_path(part_zones[0], 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1] == [2,2,1]).all()
    assert PT.get_node_from_path(part_zones[0], 'ZBC/BC/BCDSWitPL/DirichletData/field') is None
  elif comm.Get_rank() == 1:
    assert PT.get_node_from_path(part_zones[0], 'ZBC/BC') is None
    assert (PT.get_node_from_path(part_zones[1], 'ZBC/BC/BCDSWithoutPL/DirichletData/field')[1] == [1,4,3]).all()
    assert (PT.get_node_from_path(part_zones[1], 'ZBC/BC/BCDSWithPL/DirichletData/field')[1] == [100.]).all()

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("api_mode", [None, "ZoneAll", "ZoneOnly", "Tree"])
def test_dist_subregion_to_part_subregion(comm, api_mode):
  if comm.Get_rank() == 0:
    dt = dt0
    pt = """
  ZoneU.P0.N0 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        PointList IndexArray_t [[1, 12, 21]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [2,5,1]:
    ZGC ZoneGridConnectivity_t:
      GC.0 GridConnectivity_t "GhostZone.P0.N0":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t I4 [2]:
        PointListDonor IndexArray_t I4 [202]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t I4 [2]:
    ZSRWithGC.0 ZoneSubRegion_t:
      GridConnectivityRegionName Descriptor_t "GC.0":
    """.format(dtype)
  elif comm.Get_rank() == 1:
    dt = dt1
    pt = """
  ZoneU.P1.N0 Zone_t [[2,0,0]]:
    ZSRWithPL ZoneSubRegion_t:
      GridLocation GridLocation_t "Vertex":
      PointList IndexArray_t [[1,2]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [1,2]:
    ZGC ZoneGridConnectivity_t:
      GC.0 GridConnectivity_t "GhostZone.P1.N0":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t I4 [1]:
        PointListDonor IndexArray_t I4 [206]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t I4 [1]:
      GC.1 GridConnectivity_t "GhostZone.P0.N1":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t I4 [3]:
        PointListDonor IndexArray_t I4 [204]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t I4 [3]:
    ZSRWithGC.0 ZoneSubRegion_t:
      GridConnectivityRegionName Descriptor_t "GC.0":
    ZSRWithGC.1 ZoneSubRegion_t:
      GridConnectivityRegionName Descriptor_t "GC.1":
  ZoneU.P1.N1 Zone_t [[2,0,0]]:
    ZBC ZoneBC_t:
      BC BC_t:
        PointList IndexArray_t [[1, 29, 108]]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t {0} [6,3,4]:
    ZGC ZoneGridConnectivity_t:
      GC.0 GridConnectivity_t "GhostZone.P1.N0":
        GridConnectivityType GridConnectivityType_t "Abutting1to1":
        GridLocation GridLocation_t "Vertex":
        PointList IndexArray_t I4 [1]:
        PointListDonor IndexArray_t I4 [206]:
        :CGNS#GlobalNumbering UserDefinedData_t:
          Index DataArray_t I4 [1]:
    ZSRWithGC.0 ZoneSubRegion_t:
      GridConnectivityRegionName Descriptor_t "GC.0":
  """.format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  dist_zone  = PT.get_all_Zone_t(dist_tree)[0]
  part_zones = [zone for zone in PT.iter_all_Zone_t(part_tree) if 'ZoneU' in PT.get_name(zone)]
  if api_mode is None:
    BTP.dist_subregion_to_part_subregion(dist_zone, part_zones, comm)
  elif api_mode == "ZoneAll":
    dist_to_part.dist_zone_to_part_zones_all(dist_zone, part_zones, comm, exclude_dict={'FlowSolution_t' : ['*']})
  elif api_mode == "ZoneOnly":
    dist_to_part.dist_zone_to_part_zones_only(dist_zone, part_zones, comm, include_dict={'ZoneSubRegion_t' : ['*']})
  elif api_mode == "Tree":
    dist_to_part.dist_tree_to_part_tree_only_labels(dist_tree, part_tree, ["ZoneSubRegion_t"], comm)
  else:
    return

  if comm.Get_rank() == 0:
    assert (PT.get_node_from_path(part_zones[0], 'ZSRWithoutPL/field')[1] == [200,500,100]).all()
    assert PT.get_node_from_path(part_zones[0], 'ZSRWithPL') is None
    assert (PT.get_node_from_path(part_zones[0], 'ZSRWithGC.0/fieldGC')[1] == [9.]).all()
  elif comm.Get_rank() == 1:
    assert PT.get_node_from_path(part_zones[0], 'ZSRWithoutPL') is None
    assert (PT.get_node_from_path(part_zones[0], 'ZSRWithPL/field')[1] == [42,24]).all()
    assert PT.get_node_from_path(part_zones[1], 'ZSRWithPL') is None
    assert (PT.get_node_from_path(part_zones[1], 'ZSRWithoutPL/field')[1] == [600,300,400]).all()
    assert (PT.get_node_from_path(part_zones[0], 'ZSRWithGC.0/fieldGC')[1] == [1.]).all()
    assert (PT.get_node_from_path(part_zones[0], 'ZSRWithGC.1/fieldGC')[1] == [25.]).all()
    assert (PT.get_node_from_path(part_zones[1], 'ZSRWithGC.0/fieldGC')[1] == [1.]).all()
