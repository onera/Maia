import pytest
import pytest_parallel

import numpy as np

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia import npy_pdm_gnum_dtype as pdm_dtype
import maia.utils.py_utils as py_utils
from   maia.pytree.yaml   import parse_yaml_cgns

from maia.transfer.dist_to_part import index_exchange as IBTP

dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

def test_collect_distributed_pl():
  zone   = PT.new_Zone(type='Unstructured')
  zoneBC = PT.new_node('ZoneBC', 'ZoneBC_t', parent=zone)
  point_lists = [np.array([[2,4,6,8]]          , np.int32),
                 np.array([[10,20,30,40,50,60]], np.int32),
                 np.array([[100]]              , np.int32),
                 np.empty((1,0)                , np.int32)]
  point_ranges = [np.array([[35,55]]           , np.int32)]
  for i, pl in enumerate(point_lists):
    PT.new_BC('bc'+str(i+1), point_list=pl, parent=zoneBC)
  for i, pr in enumerate(point_ranges):
    bc = PT.new_BC('bc'+str(i+1+len(point_lists)), point_range=pr, parent=zoneBC)
    distri = MT.newDistribution({'Index' : np.array([10,15,20])}, parent=bc)

  collected = IBTP.collect_distributed_pl(zone, [['ZoneBC_t', 'BC_t']])
  assert len(collected) == len(point_lists) + 1
  for i in range(len(point_lists)):
    assert (collected[i] == point_lists[i]).all()
  assert (collected[-1] == np.arange(35+10, 35+15)).all()

  zone   = PT.new_Zone(type='Structured', size=[[3,2,0], [3,2,0], [3,2,0]])
  zoneBC = PT.new_node('ZoneBC', 'ZoneBC_t', parent=zone)
  PT.new_BC('bc', loc='IFaceCenter', \
      point_list=np.array([[1,1,1,1],[1,1,2,2], [1,2,1,2]]), parent=zoneBC)
  collected = IBTP.collect_distributed_pl(zone, [['ZoneBC_t', 'BC_t']])
  assert np.array_equal(collected[0], [[1,7,4,10]])

@pytest_parallel.mark.parallel(2)
def test_dist_pl_to_part_pl(comm):
  if comm.Get_rank() == 0:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  ZoneType ZoneType_t "Unstructured":
  Quad Elements_t [7,0]:
    ElementRange IndexRange_t [1,12]:
  Hexa Elements_t [17,0]:
    ElementRange IndexRange_t [13,16]:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[1, 7]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,2,6]:
      BCDSWithPL BCDataSet_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[10]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {0} [0,1,1]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[15]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {0} [0,1,3]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[6]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [0,1,2]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [0,3,6]:
""".format(dtype)
    pt = """
  ZoneU.P0.N0 Zone_t [[4,0,0]]:
    ZoneType ZoneType_t "Unstructured":
    Hexa Elements_t [17,0]:
      ElementRange IndexRange_t [1,4]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Element DataArray_t {0} [1,3,4,2]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [6,3,4,2]:
    """.format(dtype)
  elif comm.Get_rank() == 1:
    dt = """
ZoneU Zone_t [[6,0,0]]:
  ZoneType ZoneType_t "Unstructured":
  Quad Elements_t [7,0]:
    ElementRange IndexRange_t [1,12]:
  Hexa Elements_t [17,0]:
    ElementRange IndexRange_t [13,16]:
  ZBC ZoneBC_t:
    BC BC_t:
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[12, 5, 8, 2]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [2,6,6]:
      BCDSWithPL BCDataSet_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[]]:
        :CGNS#Distribution UserDefinedData_t:
          Index DataArray_t {0} [1,1,1]:
  :CGNS#Distribution UserDefinedData_t:
    Vertex DataArray_t {0} [3,6,6]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[14,16]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t {0} [1,3,3]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[2]]:
    :CGNS#Distribution UserDefinedData_t:
      Index DataArray_t [1,2,2]:
""".format(dtype)
    pt = """
  ZoneU.P1.N0 Zone_t [[3,0,0]]:
    ZoneType ZoneType_t "Unstructured":
    Quad Elements_t [7,0]:
      ElementRange IndexRange_t [1,5]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Element DataArray_t {0} [10,5,3,2,9]:
    :CGNS#GlobalNumbering UserDefinedData_t:
      Vertex DataArray_t {0} [1,5,2]:
  """.format(dtype)

  dist_zone  = parse_yaml_cgns.to_node(dt)
  part_zones = parse_yaml_cgns.to_nodes(pt)

  IBTP.dist_pl_to_part_pl(dist_zone, part_zones, ['ZoneSubRegion_t'], 'Vertex', comm)

  part_zsr = PT.get_child_from_name(part_zones[0], 'ZSRWithPL')
  assert part_zsr is not None
  assert PT.Subset.GridLocation(part_zsr) == 'Vertex'
  if comm.Get_rank() == 0:
    assert (PT.get_child_from_name(part_zsr, 'PointList')[1] == [1,4]).all()
  if comm.Get_rank() == 1:
    assert (PT.get_child_from_name(part_zsr, 'PointList')[1] == [3]).all()

  IBTP.dist_pl_to_part_pl(dist_zone, part_zones, ['FlowSolution_t', 'ZoneBC_t/BC_t/BCDataSet_t'], 'Elements', comm)

  part_sol = PT.get_child_from_name(part_zones[0], 'FlowSolWithPL')
  part_bc  = PT.get_node_from_name(part_zones[0], 'BC')
  part_ds  = PT.get_node_from_name(part_zones[0], 'BCDSWithPL')
  if comm.Get_rank() == 0:
    assert part_bc is None
    assert PT.Subset.GridLocation(part_sol) == 'CellCenter'
    assert (PT.get_child_from_name(part_sol, 'PointList')[1] == [2,3,4]).all()
    assert (PT.get_value(MT.getGlobalNumbering(part_sol, 'Index')) == [1,3,2]).all()
  if comm.Get_rank() == 1:
    assert part_sol is None
    assert PT.get_child_from_name(part_bc, 'PointList') is None #No specified in list => skipped, only child are constructed
    assert PT.Subset.GridLocation(part_ds) == 'FaceCenter'
    assert (PT.get_child_from_name(part_ds, 'PointList')[1] == [1]).all()
    assert (PT.get_value(MT.getGlobalNumbering(part_ds, 'Index')) == [1]).all()

  with pytest.raises(ValueError):
    IBTP.dist_pl_to_part_pl(dist_zone, part_zones, ['FlowSolution_t'], 'FaceCenter', comm)


def test_create_part_pointlists():
  dt = """
ZoneU Zone_t [[6,0,0]]:
  ZoneType ZoneType_t "Unstructured":
  ZBC ZoneBC_t:
    BC BC_t "BCFarfield":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[12, 5, 8, 2]]:
      BCDSWithPL BCDataSet_t:
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [[]]:
  FlowSolWithPL FlowSolution_t:
    GridLocation GridLocation_t "CellCenter":
    PointList IndexArray_t [[14,16]]:
  ZSRWithPL ZoneSubRegion_t:
    GridLocation GridLocation_t "Vertex":
    PointList IndexArray_t [[2]]:
""".format(dtype)
  pt = """
ZoneU.P1.N0 Zone_t [[3,0,0]]:
  ZoneType ZoneType_t "Unstructured":
""".format(dtype)

  dist_zone = parse_yaml_cgns.to_node(dt)
  part_zone = parse_yaml_cgns.to_node(pt)

  group_part = {'npZSRGroupIdx': np.array([0, 0, 1], dtype=np.int32),
                'npZSRGroup': np.array([42], dtype=np.int32),
                'npZSRGroupLNToGN': np.array([9], dtype=np.int32)}

  IBTP.create_part_pointlists(dist_zone, part_zone,\
      group_part, ['FlowSolution_t', 'ZoneBC_t/BC_t/BCDataSet_t'], ['FaceCenter', 'CellCenter'])

  assert PT.get_child_from_name(part_zone, 'FlowSolWithPL') is None
  part_bc  = PT.get_node_from_name(part_zone, 'BC')
  assert PT.get_value(part_bc) == "BCFarfield"
  part_ds  = PT.get_node_from_name(part_zone, 'BCDSWithPL')
  assert PT.get_child_from_name(part_bc, 'PointList') is None
  assert PT.Subset.GridLocation(part_ds) == 'FaceCenter'
  assert (PT.get_child_from_name(part_ds, 'PointList')[1] == [42]).all()
  assert (PT.get_value(MT.getGlobalNumbering(part_ds, 'Index')) == [9]).all()
