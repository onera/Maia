from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np
import Converter.Internal as I
import maia.pytree as PT
from   maia.utils.yaml   import parse_yaml_cgns
from maia.factory.partitioning.split_S import part_zone as splitS

def test_collect_S_bnd_per_dir():
  yt = """
Zone Zone_t:
  ZBC ZoneBC_t:
    bc1 BC_t:
      PointRange IndexRange_t [[1,1],[1,8],[1,6]]:
      GridLocation GridLocation_t "IFaceCenter":
    bc2 BC_t:
      PointRange IndexRange_t [[1,9],[1,1],[1,7]]:
      GridLocation GridLocation_t "Vertex":
    bc8 BC_t:
      PointRange IndexRange_t [[9,17],[1,1],[1,7]]:
      GridLocation GridLocation_t "Vertex":
    bc3 BC_t:
      PointRange IndexRange_t [[1,16],[9,9],[1,6]]:
      GridLocation GridLocation_t "JFaceCenter":
    bc4 BC_t:
      PointRange IndexRange_t [[1,16],[1,8],[1,1]]:
      GridLocation GridLocation_t "CellCenter":
    bc5 BC_t:
      PointRange IndexRange_t [[1,16],[1,8],[6,6]]:
      GridLocation GridLocation_t "CellCenter":
    bc6 BC_t:
      PointRange IndexRange_t [[17,17],[1,3],[1,7]]:
    bc7 BC_t:
      PointRange IndexRange_t [[17,17],[3,8],[5,6]]:
      GridLocation GridLocation_t "IFaceCenter":
  ZGC ZoneGridConnectivity_t:
    gc1 GridConnectivity1to1_t:
      PointRange IndexRange_t [[17,17],[3,9],[1,5]]:
"""
  zone = parse_yaml_cgns.to_node(yt)

  out = splitS.collect_S_bnd_per_dir(zone)
  assert out["xmin"] == [I.getNodeFromName(zone, name) for name in ['bc1']]
  assert out["ymin"] == [I.getNodeFromName(zone, name) for name in ['bc2', 'bc8']]
  assert out["zmin"] == [I.getNodeFromName(zone, name) for name in ['bc4']]
  assert out["xmax"] == [I.getNodeFromName(zone, name) for name in ['bc6', 'bc7', 'gc1']]
  assert out["ymax"] == [I.getNodeFromName(zone, name) for name in ['bc3']]
  assert out["zmax"] == [I.getNodeFromName(zone, name) for name in ['bc5']]

def test_intersect_pr():
  assert splitS.intersect_pr(np.array([[7,11],[1,5]]), np.array([[12,16],[1,5]])) is None
  assert (splitS.intersect_pr(np.array([[7,11],[1,5]]), np.array([[7,11],[1,5]])) \
      == np.array([[7,11],[1,5]])).all()
  assert (splitS.intersect_pr(np.array([[1,5],[1,3]]), np.array([[1,4],[3,6]])) \
      == np.array([[1,4],[3,3]])).all()

def test_zone_cell_range():
  zone = I.newZone(ztype='Structured', zsize=[[101,100,0],[101,100,0],[41,40,0]])
  assert (splitS.zone_cell_range(zone) == np.array([[1,100],[1,100],[1,40]])).all()

def test_pr_to_cell_location():
  pr = np.array([[1,1],[1,8],[1,6]])
  splitS.pr_to_cell_location(pr, 0, 'IFaceCenter', False)
  assert (pr == np.array([[1,1],[1,8],[1,6]])).all()
  pr = np.array([[10,10],[1,8],[1,6]])
  splitS.pr_to_cell_location(pr, 0, 'IFaceCenter', True)
  assert (pr == np.array([[9,9],[1,8],[1,6]])).all()
  splitS.pr_to_cell_location(pr, 0, 'IFaceCenter', True, reverse=True)
  assert (pr == np.array([[10,10],[1,8],[1,6]])).all()

  pr = np.array([[11,11],[1,9],[1,7]])
  splitS.pr_to_cell_location(pr, 0, 'Vertex', True)
  assert (pr == np.array([[10,10],[1,8],[1,6]])).all()
  splitS.pr_to_cell_location(pr, 0, 'Vertex', True, reverse=True)
  assert (pr == np.array([[11,11],[1,9],[1,7]])).all()

def test_pr_to_global_num():
  pr = np.array([[11,11],[1,9],[1,7]])
  splitS.pr_to_global_num(pr, np.array([10,1,100]), reverse=False)
  assert (pr == np.array([[11+9,11+9],[1,9],[1+99,7+99]])).all()
  splitS.pr_to_global_num(pr, np.array([10,1,100]), reverse=True)
  assert (pr == np.array([[11,11],[1,9],[1,7]])).all()

@mark_mpi_test(3)
def test_split_original_joins_S(sub_comm):
  if sub_comm.Get_rank() == 0:
    pt = """
Big.P0.N0 Zone_t:
Small.P0.N0 Zone_t:
  """
  elif sub_comm.Get_rank() == 1:
    pt = """
Small.P1.N0 Zone_t:
  ZBC ZoneBC_t:
    match2 BC_t "Big":
      GridConnectivityDonorName Descriptor_t "match1":
      Transform int[IndexDimension] [-2,-1,-3]:
      PointRange IndexRange_t [[4,1],[4,4],[5,1]]:
      distPR IndexRange_t [[7,1],[9,9],[5,1]]:
      distPRDonor IndexRange_t [[17,17],[3,9],[1,5]]:
      zone_offset DataArray_t [1,6,1]:
  """
  elif sub_comm.Get_rank() == 2:
    pt = """
Big.P2.N0 Zone_t:
  ZBC ZoneBC_t:
    match1 BC_t "Small":
      GridConnectivityDonorName Descriptor_t "match2":
      Transform int[IndexDimension] [-2,-1,-3]:
      PointRange IndexRange_t [[6,6],[3,5],[1,5]]:
      distPR IndexRange_t [[17,17],[3,9],[1,5]]:
      distPRDonor IndexRange_t [[7,1],[9,9],[5,1]]:
      zone_offset DataArray_t [12,1,1]:
Big.P2.N1 Zone_t:
  ZBC ZoneBC_t:
    match1 BC_t "Small":
      GridConnectivityDonorName Descriptor_t "match2":
      Transform int[IndexDimension] [-2,-1,-3]:
      PointRange IndexRange_t [[6,6],[1,5],[1,5]]:
      distPR IndexRange_t [[17,17],[3,9],[1,5]]:
      distPRDonor IndexRange_t [[7,1],[9,9],[5,1]]:
      zone_offset DataArray_t [12,5,1]:
Small.P2.N1 Zone_t:
  ZBC ZoneBC_t:
    match2 BC_t "Big":
      GridConnectivityDonorName Descriptor_t "match1":
      Transform int[IndexDimension] [-2,-1,-3]:
      PointRange IndexRange_t [[4,1],[4,4],[5,1]]:
      distPR IndexRange_t [[7,1],[9,9],[5,1]]:
      distPRDonor IndexRange_t [[17,17],[3,9],[1,5]]:
      zone_offset DataArray_t [4,6,1]:
  """
  part_tree  = parse_yaml_cgns.to_cgns_tree(pt)
  part_zones = I.getZones(part_tree)
  splitS.split_original_joins_S(part_zones, sub_comm)

  if sub_comm.Get_rank() == 0:
    assert PT.get_nodes_from_name(part_tree, 'match1.*') == []
    assert PT.get_nodes_from_name(part_tree, 'match2.*') == []
  elif sub_comm.Get_rank() == 1:
    assert PT.get_nodes_from_name(part_tree, 'match1.*') == []
    assert len(PT.get_nodes_from_name(part_tree, 'match2.*')) == 1
    match2 = I.getNodeFromName(part_tree, 'match2.0')
    assert I.getValue(match2) == 'Big.P2.N1'
    assert I.getType(match2) == 'GridConnectivity1to1_t'
    assert (I.getNodeFromName(match2, 'PointRange')[1] == [[4,1],[4,4],[5,1]]).all()
    assert (I.getNodeFromName(match2, 'PointRangeDonor')[1] == [[6,6],[2,5],[1,5]]).all()
  elif sub_comm.Get_rank() == 2:
    assert len(PT.get_nodes_from_name(part_tree, 'match1.*')) == 3
    assert len(PT.get_nodes_from_name(part_tree, 'match2.*')) == 2
    match1_1 = I.getNodeFromPath(part_tree, 'Base/Big.P2.N0/ZoneGridConnectivity/match1.0')
    assert I.getValue(match1_1) == 'Small.P2.N1'
    assert (I.getNodeFromName(match1_1, 'PointRange')[1] == [[6,6],[3,5],[1,5]]).all()
    assert (I.getNodeFromName(match1_1, 'PointRangeDonor')[1] == [[4,2],[4,4],[5,1]]).all()
    match1_2 = I.getNodeFromPath(part_tree, 'Base/Big.P2.N1/ZoneGridConnectivity/match1.0')
    assert I.getValue(match1_2) == 'Small.P1.N0'
    assert (I.getNodeFromName(match1_2, 'PointRange')[1] == [[6,6],[2,5],[1,5]]).all()
    assert (I.getNodeFromName(match1_2, 'PointRangeDonor')[1] == [[4,1],[4,4],[5,1]]).all()
    match1_3 = I.getNodeFromPath(part_tree, 'Base/Big.P2.N1/ZoneGridConnectivity/match1.1')
    assert I.getValue(match1_3) == 'Small.P2.N1'
    assert (I.getNodeFromName(match1_3, 'PointRange')[1] == [[6,6],[1,2],[1,5]]).all()
    assert (I.getNodeFromName(match1_3, 'PointRangeDonor')[1] == [[2,1],[4,4],[5,1]]).all()
    match2_1 = I.getNodeFromPath(part_tree, 'Base/Small.P2.N1/ZoneGridConnectivity/match2.0')
    assert I.getValue(match2_1) == 'Big.P2.N0'
    assert (I.getNodeFromName(match2_1, 'PointRange')[1] == [[4,2],[4,4],[5,1]]).all()
    assert (I.getNodeFromName(match2_1, 'PointRangeDonor')[1] == [[6,6],[3,5],[1,5]]).all()
    match2_2 = I.getNodeFromPath(part_tree, 'Base/Small.P2.N1/ZoneGridConnectivity/match2.1')
    assert I.getValue(match2_2) == 'Big.P2.N1'
    assert (I.getNodeFromName(match2_2, 'PointRange')[1] == [[2,1],[4,4],[5,1]]).all()
    assert (I.getNodeFromName(match2_2, 'PointRangeDonor')[1] == [[6,6],[1,2],[1,5]]).all()

def test_create_zone_gnums():
  dist_zone_cell = np.array([6,8,4])
  cell_window = np.array([[4,6], [6,8], [1,3]])
  vtx_gnum, face_gnum, cell_gnum = splitS.create_zone_gnums(cell_window, dist_zone_cell, dtype=np.int32)

  expected_vtx = np.array([39,40,41,46,47,48,53,54,55,102,103,104,109,110,111,116,117,118,
                           165,166,167,172,173,174,179,180,181])
  expected_face = np.array([ 39 ,40 ,41 ,46 ,47 ,48 ,95 ,96 ,97 ,102,103,104,258,259,264,265,270,271,
                             312,313,318,319,324,325,474,475,480,481,522,523,528,529,570,571,576,577])
  expected_cell = np.array([34,35,40,41,82,83,88,89])

  assert (vtx_gnum  == expected_vtx ).all()
  assert (face_gnum == expected_face).all()
  assert (cell_gnum == expected_cell).all()
  assert vtx_gnum.dtype == face_gnum.dtype == cell_gnum.dtype == np.int32
