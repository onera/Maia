from pytest_mpi_check._decorator import mark_mpi_test

import mpi4py.MPI as MPI
import numpy as np
import Converter.Internal as I

from maia.utils import parse_yaml_cgns
from maia.transform.dist_tree import add_joins_ordinal

class Test_compare_pointrange():
  def test_ok(self):
    jn1 = I.newGridConnectivity1to1(pointRange     =[[17,17],[3,9],[1,5]], pointRangeDonor=[[7,1],[9,9],[5,1]])
    jn2 = I.newGridConnectivity1to1(pointRangeDonor=[[17,17],[3,9],[1,5]], pointRange     =[[7,1],[9,9],[5,1]])
    assert(add_joins_ordinal._compare_pointrange(jn1, jn2) == True)
  def test_ko(self):
    jn1 = I.newGridConnectivity1to1(pointRange     =[[17,17],[3,9],[1,5]], pointRangeDonor=[[7,1],[9,9],[5,1]])
    jn2 = I.newGridConnectivity1to1(pointRangeDonor=[[17,17],[3,9],[1,5]], pointRange     =[[1,7],[9,9],[1,5]])
    assert(add_joins_ordinal._compare_pointrange(jn1, jn2) == False)
  def test_empty(self):
    jn1 = I.newGridConnectivity1to1(pointRange     =np.empty((3,2), np.int32), pointRangeDonor=np.empty((3,2), np.int32))
    jn2 = I.newGridConnectivity1to1(pointRangeDonor=np.empty((3,2), np.int32), pointRange     =np.empty((3,2), np.int32))
    assert(add_joins_ordinal._compare_pointrange(jn1, jn2) == True)

class Test_compare_pointlist():
  def test_ok(self):
    jn1 = I.newGridConnectivity1to1(pointList     =[[12,14,16,18]], pointListDonor=[[9,7,5,3]])
    jn2 = I.newGridConnectivity1to1(pointListDonor=[[12,14,16,18]], pointList     =[[9,7,5,3]])
    assert(add_joins_ordinal._compare_pointlist(jn1, jn2) == True)
  def test_ko(self):
    jn1 = I.newGridConnectivity1to1(pointList     =[[12,14,16,18]], pointListDonor=[[9,7,5,3]])
    jn2 = I.newGridConnectivity1to1(pointListDonor=[[12,14,16,18]], pointList     =[[3,9,5,7]])
    assert(add_joins_ordinal._compare_pointlist(jn1, jn2) == False)
  def test_empty(self):
    jn1 = I.newGridConnectivity1to1(pointList     =np.empty((1,0), np.int32), pointListDonor=np.empty((1,0), np.int32))
    jn2 = I.newGridConnectivity1to1(pointListDonor=np.empty((1,0), np.int32), pointList     =np.empty((1,0), np.int32))
    assert(add_joins_ordinal._compare_pointlist(jn1, jn2) == True)

@mark_mpi_test(1)
def test_add_joins_ordinal(sub_comm):
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t [[27],[8],[0]]:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [1,4,7,10]:    # HERE
        PointListDonor IndexArray_t [13,16,7,10]:  # HERE
  ZoneB Zone_t [[27],[8],[0]]:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity_t "ZoneA":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [13,16,7,10]:  # HERE
        PointListDonor IndexArray_t [1,4,7,10]:    # HERE
      matchBC1 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [32,34]:
        PointListDonor IndexArray_t [1,3]:
      matchBC2 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [33,35]:
        PointListDonor IndexArray_t [2,4]:
Base1 CGNSBase_t [3,3]:
  ZoneC Zone_t [[18],[4],[0]]:
    ZGC ZoneGridConnectivity_t:
      matchCB2 GridConnectivity_t "Base0/ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList      IndexArray_t [2,4]:         # HERE
        PointListDonor IndexArray_t [33,35]:       # HERE
      matchCB1 GridConnectivity_t "Base0/ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [1,3]:
        PointListDonor IndexArray_t [32,34]:
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)

  add_joins_ordinal.add_joins_ordinal(dist_tree, sub_comm)

  assert I.getNodeFromPath(dist_tree, 'Base0/ZoneA/ZGC/matchAB/Ordinal'    )[1][0] == 1
  assert I.getNodeFromPath(dist_tree, 'Base0/ZoneA/ZGC/matchAB/OrdinalOpp' )[1][0] == 2
  assert I.getNodeFromPath(dist_tree, 'Base0/ZoneB/ZGC/matchBA/Ordinal'    )[1][0] == 2
  assert I.getNodeFromPath(dist_tree, 'Base0/ZoneB/ZGC/matchBA/OrdinalOpp' )[1][0] == 1
  assert I.getNodeFromPath(dist_tree, 'Base0/ZoneB/ZGC/matchBC1/Ordinal'   )[1][0] == 3
  assert I.getNodeFromPath(dist_tree, 'Base0/ZoneB/ZGC/matchBC1/OrdinalOpp')[1][0] == 6
  assert I.getNodeFromPath(dist_tree, 'Base0/ZoneB/ZGC/matchBC2/Ordinal'   )[1][0] == 4
  assert I.getNodeFromPath(dist_tree, 'Base0/ZoneB/ZGC/matchBC2/OrdinalOpp')[1][0] == 5
  assert I.getNodeFromPath(dist_tree, 'Base1/ZoneC/ZGC/matchCB2/Ordinal'   )[1][0] == 5
  assert I.getNodeFromPath(dist_tree, 'Base1/ZoneC/ZGC/matchCB2/OrdinalOpp')[1][0] == 4
  assert I.getNodeFromPath(dist_tree, 'Base1/ZoneC/ZGC/matchCB1/Ordinal'   )[1][0] == 6
  assert I.getNodeFromPath(dist_tree, 'Base1/ZoneC/ZGC/matchCB1/OrdinalOpp')[1][0] == 3


@mark_mpi_test(3)
def test_add_joins_ordinal_3p(sub_comm):
  yt = """
Base0 CGNSBase_t [3,3]:
  ZoneA Zone_t [[27],[8],[0]]:
    ZGC ZoneGridConnectivity_t:
      matchAB GridConnectivity_t "ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [1,4,7,10]:
        PointListDonor IndexArray_t [13,16,7,10]:
  ZoneB Zone_t [[27],[8],[0]]:
    ZGC ZoneGridConnectivity_t:
      matchBA GridConnectivity_t "ZoneA":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [13,16,7,10]:
        PointListDonor IndexArray_t [1,4,7,10]:
      matchBC1 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [32,34]:
        PointListDonor IndexArray_t [1,3]:
      matchBC2 GridConnectivity_t "Base1/ZoneC":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [33,35]:
        PointListDonor IndexArray_t [2,4]:
Base1 CGNSBase_t [3,3]:
  ZoneC Zone_t [[18],[4],[0]]:
    ZGC ZoneGridConnectivity_t:
      matchCB2 GridConnectivity_t "Base0/ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [2,4]:
        PointListDonor IndexArray_t [33,35]:
      matchCB1 GridConnectivity_t "Base0/ZoneB":
        GridLocation GridLocation_t "FaceCenter":
        PointList IndexArray_t [1,3]:
        PointListDonor IndexArray_t [32,34]:
"""
  dist_tree = parse_yaml_cgns.to_cgns_tree(yt)

  #Correct tree to simulate join distribution. One proc have do dist data!
  match_names = ['matchAB', 'matchBA', 'matchBC1', 'matchBC2', 'matchCB1', 'matchCB2']
  for match_name in match_names:
    match_node = I.getNodeFromName(dist_tree, match_name)
    pl_n       = I.getNodeFromName(match_node, 'PointList')
    pld_n      = I.getNodeFromName(match_node, 'PointListDonor')
    if sub_comm.Get_rank() == 0:
      pl_n[1]  = pl_n[1][0:1]
      pld_n[1] = pld_n[1][0:1]
    if sub_comm.Get_rank() == 1:
      pl_n[1]  = pl_n[1][1:3]
      pld_n[1] = pld_n[1][1:3]
    if sub_comm.Get_rank() == 2:
      pl_n[1]  = pl_n[1][3:4]
      pld_n[1] = pld_n[1][3:4]

  add_joins_ordinal.add_joins_ordinal(dist_tree, sub_comm)

  expected_ordinal     = [1,2,3,4,5,6]
  expected_ordinal_opp = [2,1,6,5,4,3]
  for base in I.getBases(dist_tree):
    for zone in I.getZones(base):
      for zgc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
        for i, gc in enumerate(I.getNodesFromType(dist_tree, 'GridConnectivity_t')):
          assert I.getNodeFromName1(gc, 'Ordinal')[1]    == expected_ordinal[i]
          assert I.getNodeFromName1(gc, 'OrdinalOpp')[1] == expected_ordinal_opp[i]

