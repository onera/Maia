from pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal as I
import numpy as np

from maia import npy_pdm_gnum_dtype as pdm_dtype
from maia.utils.yaml   import parse_yaml_cgns
from maia.transfer.dist_to_part import recover_jn as JBTP

dtype = 'I4' if pdm_dtype == np.int32 else 'I8'

@mark_mpi_test(2)
def test_get_pl_donor(sub_comm):
  if sub_comm.Get_rank() == 0:
    dt = """
ZoneA Zone_t [[6,0,0]]:
  ZGC ZoneGridConnectivity_t:
    matchAB GridConnectivity_t "ZoneB":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[1, 7]]:
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridConnectivityDonorName Descriptor_t "matchBA":
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,2,6]:
ZoneB Zone_t [[12,0,0]]:
  ZGC ZoneGridConnectivity_t:
    matchBA GridConnectivity_t "ZoneA":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[8,6,14]]:
      GridConnectivityDonorName Descriptor_t "matchAB":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [0,3,6]:
""".format(dtype)
    pt = """
ZoneA.P0.N0 Zone_t [[6,0,0]]:
  ZGC ZoneGridConnectivity_t:
    matchAB GridConnectivity_t "ZoneB":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[25,21,11]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [2,3,5]:
""".format(dtype)
  elif sub_comm.Get_rank() == 1:
    dt = """
ZoneA Zone_t [[6,0,0]]:
  ZGC ZoneGridConnectivity_t:
    matchAB GridConnectivity_t "ZoneB":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridConnectivityDonorName Descriptor_t "matchBA":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[2,12,5,3]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [2,6,6]:
ZoneB Zone_t [[12,0,0]]:
  ZGC ZoneGridConnectivity_t:
    matchBA GridConnectivity_t "ZoneA":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridConnectivityDonorName Descriptor_t "matchAB":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[4,9,10]]:
      :CGNS#Distribution UserDefinedData_t:
        Index DataArray_t {0} [3,6,6]:
""".format(dtype)
    pt = """
ZoneA.P1.N0 Zone_t:
  ZGC ZoneGridConnectivity_t:
    matchAB GridConnectivity_t "ZoneB":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[8,13,2]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [1,6,4]:
ZoneB.P1.N0 Zone_t:
  ZGC ZoneGridConnectivity_t:
    matchBA GridConnectivity_t "ZoneA":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[8,12,9,20,1]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [5,3,1,2,6]:
ZoneB.P1.N1 Zone_t:
  ZGC ZoneGridConnectivity_t:
    matchBA GridConnectivity_t "ZoneA":
      GridConnectivityType GridConnectivityType_t "Abutting1to1":
      GridLocation GridLocation_t "FaceCenter":
      PointList IndexArray_t [[5]]:
      :CGNS#GlobalNumbering UserDefinedData_t:
        Index DataArray_t {0} [4]:
""".format(dtype)

  dist_tree = parse_yaml_cgns.to_cgns_tree(dt)
  part_tree = parse_yaml_cgns.to_cgns_tree(pt)

  JBTP.get_pl_donor(dist_tree, part_tree, sub_comm)
  
  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneA.P0.N0/ZGC/matchAB/PointListDonor')[1] == [20,12,8]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneA.P0.N0/ZGC/matchAB/Donor')[1][:,0] == [1,1,1]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneA.P0.N0/ZGC/matchAB/Donor')[1][:,1] == [0,0,0]).all()
    assert I.getValue(I.getNodeFromPath(part_tree, 'Base/ZoneA.P0.N0/ZGC/matchAB')) == 'ZoneB'
  if sub_comm.Get_rank() == 1:
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneA.P1.N0/ZGC/matchAB/PointListDonor')[1] == [9,1,5]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneA.P1.N0/ZGC/matchAB/Donor')[1][:,0] == [1,1,1]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneA.P1.N0/ZGC/matchAB/Donor')[1][:,1] == [0,0,1]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneB.P1.N0/ZGC/matchBA/PointListDonor')[1] == [11,21,8,25,13]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneB.P1.N0/ZGC/matchBA/Donor')[1][:,0] == [0,0,1,0,1]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneB.P1.N0/ZGC/matchBA/Donor')[1][:,1] == [0,0,0,0,0]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneB.P1.N1/ZGC/matchBA/PointListDonor')[1] == [2]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneB.P1.N1/ZGC/matchBA/Donor')[1][:,0] == [1]).all()
    assert (I.getNodeFromPath(part_tree, 'Base/ZoneB.P1.N1/ZGC/matchBA/Donor')[1][:,1] == [0]).all()
    assert I.getValue(I.getNodeFromPath(part_tree, 'Base/ZoneA.P1.N0/ZGC/matchAB')) == 'ZoneB'
    assert I.getValue(I.getNodeFromPath(part_tree, 'Base/ZoneB.P1.N0/ZGC/matchBA')) == 'ZoneA'
