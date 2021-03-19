import pytest
from pytest_mpi_check._decorator import mark_mpi_test

import numpy      as np
import Converter.Internal as I

from maia import npy_pdm_gnum_dtype as pdm_gnum_dtype
import maia.tree_exchange.part_to_dist.index_exchange as IPTB

@mark_mpi_test(4)
def test_create_part_pl_gnum_unique(sub_comm):
  part_zones = [I.newZone('Zone.P{0}.N0'.format(sub_comm.Get_rank()))]
  if sub_comm.Get_rank() == 0:
    I.newZoneSubRegion("ZSR", pointList=[[2,4,6,8]], gridLocation='Vertex', parent=part_zones[0])
  if sub_comm.Get_rank() == 2:
    part_zones.append(I.newZone('Zone.P2.N1'))
    I.newZoneSubRegion("ZSR", pointList=[[1,3]], gridLocation='Vertex', parent=part_zones[0])
    I.newZoneSubRegion("ZSR", pointList=[[2,4,6]], gridLocation='Vertex', parent=part_zones[1])
  if sub_comm.Get_rank() == 3:
    part_zones = []
  IPTB.create_part_pl_gnum_unique(part_zones, "ZSR", sub_comm)

  for p_zone in part_zones:
    if I.getNodeFromName1(p_zone, "ZSR") is not None:
      assert I.getNodeFromPath(p_zone, "ZSR/:CGNS#GlobalNumbering/Index") is not None 
  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromName(part_zones[0], 'Index')[1] == [1,2,3,4]).all()
  if sub_comm.Get_rank() == 2:
    assert (I.getNodeFromName(part_zones[1], 'Index')[1] == [7,8,9]).all()
    assert I.getNodeFromName(part_zones[0], 'Index')[1].dtype == pdm_gnum_dtype

@mark_mpi_test(4)
def test_create_part_pl_gnum(sub_comm):
  dist_zone = I.newZone('Zone')
  part_zones = [I.newZone('Zone.P{0}.N0'.format(sub_comm.Get_rank()))]
  distri_ud0 = I.createUniqueChild(part_zones[0], ':CGNS#GlobalNumbering', 'UserDefinedData_t')
  if sub_comm.Get_rank() == 0:
    I.newZoneSubRegion("ZSR", pointList=[[1,8,5,2]], gridLocation='Vertex', parent=part_zones[0])
    I.newDataArray('Vertex', [22,18,5,13,9,11,6,4], parent=distri_ud0)
  elif sub_comm.Get_rank() == 1:
    I.newDataArray('Vertex', [5,16,9,17,22], parent=distri_ud0)
  elif sub_comm.Get_rank() == 2:
    I.newDataArray('Vertex', [13,8,9,6,2], parent=distri_ud0)
    part_zones.append(I.newZone('Zone.P2.N1'))
    distri_ud1 = I.createUniqueChild(part_zones[1], ':CGNS#GlobalNumbering', 'UserDefinedData_t')
    I.newDataArray('Vertex', [4,9,13,1,7,6], parent=distri_ud1)
    I.newZoneSubRegion("ZSR", pointList=[[1,3]], gridLocation='Vertex', parent=part_zones[0])
    I.newZoneSubRegion("ZSR", pointList=[[2,4,6]], gridLocation='Vertex', parent=part_zones[1])
  elif sub_comm.Get_rank() == 3:
    part_zones = []

  IPTB.create_part_pl_gnum(dist_zone, part_zones, "ZSR", sub_comm)

  for p_zone in part_zones:
    if I.getNodeFromName1(p_zone, "ZSR") is not None:
      assert I.getNodeFromPath(p_zone, "ZSR/:CGNS#GlobalNumbering/Index") is not None 
  if sub_comm.Get_rank() == 0:
    assert (I.getNodeFromName(part_zones[0], 'Index')[1] == [7,2,4,6]).all()
    assert I.getNodeFromName(part_zones[0], 'Index')[1].dtype == pdm_gnum_dtype
  if sub_comm.Get_rank() == 2:
    assert (I.getNodeFromName(part_zones[0], 'Index')[1] == [5,4]).all()
    assert (I.getNodeFromName(part_zones[1], 'Index')[1] == [4,1,3]).all()

@mark_mpi_test(4)
def test_part_pl_to_dist_pl(sub_comm):
  dist_zone = I.newZone('Zone')
  dist_zsr = I.newZoneSubRegion("ZSR", gridLocation='Vertex', parent=dist_zone)
  part_zones = [I.newZone('Zone.P{0}.N0'.format(sub_comm.Get_rank()))]
  distri_ud0 = I.createUniqueChild(part_zones[0], ':CGNS#GlobalNumbering', 'UserDefinedData_t')
  if sub_comm.Get_rank() == 0:
    I.newDataArray('Vertex', [22,18,5,13,9,11,6,4], parent=distri_ud0)
    zsr = I.newZoneSubRegion("ZSR", pointList=[[1,8,5,2]], gridLocation='Vertex', parent=part_zones[0])
    distri_ud_zsr = I.createUniqueChild(zsr, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
    I.newDataArray('Index', [7,2,4,6], parent=distri_ud_zsr)
  elif sub_comm.Get_rank() == 1:
    I.newDataArray('Vertex', [5,16,9,17,22], parent=distri_ud0)
  elif sub_comm.Get_rank() == 2:
    I.newDataArray('Vertex', [13,8,9,6,2], parent=distri_ud0)
    part_zones.append(I.newZone('Zone.P2.N1'))
    distri_ud1 = I.createUniqueChild(part_zones[1], ':CGNS#GlobalNumbering', 'UserDefinedData_t')
    I.newDataArray('Vertex', [4,9,13,1,7,6], parent=distri_ud1)
    zsr = I.newZoneSubRegion("ZSR", pointList=[[1,3]], gridLocation='Vertex', parent=part_zones[0])
    distri_ud_zsr = I.createUniqueChild(zsr, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
    I.newDataArray('Index', [5,4], parent=distri_ud_zsr)
    zsr = I.newZoneSubRegion("ZSR", pointList=[[2,4,6]], gridLocation='Vertex', parent=part_zones[1])
    distri_ud_zsr = I.createUniqueChild(zsr, ':CGNS#GlobalNumbering', 'UserDefinedData_t')
    I.newDataArray('Index', [4,1,3], parent=distri_ud_zsr)
  elif sub_comm.Get_rank() == 3:
    part_zones = []

  IPTB.part_pl_to_dist_pl(dist_zone, part_zones, "ZSR", sub_comm)

  dist_pl     = I.getNodeFromPath(dist_zsr, 'PointList')[1]
  dist_distri = I.getNodeFromPath(dist_zsr, ':CGNS#Distribution/Index')[1]
  assert dist_distri.dtype == pdm_gnum_dtype

  if sub_comm.Get_rank() == 0:
    assert (dist_distri == [0,2,7]).all()
    assert (dist_pl     == [1,4]  ).all()
  elif sub_comm.Get_rank() == 1:
    assert (dist_distri == [2,4,7]).all()
    assert (dist_pl     == [6,9]  ).all()
  elif sub_comm.Get_rank() == 2:
    assert (dist_distri == [4,6,7]).all()
    assert (dist_pl     == [13,18]).all()
  elif sub_comm.Get_rank() == 3:
    assert (dist_distri == [6,7,7]).all()
    assert (dist_pl     == [22]   ).all()
