import Converter.Internal as I
import numpy as np

import maia.utils.py_utils as py_utils
from maia.tree_exchange.dist_to_part import index_exchange as IBTP


def test_collect_distributed_pl():
  zone   = I.newZone(ztype='Unstructured')
  zoneBC = I.newZoneBC(parent=zone)
  point_lists = [np.array([[2,4,6,8]]          , np.int32),
                 np.array([[10,20,30,40,50,60]], np.int32),
                 np.array([[100]]              , np.int32),
                 np.empty((1,0)                , np.int32)]
  point_ranges = [np.array([[3,3],[1,3],[1,3]] , np.int32), #This one should be ignored
                  np.array([[35,55]]           , np.int32)]
  for i, pl in enumerate(point_lists):
    I.newBC('bc'+str(i+1), pointList=pl, parent=zoneBC)
  for i, pr in enumerate(point_ranges):
    bc = I.newBC('bc'+str(i+1+len(point_lists)), pointRange=pr, parent=zoneBC)
    distri = I.createUniqueChild(bc, ":CGNS#Distribution", "UserDefinedData_t")
    I.newDataArray('Index', [10,15,20], parent=distri)

  collected = IBTP.collect_distributed_pl(zone, ['ZoneBC_t/BC_t'])
  assert len(collected) == len(point_lists) + 1
  for i in range(len(point_lists)):
    assert (collected[i] == point_lists[i]).all()
  assert (collected[-1] == np.arange(35+10, 35+15)).all()
