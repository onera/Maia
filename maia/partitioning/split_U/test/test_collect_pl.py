import Converter.Internal as I
import numpy as np

import maia.utils.py_utils as py_utils
from maia.partitioning.split_U import collect_pl


def test_concatenate_point_list_of_types():
  zone   = I.newZone(ztype='Unstructured')
  zoneBC = I.newZoneBC(parent=zone)
  point_lists = [np.array([[2,4,6,8]]          , np.int32),
                 np.array([[10,20,30,40,50,60]], np.int32),
                 np.array([[100]]              , np.int32),
                 np.empty((1,0)                , np.int32)]
  for i, pl in enumerate(point_lists):
    I.newBC('bc'+str(i+1), pointList=pl, parent=zoneBC)

  collected = collect_pl.collect_distributed_pl(zone, ['ZoneBC_t/BC_t'])
  assert len(collected) == len(point_lists)
  for i in range(len(collected)):
    assert (collected[i] == point_lists[i]).all()
