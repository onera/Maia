import Converter.Internal as I
import numpy as np
from maia.partitioning.split_U import cgns_to_pdm_dmesh as C2P

def test_concatenate_point_list():
  pl1 = np.array([[2, 4, 6, 8]])
  pl2 = np.array([[10, 20, 30, 40, 50, 60]])
  pl3 = np.array([[100]])
  plvoid = np.empty((1,0))

  #No pl at all in the mesh
  none_idx, none = C2P.concatenate_point_list([])
  assert none_idx == [0]
  assert isinstance(none, np.ndarray)
  assert none.shape == (0,)

  #A pl, but with no data
  empty_idx, empty = C2P.concatenate_point_list([plvoid])
  assert (none_idx == [0,0]).all()
  assert isinstance(empty, np.ndarray)
  assert empty.shape == (0,)
  
  # A pl with data
  one_idx, one = C2P.concatenate_point_list([pl1])
  assert (one_idx == [0,4]).all()
  assert (one     == pl1[0]).all()

  # Several pl
  merged_idx, merged = C2P.concatenate_point_list([pl1, pl2, pl3])
  assert (merged_idx == [0, pl1.size, pl1.size+pl2.size, pl1.size+pl2.size+pl3.size]).all()
  assert (merged[0:pl1.size]                 == pl1[0]).all()
  assert (merged[pl1.size:pl1.size+pl2.size] == pl2[0]).all()
  assert (merged[pl1.size+pl2.size:]         == pl3[0]).all()
  # Several pl, some with no data
  merged_idx, merged = C2P.concatenate_point_list([pl1, plvoid, pl2])
  assert (merged_idx == [0, 4, 4, 10]).all()
  assert (merged[0:4 ] == pl1[0]).all()
  assert (merged[4:10] == pl2[0]).all()

def test_concatenate_point_list_of_types():
  zone   = I.newZone(ztype='Unstructured')
  zoneBC = I.newZoneBC(parent=zone)
  point_lists = [np.array([[2,4,6,8]]          , np.int32),
                 np.array([[10,20,30,40,50,60]], np.int32),
                 np.array([[100]]              , np.int32),
                 np.empty((1,0)                , np.int32)]
  for i, pl in enumerate(point_lists):
    I.newBC('bc'+str(i+1), pointList=pl, parent=zoneBC)

  merged_bc_idx, merged_bc = C2P.concatenate_point_list_of_types(zone, 'ZoneBC_t/BC_t')
  expected_merged_idx, expected_merged = C2P.concatenate_point_list(point_lists)
  assert (merged_bc_idx == expected_merged_idx).all()
  assert (merged_bc     == expected_merged    ).all()
