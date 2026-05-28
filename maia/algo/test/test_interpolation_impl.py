import numpy as np
from maia.utils import vstride as vs

import maia.pytree as PT
import maia

from maia.algo import interpolation_impl as ITP

# Sample tree + functions used by test_{d|p}interpolation_cons
minimal_tri = """
  zone Zone_t [[6, 5, 0]]:
    ZoneType ZoneType_t "Unstructured":
    GridCoordinates GridCoordinates_t:
      CoordinateX DataArray_t R8 [0, 0, 0, 1, 1, 0.5]:
      CoordinateY DataArray_t R8 [1, 0.5, 0, 0, 1, 0.5]:
      CoordinateZ DataArray_t R8 [0, 0, 0, 0, 0, 0]:
    TRI Elements_t [5, 0]:
      ElementRange IndexRange_t [1, 5]:
      ElementConnectivity DataArray_t [1,2,6, 2,3,6, 3,4,6, 4,5,6, 5,1,6]:
    Geometry_0d DiscreteData_t:
      DualVol24 DataArray_t R8 [3, 2, 3, 4, 4, 8]:
    Geometry_2d DiscreteData_t:
      GridLocation GridLocation_t "CellCenter":
      Measure DataArray_t R8 [0.125, 0.125, 0.25, 0.25, 0.25]:
"""

def union(*trees):
  for i,tree in enumerate(trees):
    PT.set_name(PT.get_node_from_label(tree, 'Zone_t'), f'Zone_{i}')
  return PT.union(*trees)

def integrated_val(tree, field, comm):
  maia.algo.compute_elements_measure(tree, 'CellCenter', comm)
  tot = 0
  for zone in PT.get_all_Zone_t(tree):
    f   = PT.get_np_value(PT.find_node_from_name(zone, field))
    vol = PT.get_np_value(PT.find_node_from_name(zone, 'Measure'))
    tot += (f*vol).sum()
  return comm.allreduce(tot)

def test_cell_tgt_to_vtx_tgt():
  n_vtx = 12
  cell_vtx = vs.array([[5,7,1,6], [3,9,5], [3,5,2,1], [10,4,8], [10,2,3]], dtype=np.int32)
  cell_tgt = vs.array([[],        [11,9],  [101],     [],       [6,2,1]])
  cell_vtx_wgt = np.array([3,9,5, 3,9,5, 3,5,2,1, 10,2,3, 10,2,3, 10,2,3], dtype=np.float64)*0.1

  expctd_vtx_to_tgt = vs.array([[101], [1,2,6,101], [1,2,6,9,11,101], [], [9,11,101], [], 
                                [], [], [9,11], [1,2,6], [],[]])
  expctd_vtx_to_tgt_wgt = vs.from_displs(expctd_vtx_to_tgt.displs,
    np.array([1, 2,2,2,2, 3,3,3,3,3,3, 5,5,5, 9,9, 10,10,10], dtype=np.float64)*0.1)
  # Vtx 2 appears in cells 3 & 5. Thoses cells have localized tgt ids 101,6,2,1 in them, so we
  # expect to get tgt 101,6,2 and 1 for vtx2

  vtx_to_tgt, vtx_to_tgt_wgt = ITP._cell_tgt_to_vtx_tgt(cell_vtx, cell_tgt, cell_vtx_wgt, n_vtx)

  # The order of `vtx_to_tgt` does not matter and is not specified by the algorithm,
  # so whatever we get, we can order it before checking it
  vtx_to_tgt = vs.sort(vtx_to_tgt, vs.INNER_AXIS)

  assert vs.array_equal(vtx_to_tgt, expctd_vtx_to_tgt)
  assert vs.array_equal(vtx_to_tgt_wgt, expctd_vtx_to_tgt_wgt)
  # Logs detail
  # active_cell = [1,1, 2, 4,4,4]
  # active_cell_vtx = [3,9,5, 3,9,5,   3,5,2,1,   10,2,3, 10,2,3, 10,2,3] (counts = [3,3,4,3,3,3])
  # vtx_to_tgt_n = [1,4,6,0,3,0,0,0,2,3,0,0] (size = n_vtx = 12) <-- nb of apparition for each vertex
  # cell_tgt_extended = [11,11,11, 9,9,9, 101,101,101,101, 6,6,6, 2,2,2, 1,1,1] <-- each tgt point is repeted times
  #                                                                              the number of vertex in the src cell
  # sort_idx = [9,14,8,11,17,0,6,3,12,15,18,2,5,7,4,1,13,10,16] <-- Selection order to sort following vertices order

def test_interpolator_reductions():
  class Empty: #Used to create a interpolator like object
    pass
  fake_interpolator = Empty()

  fake_interpolator.sending_gnums = [{'come_from_idx' : np.array([0,1,2,3])}]
  data = np.array([1,2,3], np.int32)
  out = ITP.Interpolator._reduce_single_val(fake_interpolator, 0, data)
  assert np.array_equal(out, data)

  fake_interpolator.sending_gnums = [{'come_from_idx' : np.array([0,2,4,6])}]
  fake_interpolator.tgt_weight = [1./np.array([1,1,1E-20,1,3,1])]
  data = np.array([1,2, 10,11, 20,30], np.float64)
  out = ITP.Interpolator._reduce_weighted_mean(fake_interpolator, 0, data)
  assert (out == np.array([1.5, 10., 27.5])).all()

  fake_interpolator.sending_gnums = [{'come_from_idx' : np.array([0,2,5,6])}]
  fake_interpolator.tgt_weight = [1./np.array([1,1, 1E-20,1,3, 1])]
  data = np.array([1,2, 10,11,20, 30], np.float64)
  out = ITP.Interpolator._reduce_weighted_mean(fake_interpolator, 0, data)
  assert (out == np.array([1.5, 10., 30.0])).all()