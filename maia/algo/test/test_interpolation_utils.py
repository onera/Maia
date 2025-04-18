import numpy as np
from maia.utils import vstride as vs

from maia.algo import interpolation_utils as ITPU

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

  vtx_to_tgt, vtx_to_tgt_wgt = ITPU._cell_tgt_to_vtx_tgt(cell_vtx, cell_tgt, cell_vtx_wgt, n_vtx)

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
  out = ITPU.Interpolator._reduce_single_val(fake_interpolator, 0, data)
  assert np.array_equal(out, data)

  fake_interpolator.sending_gnums = [{'come_from_idx' : np.array([0,2,4,6])}]
  fake_interpolator.tgt_weight = [1./np.array([1,1,1E-20,1,3,1])]
  data = np.array([1,2, 10,11, 20,30], np.float64)
  out = ITPU.Interpolator._reduce_weighted_mean(fake_interpolator, 0, data)
  assert (out == np.array([1.5, 10., 27.5])).all()

  fake_interpolator.sending_gnums = [{'come_from_idx' : np.array([0,2,5,6])}]
  fake_interpolator.tgt_weight = [1./np.array([1,1, 1E-20,1,3, 1])]
  data = np.array([1,2, 10,11,20, 30], np.float64)
  out = ITPU.Interpolator._reduce_weighted_mean(fake_interpolator, 0, data)
  assert (out == np.array([1.5, 10., 30.0])).all()