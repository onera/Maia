import pytest_parallel

import numpy as np
from maia.factory.partitioning.load_balancing import balancing_quality

from maia.utils import logging as mlog

class log_capture:
  def __init__(self):
    self.logs = ''
  def log(self, msg):
    self.logs += msg

class Test_balancing_quality:
  def test_single_zone(self):
    repart = np.array([[20,20,20,25,20]], dtype=np.int32) 
    out = np.asarray(balancing_quality.compute_balance_and_splits_seq(repart)) 
    assert np.all(out[:3] == [21, 20, 25])
    assert np.all(abs(out[3:5] - np.array([0.8944, 0.04259])) < 1E-4)
    assert np.all(out[5:] == [5,20,25])

  def test_empty_zones(self):
    # Case where the sum of elts is 0, ie ideal_load = 0
    repart = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
    out = np.asarray(balancing_quality.compute_balance_and_splits_seq(repart))
    assert (out == [0, 0, 0 ,0 ,0 ,0 ,0 ,0]).all()

  def test_single_proc(self):
    repart = np.array([[100], [20], [300]], dtype=np.int32) 
    out = np.asarray(balancing_quality.compute_balance_and_splits_seq(repart))
    assert np.all(out == [420, 420, 420, 0, 0, 3, 20, 300])

  def test_multiple_A(self):
    # Also test display in this test
    mlog.add_printer_to_logger('maia-stats', log_collector := log_capture())

    repart = np.array([[0, 75,  0], 
                       [75, 0,75]], dtype=np.int32)
    out = np.asarray(balancing_quality.compute_balance_and_splits_seq(repart, display=True))
    assert np.all(out == [75,75,75,0,0,3,75,75])

    excepted_log = "  ---> Mean   size : 75\n  ---> rMini  size : 75\n  ---> rMaxi  size : 75\n" \
                   "  ---> rms         : 0.0\n  ---> rmscp       : 0.0\n  ---> worse delta : 0 (0.00%)\n" \
                   "  ---> n_cuts      : 3\n"
    assert log_collector.logs == excepted_log

  def test_multiple_B(self):
    repart = np.array([[50, 50], 
                       [15, 15],
                       [45,  0],
                       [ 0, 10]], dtype=np.int32)
    out = np.asarray(balancing_quality.compute_balance_and_splits_seq(repart))
    assert np.all(out[:3] == [93, 75, 110])
    assert np.all(abs(out[3:5] - np.array([12.3794, 0.1331])) < 1E-4)
    assert np.all(out[5:] == [6,10,50])

@pytest_parallel.mark.parallel(2)
def test_balancing_quality_par(comm):
  mlog.add_printer_to_logger('maia-stats', log_collector := log_capture())
  if comm.Get_rank() == 0:
    repart = np.array([50, 15, 45, 0]) 
  elif comm.Get_rank() == 1:
    repart = np.array([50, 15, 0, 10]) 
  out = np.asarray(balancing_quality.compute_balance_and_splits(repart, comm, display=True))
  assert np.all(out[:3] == [93, 75, 110])
  assert np.all(abs(out[3:5] - np.array([12.3794, 0.1331])) < 1E-4)
  assert np.all(out[5:] == [6,10,50])

  log_lines = log_collector.logs.split('\n')
  assert log_lines[0] == f"  ---> Mean   size : {int(out[0])}"
  assert log_lines[1] == f"  ---> rMini  size : {int(out[1])}"
  assert log_lines[2] == f"  ---> rMaxi  size : {int(out[2])}"
  assert log_lines[5] == "  ---> worse delta : 17 (18.28%)"
  assert log_lines[6] == "  ---> n_cuts      : 6"
  
  # Case where the sum of elts is 0, ie ideal_load = 0
  repart = np.array([[0, 0, 0, 0]])
  out = np.asarray(balancing_quality.compute_balance_and_splits(repart, comm))
  assert (out == [0, 0, 0 ,0 ,0 ,0 ,0 ,0]).all()
