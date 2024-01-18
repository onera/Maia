import pytest_parallel

import numpy as np
from maia.factory.partitioning.load_balancing import multi_zone_balancing
from maia.factory.partitioning.load_balancing import single_zone_balancing
from maia.factory.partitioning.load_balancing import balancing_quality

class Test_balancing_quality:
  def test_single_zone(self):
    repart = np.array([[20,20,20,25,20]], dtype=np.int32) 
    out = np.asarray(balancing_quality.compute_balance_and_splits_seq(repart))
    assert np.all(out[:3] == [21, 20, 25])
    assert np.all(abs(out[3:5] - np.array([0.8944, 0.04259])) < 1E-4)
    assert np.all(out[5:] == [5,20,25])

  def test_single_proc(self):
    repart = np.array([[100], [20], [300]], dtype=np.int32) 
    out = np.asarray(balancing_quality.compute_balance_and_splits_seq(repart))
    assert np.all(out == [420, 420, 420, 0, 0, 3, 20, 300])

  def test_multiple_A(self):
    repart = np.array([[0, 75,  0], 
                       [75, 0,75]], dtype=np.int32)
    out = np.asarray(balancing_quality.compute_balance_and_splits_seq(repart))
    assert np.all(out == [75,75,75,0,0,3,75,75])

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
  if comm.Get_rank() == 0:
    repart = np.array([50, 15, 45, 0]) 
  elif comm.Get_rank() == 1:
    repart = np.array([50, 15, 0, 10]) 
  out = np.asarray(balancing_quality.compute_balance_and_splits(repart, comm))
  assert np.all(out[:3] == [93, 75, 110])
  assert np.all(abs(out[3:5] - np.array([12.3794, 0.1331])) < 1E-4)
  assert np.all(out[5:] == [6,10,50])

def test_single_zone_balancing():
  out = single_zone_balancing.homogeneous_repart(30,3)
  assert np.all(out == [10,10,10])
  out = single_zone_balancing.homogeneous_repart(31,3)
  assert np.all(out == [11,10,10])

def test_multi_zone_balancing():
  diczone = {'zoneA' : 100, 'zoneB': 200, 'zoneC':100}
  repart_zones = multi_zone_balancing.balance_with_uniform_weights(diczone, 3)
  assert repart_zones['zoneA'] == [0, 100, 0]
  assert repart_zones['zoneB'] == [200, 0, 0]
  assert repart_zones['zoneC'] == [0, 0, 100]

  repart_zones = multi_zone_balancing.balance_with_non_uniform_weights(diczone, 3)
  assert repart_zones['zoneA'] == [100, 0, 0]
  assert repart_zones['zoneB'] == [34, 34, 132]
  assert repart_zones['zoneC'] == [0, 100, 0]

