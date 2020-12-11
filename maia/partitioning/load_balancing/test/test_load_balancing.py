import numpy as np
from maia.partitioning.load_balancing import balancing_quality

def test_balancing_quality():
  #Single zone
  repart = np.array([[20,20,20,25,20]], dtype=np.int32) 
  out = np.asarray(balancing_quality.compute_balance_and_splits(repart))
  assert np.all(out[:3] == [21, 20, 25])
  assert np.all(abs(out[3:5] - np.array([0.8944, 0.04259])) < 1E-4)
  assert np.all(out[5:] == [5,20,25])

  #Single proc
  repart = np.array([[100], [20], [300]], dtype=np.int32) 
  out = np.asarray(balancing_quality.compute_balance_and_splits(repart))
  assert np.all(out == [420, 420, 420, 0, 0, 3, 20, 300])

  #Multiple
  repart = np.array([[0, 75,  0], 
                     [75, 0,75]], dtype=np.int32)
  out = np.asarray(balancing_quality.compute_balance_and_splits(repart))
  assert np.all(out == [75,75,75,0,0,3,75,75])

  repart = np.array([[50, 50], 
                     [15, 15],
                     [45,  0],
                     [ 0, 10]], dtype=np.int32)
  out = np.asarray(balancing_quality.compute_balance_and_splits(repart))
  assert np.all(out[:3] == [93, 75, 110])
  assert np.all(abs(out[3:5] - np.array([12.3794, 0.1331])) < 1E-4)
  assert np.all(out[5:] == [6,10,50])

