import pytest
import numpy as np

from maia.factory.partitioning.load_balancing import multi_zone_balancing  as MZB
from maia.factory.partitioning.load_balancing import single_zone_balancing as SZB

def test_karmarkar_karp():
    assert MZB.karmarkar_karp([5,9,2,4], 1) == [[1,0,3,2]]
    assert MZB.karmarkar_karp([5,9,2,4], 4) == [[2],[3],[0],[1]]
    assert MZB.karmarkar_karp([5,9,2,4], 2) == [[3,0], [2,1]]
    assert MZB.karmarkar_karp([5,9,1,3], 2) == [[1], [2,0,3]]
    assert MZB.karmarkar_karp([9,4,7,6,8,5], 3) == [[1,0], [5,4], [3,2]]

def test_single_zone_balancing():
  out = SZB.homogeneous_repart(30,3)
  assert np.all(out == [10,10,10])
  out = SZB.homogeneous_repart(31,3)
  assert np.all(out == [11,10,10])

def test_multi_zone_balancing():
  diczone = {'zoneA' : 100, 'zoneB': 200, 'zoneC':100}
  repart_zones = MZB.balance_with_uniform_weights(diczone, 3)
  assert repart_zones['zoneA'] == [0, 100, 0]
  assert repart_zones['zoneB'] == [200, 0, 0]
  assert repart_zones['zoneC'] == [0, 0, 100]

  diczone = {'zoneA' : 100, 'zoneB': 200, 'zoneC':100}
  repart_zones = MZB.balance_with_non_uniform_weights(diczone, 3)
  assert repart_zones['zoneA'] == [100, 0, 0]
  assert repart_zones['zoneB'] == [34, 34, 132]
  assert repart_zones['zoneC'] == [0, 100, 0]
  

  # Test case "some zone not affected and ranks remain"
  diczone = {'zoneA': 50, 'zoneB': 30, 'zoneC': 20, 'zoneD': 10}
  repart_zones =  MZB.balance_with_uniform_weights(diczone, 2)
  assert repart_zones['zoneA'] == [50, 0]  
  assert repart_zones['zoneB'] == [0, 30]  
  assert repart_zones['zoneC'] == [0, 20]  
  assert repart_zones['zoneD'] == [10, 0]  

  # Test case "all zone affected but ranks remains"
  diczone = {'zoneA': 100, 'zoneB': 50, 'zoneC': 50}
  repart_zones = MZB.balance_with_uniform_weights(diczone, 5)
  assert repart_zones['zoneA'] == [34, 33, 0, 0, 33]  
  assert repart_zones['zoneB'] == [0, 0, 50, 0, 0]   
  assert repart_zones['zoneC'] == [0, 0, 0, 50,0]   

  # Test case were a zone becomes small enought to fit on one rank
  diczone = {'zoneA': 10, 'zoneB': 200, 'zoneC': 100}
  repart_zones = MZB.balance_with_non_uniform_weights(diczone, 3)
  assert repart_zones['zoneA'] == [10, 0, 0]
  assert repart_zones['zoneB'] == [92, 83, 25]
  assert repart_zones['zoneC'] == [0, 21, 79]
