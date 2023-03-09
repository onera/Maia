import pytest

from maia.factory.partitioning.load_balancing import multi_zone_balancing as ZB

def test_karmarkar_karp():
    assert ZB.karmarkar_karp([5,9,2,4], 1) == [[1,0,3,2]]
    assert ZB.karmarkar_karp([5,9,2,4], 4) == [[2],[3],[0],[1]]
    assert ZB.karmarkar_karp([5,9,2,4], 2) == [[3,0], [2,1]]
