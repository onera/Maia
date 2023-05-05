import pytest
import numpy as np

import maia.pytree as PT

from maia.pytree.maia import metrics

def test_dtree_nbytes():
  zone = PT.new_Zone('Zone', type='Unstructured')
  PT.new_GridCoordinates(fields={'cx':np.ones(100, dtype=float)}, parent=zone)
  sizes = metrics.dtree_nbytes(zone)
  assert 800 <= sizes[0] and sizes[0] <= 1400 # Architecture dependant
  assert sizes[1] == 12
  assert sizes[2] == 800
