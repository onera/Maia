import pytest
import numpy as np

import maia.pytree as PT

from maia.pytree.maia import metrics

def test_dtree_nbytes():
  zone = PT.new_Zone('Zone', type='Unstructured')
  PT.new_GridCoordinates(fields={'cx':np.ones(100, dtype=float)}, parent=zone)
  assert metrics.dtree_nbytes(zone) == (1105, 12, 800)
