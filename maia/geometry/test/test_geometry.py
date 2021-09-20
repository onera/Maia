import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import numpy as np

import Converter.Internal as I

from maia.generate.dcube_generator import dcube_generate

from maia.geometry import geometry

@mark_mpi_test(1)
def test_compute_cell_center(sub_comm):
  tree = dcube_generate(3, 1., [0,0,0], sub_comm)
  zone = I.getZones(tree)[0]

  cell_center = geometry.compute_cell_center(zone)
  expected_cell_center = np.array([0.25, 0.25, 0.25, 
                                   0.75, 0.25, 0.25, 
                                   0.25, 0.75, 0.25, 
                                   0.75, 0.75, 0.25, 
                                   0.25, 0.25, 0.75, 
                                   0.75, 0.25, 0.75, 
                                   0.25, 0.75, 0.75, 
                                   0.75, 0.75, 0.75])
  assert (cell_center == expected_cell_center).all()

  zone_no_ng = I.rmNodesByType(zone, 'Elements_t')
  with pytest.raises(NotImplementedError):
    geometry.compute_cell_center(zone_no_ng)
