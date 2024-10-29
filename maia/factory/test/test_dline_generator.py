import pytest
import pytest_parallel

import maia
import maia.pytree as PT
from   maia.factory import generate_dist_line

import numpy as np


@pytest_parallel.mark.parallel(2)
def test_generate_line(comm):

  dist_tree = generate_dist_line(np.array([0., 0., 0.]),
                                 np.array([1., 2., 0.]), 5, comm)
  zone = PT.get_node_from_label(dist_tree, "Zone_t")
  assert PT.Zone.n_vtx (zone)==5
  assert PT.Zone.n_cell(zone)==4