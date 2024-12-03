import pytest
import pytest_parallel

import maia
import maia.pytree as PT
from   maia.factory.dline_generator import generate_dist_line

import numpy as np


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("phy_dim", [1,2,3])
def test_generate_line(phy_dim, comm):

  start = np.array([0.,0.,0.][:phy_dim])
  end   = np.array([1.,2.,0.][:phy_dim])
  dist_tree = generate_dist_line(5, start, end, comm)

  zone = PT.get_node_from_label(dist_tree, "Zone_t")
  base = PT.get_node_from_label(dist_tree, "CGNSBase_t")
  assert (PT.get_value(base) == [1, phy_dim]).all()
  assert PT.Zone.n_vtx (zone)==5
  assert PT.Zone.n_cell(zone)==4

  assert len([c for c in PT.Zone.coordinates(zone) if c is None]) == 3-phy_dim