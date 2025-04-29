import pytest
import pytest_parallel
import numpy as np

import maia.pytree as PT

import maia

from maia.algo import geosearch as GS



@pytest_parallel.mark.parallel(2)
def test_exclusive(comm):
  dtree_src = maia.factory.generate_dist_block(5, 'Poly', comm)
  dtree_tgt = maia.factory.generate_dist_block(5, 'Poly', comm)

  ptree_src = maia.factory.partition_dist_tree(dtree_src, comm) 
  ptree_tgt = maia.factory.partition_dist_tree(dtree_tgt, comm) 

  with pytest.raises(ValueError):
    GS.localize_points(dtree_src, ptree_tgt, 'Vertex', comm)
  with pytest.raises(ValueError):
    GS.find_closest_points(ptree_src, dtree_tgt, 'CellCenter', comm)