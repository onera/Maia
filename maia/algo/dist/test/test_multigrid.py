import pytest
import pytest_parallel
from   mpi4py import MPI
import numpy as np

import maia
import maia.pytree as PT

from maia.algo.dist import multigrid_s


@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("nb_lvl", [1,2])
def test_multigrid_s(nb_lvl, comm):
  
  dt = maia.factory.generate_dist_block(5, "S", comm) # 5 = 4n+1 avec n=1
  trees = maia.algo.dist.multigrid_s(dt, nb_lvl, comm)
  
  assert(len(trees) == nb_lvl+1)
  
  for t, tree in enumerate(trees):
    zone = PT.get_node_from_path(tree, 'Base/zone')
    assert np.all(PT.Zone.VertexSize(zone) == np.array([2**(2-t)+1, 2**(2-t)+1, 2**(2-t)+1]))
    cx,cy,cz = PT.Zone.coordinates(zone)
    if t == 0:
      assert np.all(np.isin(np.unique(cx), [0, 0.25, 0.5, 0.75, 1.]))
      assert np.all(np.isin(np.unique(cy), [0, 0.25, 0.5, 0.75, 1.]))
      assert np.all(np.isin(np.unique(cz), [0, 0.25, 0.5, 0.75, 1.]))
    elif t == 1:
      assert np.all(np.isin(np.unique(cx), [0, 0.5, 1.]))
      assert np.all(np.isin(np.unique(cy), [0, 0.5, 1.]))
      assert np.all(np.isin(np.unique(cz), [0, 0.5, 1.]))
    elif t == 2:
      assert np.all(np.isin(np.unique(cx), [0, 1.]))
      assert np.all(np.isin(np.unique(cy), [0, 1.]))
      assert np.all(np.isin(np.unique(cz), [0, 1.]))
  
  assert PT.get_node_from_path(trees[0], 'Base/zone/MultiGridCellInfo/CurUnstIdx') is not None
  
  coarse_ids_lvl0 = PT.get_value(PT.get_node_from_path(trees[0], 'Base/zone/MultiGridCellInfo/CoarseUnstIdx'))
  # assert np.all(coarse_ids_lvl0<9)
  coarse_ids_lvl0_ref = np.array([1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4])
  assert np.all(coarse_ids_lvl0 == coarse_ids_lvl0_ref+4*comm.rank)
  
  if nb_lvl == 2:
    coarse_ids_lvl1 = PT.get_value(PT.get_node_from_path(trees[1], 'Base/zone/MultiGridCellInfo/CoarseUnstIdx'))
    assert np.all(coarse_ids_lvl1==1)
  
  assert PT.get_node_from_path(trees[nb_lvl], 'Base/zone/MultiGridCellInfo') == None
