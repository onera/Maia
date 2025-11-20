import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.algo.dist import multigrid_s

@pytest_parallel.mark.parallel(2)
def test_multigrid_s_2D(comm):
  n_lvl = 3
  coeff = 2**n_lvl
  n1 = 8
  n2 = 4
  tree = maia.factory.generate_dist_block([coeff*n1+1,coeff*n2+1], "S", comm)

  # Split a bc in two for test
  zbc = PT.find_node_from_name(tree, 'ZoneBC')
  bc1 = PT.find_child_from_name(zbc, 'Xmax')
  bc2 = PT.deep_copy(bc1)
  pr1 = PT.get_np_value(PT.find_child_from_name(bc1, 'PointRange'))
  pr2 = PT.get_np_value(PT.find_child_from_name(bc2, 'PointRange'))
  PT.set_name(bc1, 'Xmax_1')
  PT.set_name(bc2, 'Xmax_2')
  pr1[1,1] = coeff + 1
  pr2[1,0] = coeff + 1
  others = PT.get_children_from_predicate(zbc, ~PT.pred.name_matches('Xmax*'))
  PT.set_children(zbc, others + [bc1, bc2])


  trees = maia.algo.dist.multigrid_s(tree, n_lvl, comm)

  # Check computed coarse index (on thin grid): we compute actual cell idx
  # on coarse mesh, and move it on thin mesh by interpolation to compare.
  for i, tree in enumerate(trees[:-1]):
    coarse = PT.shallow_copy(trees[i+1])
    from maia.utils import s_numbering
    coarse_zone = PT.get_all_Zone_t(coarse)[0]
    coarse_distri = MT.distribution_value(coarse_zone, 'Cell')
    fi, fj = s_numbering.index_to_ij(np.arange(coarse_distri[0]+1, coarse_distri[1]+1),
                                     PT.Zone.CellSize(coarse_zone))
    PT.new_FlowSolution('CoarseId',
                        loc='CellCenter',
                        fields={'I': fi, 'J': fj},
                        parent=coarse_zone)

    maia.algo.interpolate(coarse, tree, comm, ['CoarseId'], 'CellCenter')

    expected_i = PT.get_np_value(PT.find_node_from_path(tree, 'Base/zone/CoarseId/I'))
    computed_i = PT.get_np_value(PT.find_node_from_path(tree, 'Base/zone/MultiGridCellInfo/ICoarseIdx'))
    assert np.array_equal(computed_i, expected_i)

    expected_j = PT.get_np_value(PT.find_node_from_path(tree, 'Base/zone/CoarseId/J'))
    computed_j = PT.get_np_value(PT.find_node_from_path(tree, 'Base/zone/MultiGridCellInfo/JCoarseIdx'))
    assert np.array_equal(computed_j, expected_j)

  # Check splited BC sizes (small subset is 25% total size)
  for i, tree in enumerate(trees):
    bc1 = PT.find_node_from_name(tree, 'Xmax_1')
    bc2 = PT.find_node_from_name(tree, 'Xmax_2')
    assert 3*(PT.Subset.n_elem(bc1)-1) == (PT.Subset.n_elem(bc2) - 1)
    assert (PT.Subset.n_elem(bc1) + PT.Subset.n_elem(bc2) - 1) == 2**(n_lvl-i) * n2 + 1
  

@pytest_parallel.mark.parallel(2)
def test_multigrid_s(comm):
  
  nb_vtx_per_dir = 5 # 5 = 4n+1 avec n=1
  nb_lvl = 2
  
  dt = maia.factory.generate_dist_block(nb_vtx_per_dir, "S", comm)
  trees = maia.algo.dist.multigrid_s(dt, nb_lvl, comm)
  
  assert(len(trees) == nb_lvl+1)
  
  for t, tree in enumerate(trees):
    zone = PT.find_node_from_path(tree, 'Base/zone')
    assert PT.Zone.VertexSize(zone) == (2**(2-t)+1, 2**(2-t)+1, 2**(2-t)+1)
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
  
  icoarse_ids_lvl0 = PT.get_value(PT.get_node_from_path(trees[0], 'Base/zone/MultiGridCellInfo/ICoarseIdx'))
  icoarse_ids_lvl0_ref = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
  assert np.all(icoarse_ids_lvl0 == icoarse_ids_lvl0_ref)
  
  jcoarse_ids_lvl0 = PT.get_value(PT.get_node_from_path(trees[0], 'Base/zone/MultiGridCellInfo/JCoarseIdx'))
  jcoarse_ids_lvl0_ref = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
  assert np.all(jcoarse_ids_lvl0 == jcoarse_ids_lvl0_ref)
  
  kcoarse_ids_lvl0 = PT.get_value(PT.get_node_from_path(trees[0], 'Base/zone/MultiGridCellInfo/KCoarseIdx'))
  kcoarse_ids_lvl0_ref = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
  assert np.all(kcoarse_ids_lvl0 == kcoarse_ids_lvl0_ref+comm.rank)
  
  icoarse_ids_lvl1 = PT.get_value(PT.get_node_from_path(trees[1], 'Base/zone/MultiGridCellInfo/ICoarseIdx'))
  assert np.all(icoarse_ids_lvl1==1)
  jcoarse_ids_lvl1 = PT.get_value(PT.get_node_from_path(trees[1], 'Base/zone/MultiGridCellInfo/JCoarseIdx'))
  assert np.all(jcoarse_ids_lvl1==1)
  kcoarse_ids_lvl1 = PT.get_value(PT.get_node_from_path(trees[1], 'Base/zone/MultiGridCellInfo/KCoarseIdx'))
  assert np.all(kcoarse_ids_lvl1==1)
  
  assert PT.get_node_from_path(trees[nb_lvl], 'Base/zone/MultiGridCellInfo') == None
