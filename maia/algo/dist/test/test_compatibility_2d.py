import pytest
import pytest_parallel

import maia
import maia.pytree        as PT

from maia.algo.dist import compatibility_2d as cpt2d

is_bar   = PT.pred.is_elmt_of_type('BAR_2')
is_ngon  = PT.pred.is_elmt_of_type('NGON_n')
is_nface = PT.pred.is_elmt_of_type('NFACE_n')

@pytest_parallel.mark.parallel(2)
def test_convert_std_to_3dlike(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'QUAD_4', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)

  cpt2d.poly2d_convert_std_to_3dlike(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]

  assert len(PT.get_nodes_from_predicate(zone, is_bar)) == 0
  assert len(PT.get_nodes_from_predicate(zone, is_ngon)) == 1
  assert len(PT.get_nodes_from_predicate(zone, is_nface)) == 1

@pytest_parallel.mark.parallel(2)
def test_convert_3dlike_to_std(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'QUAD_4', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  dist_tree_save = PT.deep_copy(dist_tree)

  cpt2d.poly2d_convert_std_to_3dlike(dist_tree, comm)
  cpt2d.poly2d_convert_3dlike_to_std(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]

  assert len(PT.get_nodes_from_predicate(zone, is_bar)) == 1
  assert len(PT.get_nodes_from_predicate(zone, is_ngon)) == 1
  assert len(PT.get_nodes_from_predicate(zone, is_nface)) == 0
  assert PT.is_same_tree(dist_tree, dist_tree_save)
