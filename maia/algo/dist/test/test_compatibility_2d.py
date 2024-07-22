import pytest
import pytest_parallel

import maia
import maia.pytree        as PT

from maia.algo.dist.compatibility_2d import convert_std_to_cass_2d_u, convert_cass_to_std_2d_u

is_bar   = lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'BAR_2'
is_ngon  = lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'NGON_n'
is_nface = lambda n: PT.get_label(n) == 'Elements_t' and PT.Element.CGNSName(n) == 'NFACE_n'

@pytest_parallel.mark.parallel(2)
def test_convert_std_to_cass_2d_u(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'QUAD_4', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)

  convert_std_to_cass_2d_u(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]

  assert len(PT.get_nodes_from_predicate(zone, is_bar)) == 0
  assert len(PT.get_nodes_from_predicate(zone, is_ngon)) == 1
  assert len(PT.get_nodes_from_predicate(zone, is_nface)) == 1

@pytest_parallel.mark.parallel(2)
def test_convert_cass_to_std_2d_u(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'QUAD_4', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  dist_tree_save = PT.deep_copy(dist_tree)

  convert_std_to_cass_2d_u(dist_tree, comm)
  convert_cass_to_std_2d_u(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]

  assert len(PT.get_nodes_from_predicate(zone, is_bar)) == 1
  assert len(PT.get_nodes_from_predicate(zone, is_ngon)) == 1
  assert len(PT.get_nodes_from_predicate(zone, is_nface)) == 0
  assert PT.is_same_tree(dist_tree, dist_tree_save)
