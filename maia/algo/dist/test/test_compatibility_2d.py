import pytest
import pytest_parallel
import mpi4py.MPI as MPI
import numpy as np

import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

from maia.algo.dist.compatibility_2d import convert_std_to_cass_2d_u, convert_cass_to_std_2d_u

@pytest_parallel.mark.parallel(2)
def test_convert_std_to_cass_2d_u(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'QUAD_4', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  convert_std_to_cass_2d_u(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]
  is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
  assert (len(PT.get_nodes_from_predicate(zone, is_bar)) == 0)
  is_ngon = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 22)
  assert (len(PT.get_nodes_from_predicate(zone, is_ngon)) == 1)
  is_nface = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 23)
  assert (len(PT.get_nodes_from_predicate(zone, is_nface)) == 1)

@pytest_parallel.mark.parallel(2)
def test_convert_cass_to_std_2d_u(comm):
  dist_tree = maia.factory.generate_dist_block(11, 'QUAD_4', comm)
  maia.algo.dist.convert_elements_to_ngon(dist_tree, comm)
  dist_tree_save = PT.deep_copy(dist_tree)
  convert_std_to_cass_2d_u(dist_tree, comm)
  convert_cass_to_std_2d_u(dist_tree, comm)
  zone = PT.get_all_Zone_t(dist_tree)[0]
  is_bar = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 3)
  assert (len(PT.get_nodes_from_predicate(zone, is_bar)) == 1)
  is_ngon = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 22)
  assert (len(PT.get_nodes_from_predicate(zone, is_ngon)) == 1)
  is_nface = lambda n: (PT.get_label(n) == 'Elements_t') and (PT.get_value(n)[0] == 23)
  assert (len(PT.get_nodes_from_predicate(zone, is_nface)) == 0)
  assert PT.is_same_tree(dist_tree, dist_tree_save)
