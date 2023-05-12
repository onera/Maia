import pytest_parallel
import maia.pytree as PT

from maia.factory import dplane_generator

@pytest_parallel.mark.parallel([1,3])
def test_dplane_generate(comm):
  # Do not test value since this is a PDM function
  xmin, xmax = 0., 1.
  ymin, ymax = 0., 1.
  nx, ny = 8, 8
  have_random = 0
  init_random = 1
  dist_tree = dplane_generator.dplane_generate(xmin, xmax, ymin, ymax,\
      have_random, init_random, nx, ny, comm)

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell') is not None
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 4
  assert PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1].shape[0] + 1 == \
         PT.get_node_from_path(zone, 'NGonElements/ElementStartOffset')[1].shape[0]
