from pytest_mpi_check._decorator import mark_mpi_test
import Converter.Internal as I

from maia.generate import dplane_generator

@mark_mpi_test([1,3])
def test_dplane_generate(sub_comm):
  # Do not test value since this is a PDM function
  xmin, xmax = 0., 1.
  ymin, ymax = 0., 1.
  nx, ny = 8, 8
  have_random = 0
  init_random = 1
  dist_tree = dplane_generator.dplane_generate(xmin, xmax, ymin, ymax,\
      have_random, init_random, nx, ny, sub_comm)

  zones = I.getZones(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex') is not None
  assert I.getNodeFromPath(zone, ':CGNS#Distribution/Cell') is not None
  assert len(I.getNodesFromType(zone, 'BC_t')) == 4
  assert I.getNodeFromPath(zone, 'NGonElements/ParentElements')[1].shape[0] + 1 == \
         I.getNodeFromPath(zone, 'NGonElements/ElementStartOffset')[1].shape[0]
