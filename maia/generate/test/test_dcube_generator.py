from pytest_mpi_check._decorator import mark_mpi_test
import Converter.Internal as I

from maia.generate import dcube_generator

@mark_mpi_test([1,3])
def test_dcube_generate(sub_comm):
  # Do not test value since this is a PDM function
  dist_tree = dcube_generator.dcube_generate(5, 1., [0., 0., 0.], sub_comm)

  zones = I.getZones(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex') is not None
  assert I.getNodeFromPath(zone, ':CGNS#Distribution/Cell') is not None
  assert len(I.getNodesFromType(zone, 'BC_t')) == 6
  assert I.getNodeFromPath(zone, 'NGonElements/ParentElements')[1].shape[0] + 1 == \
         I.getNodeFromPath(zone, 'NGonElements/ElementStartOffset')[1].shape[0]
