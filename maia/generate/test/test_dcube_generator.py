import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import Converter.Internal as I
from maia.sids                               import elements_utils as EU

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

@mark_mpi_test([1,3])
@pytest.mark.parametrize("cgns_elmt_type", ["TRI_3", "QUAD_4", "TETRA_4", "PENTA_6", "HEXA_8"])
def test_dcube_nodal_generate(sub_comm, cgns_elmt_type):
  # Do not test value since this is a PDM function
  dist_tree = dcube_generator.dcube_nodal_generate(5, 1., [0., 0., 0.], cgns_elmt_type, sub_comm)

  zones = I.getZones(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert I.getNodeFromPath(zone, ':CGNS#Distribution/Vertex') is not None
  assert I.getNodeFromPath(zone, ':CGNS#Distribution/Cell') is not None
  if(cgns_elmt_type == "TRI_3" or cgns_elmt_type == "QUAD_4" ):
    assert len(I.getNodesFromType(zone, 'BC_t')) == 4
    assert len(I.getNodesFromType(zone, 'Elements_t')) == 5
  else:
    assert len(I.getNodesFromType(zone, 'BC_t')) == 6
    assert len(I.getNodesFromType(zone, 'Elements_t')) == 7

