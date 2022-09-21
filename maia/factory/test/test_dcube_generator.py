import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import maia.pytree        as PT

from maia.factory import dcube_generator

@mark_mpi_test([1,3])
def test_dcube_generate(sub_comm):
  # Do not test value since this is a PDM function
  dist_tree = dcube_generator.dcube_generate(5, 1., [0., 0., 0.], sub_comm)

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell') is not None
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 6
  assert PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1].shape[0] + 1 == \
         PT.get_node_from_path(zone, 'NGonElements/ElementStartOffset')[1].shape[0]

@mark_mpi_test([1,3])
@pytest.mark.parametrize("cgns_elmt_name", ["TRI_3", "QUAD_4", "TETRA_4", "PENTA_6", "HEXA_8"])
def test_dcube_nodal_generate(sub_comm, cgns_elmt_name):
  # Do not test value since this is a PDM function
  dist_tree = dcube_generator.dcube_nodal_generate(5, 1., [0., 0., 0.], cgns_elmt_name, sub_comm)

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell') is not None
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 4 if cgns_elmt_name in ['TRI_3', 'QUAD_4'] else 6
  #For PENTA_6 we have 1 volumic and 2 surfacic
  assert len(PT.get_children_from_label(zone, 'Elements_t')) == 3 if cgns_elmt_name == 'PENTA_6' else 2

@mark_mpi_test([2])
def test_dcube_nodal_generate_ridges(sub_comm):
  # Do not test value since this is a PDM function
  dist_tree = dcube_generator.dcube_nodal_generate(5, 1., [0., 0., 0.], 'PYRA_5', sub_comm, get_ridges=True)

  zone = PT.get_all_Zone_t(dist_tree)[0]
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 6
  assert PT.get_names(PT.get_children_from_label(zone, 'Elements_t')) == \
                   ['PYRA_5.0', 'TRI_3.0', 'QUAD_4.1', 'BAR_2.0', 'NODE.0']
