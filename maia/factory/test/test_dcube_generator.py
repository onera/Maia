import pytest
import pytest_parallel
import maia.pytree        as PT

from maia.factory import dcube_generator

def check_dims(tree, expected_cell_dim, expected_phy_dim):
  base = PT.get_child_from_label(tree, 'CGNSBase_t')
  zone = PT.get_child_from_label(base, 'Zone_t')

  assert tuple(PT.get_value(base)) == (expected_cell_dim, expected_phy_dim)
  assert PT.get_value(zone).shape[0] == expected_cell_dim
  
  assert (PT.get_node_from_name(zone, 'CoordinateZ') is not None) == (expected_phy_dim >= 3)
  assert (PT.get_node_from_name(zone, 'CoordinateY') is not None) == (expected_phy_dim >= 2)

  assert (PT.get_node_from_name(zone, 'Zmax') is not None) == (expected_cell_dim >= 3)
  assert (PT.get_node_from_name(zone, 'Ymin') is not None) == (expected_cell_dim >= 2)

  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 2*expected_cell_dim
  assert (PT.get_node_from_name(zone, 'Face') is not None) == (expected_cell_dim >= 3) # Distribution

@pytest_parallel.mark.parallel([1,3])
def test_dcube_generate(comm):
  # Do not test value since this is a PDM function
  dist_tree = dcube_generator.dcube_generate(5, 1., [0., 0., 0.], comm)

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell') is not None
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 6
  assert PT.get_node_from_path(zone, 'NGonElements/ParentElements')[1].shape[0] + 1 == \
         PT.get_node_from_path(zone, 'NGonElements/ElementStartOffset')[1].shape[0]

@pytest_parallel.mark.parallel([2])
def test_dcube_S_generate(comm):

  # Basic version
  tree = dcube_generator.generate_dist_block([15,13,12], "S", comm, origin=[0., 0.,0])
  check_dims(tree, 3, 3)
  tree = dcube_generator.generate_dist_block([15,13], "S", comm, origin=[0., 0.])
  check_dims(tree, 2, 2)
  tree = dcube_generator.generate_dist_block([15,13,1], "S", comm, origin=[0., 0.,0])
  check_dims(tree, 2, 3)
  tree = dcube_generator.generate_dist_block([15], "S", comm, origin=[0.])
  check_dims(tree, 1, 1)
  tree = dcube_generator.generate_dist_block([15,1], "S", comm, origin=[0., 0.])
  check_dims(tree, 1, 2)
  tree = dcube_generator.generate_dist_block([15,1, 1], "S", comm, origin=[0., 0.,0])
  check_dims(tree, 1, 3)

  # Shortcut version
  tree = dcube_generator.generate_dist_block(10, "S", comm, origin=[0., 0.,0]) # 3, 3
  check_dims(tree, 3, 3)
  tree = dcube_generator.generate_dist_block(10, "S", comm, origin=[0., 0.]) # 2, 2
  check_dims(tree, 2, 2)
  tree = dcube_generator.generate_dist_block(10, "S", comm, origin=[0.]) # 1, 1
  check_dims(tree, 1, 1)

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("cgns_elmt_name", ["TRI_3", "QUAD_4", "TETRA_4", "PENTA_6", "HEXA_8"])
def test_dcube_nodal_generate(comm, cgns_elmt_name):
  # Do not test value since this is a PDM function
  dist_tree = dcube_generator.dcube_nodal_generate(5, 1., [0., 0., 0.], cgns_elmt_name, comm)

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell')[1][2] > 0
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 4 if cgns_elmt_name in ['TRI_3', 'QUAD_4'] else 6
  #For PENTA_6 we have 1 volumic and 2 surfacic
  assert len(PT.get_children_from_label(zone, 'Elements_t')) == 3 if cgns_elmt_name == 'PENTA_6' else 2

@pytest_parallel.mark.parallel([2])
def test_dcube_nodal_generate_ridges(comm):
  # Do not test value since this is a PDM function
  dist_tree = dcube_generator.dcube_nodal_generate(5, 1., [0., 0., 0.], 'PYRA_5', comm, get_ridges=True)

  zone = PT.get_all_Zone_t(dist_tree)[0]
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 6
  assert PT.get_names(PT.get_children_from_label(zone, 'Elements_t')) == \
                   ['PYRA_5.0', 'TRI_3.0', 'QUAD_4.1', 'BAR_2.0', 'NODE.0']
