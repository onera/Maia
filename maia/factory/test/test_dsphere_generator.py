import pytest
from pytest_mpi_check._decorator import mark_mpi_test
import maia.pytree        as PT

from maia.factory import dsphere_generator

@mark_mpi_test(3)
@pytest.mark.parametrize("elmt_name", ["TRI_3", "NGON_n"])
def test_dsphere_surf_generate(elmt_name, sub_comm):
  # Do not test value since this is a PDM function
  dist_tree = dsphere_generator.generate_dist_sphere(6, elmt_name, sub_comm)
  assert (PT.get_all_CGNSBase_t(dist_tree)[0][1] == [2,3]).all()

  assert len(PT.get_all_Zone_t(dist_tree)) == 1
  zone = PT.get_all_Zone_t(dist_tree)[0]

  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell') is not None

  if elmt_name == "TRI_3":
    surf_elt = PT.get_node_from_name(dist_tree, 'TRI_3.0')
  else:
    surf_elt = PT.get_node_from_name(dist_tree, 'NGonElements')
  assert PT.Element.Size(surf_elt) == 720

  assert len(PT.get_nodes_from_label(dist_tree, 'BC_t')) == 0

@mark_mpi_test(3)
@pytest.mark.parametrize("elmt_name", ["TETRA_4", "NFACE_n"])
def test_dsphere_vol_generate(elmt_name, sub_comm):
  # Do not test value since this is a PDM function
  dist_tree = dsphere_generator.generate_dist_sphere(6, elmt_name, sub_comm)

  assert len(PT.get_all_Zone_t(dist_tree)) == 1
  zone = PT.get_all_Zone_t(dist_tree)[0]

  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell') is not None

  if elmt_name == "TETRA_4":
    vol_elt = PT.get_node_from_name(dist_tree, 'TETRA_4.0')
    surf_elt = PT.get_node_from_name(dist_tree, 'TRI_3.0')
    assert PT.Element.Size(surf_elt) == 720
  else:
    vol_elt = PT.get_node_from_name(dist_tree, 'NFaceElements')
    surf_elt = PT.get_node_from_name(dist_tree, 'NGonElements')
    assert PT.Element.Size(surf_elt) == 9000
  assert PT.Element.Size(vol_elt) == 4320

  bc = PT.get_node_from_name(dist_tree, 'Skin')
  assert PT.get_node_from_name(bc, 'Index')[1][2] == 720

@mark_mpi_test(3)
@pytest.mark.parametrize("elmt_name", ["TETRA_4", "NFACE_n"])
def test_dsphere_hollow_generate(elmt_name, sub_comm):
  # Do not test value since this is a PDM function
  dist_tree = dsphere_generator.dsphere_hollow_nodal_generate(5, .5, 2, [0., 0., 0.], sub_comm, 4, .5)

  assert len(PT.get_all_Zone_t(dist_tree)) == 1
  zone = PT.get_all_Zone_t(dist_tree)[0]

  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Vertex') is not None
  assert PT.get_node_from_path(zone, ':CGNS#Distribution/Cell') is not None

  vol_elt = PT.get_node_from_name(dist_tree, 'PENTA_6.0')
  surf_elt = PT.get_node_from_name(dist_tree, 'TRI_3.0')
  assert PT.Element.Size(surf_elt) == 1440
  assert PT.Element.Size(vol_elt) == 2880

  assert len(PT.get_nodes_from_label(dist_tree, 'BC_t')) == 2

