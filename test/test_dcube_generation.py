import pytest
import os
from   pytest_mpi_check._decorator import mark_mpi_test

import maia.pytree        as PT
import maia.pytree.maia   as MT


import maia
from maia.factory import generate_dist_block
from maia.factory import dplane_generator as DPG

"""
Some regular meshes can be directly generated in their distributed version
"""

@mark_mpi_test([1,3])
def test_generate_dcube_ngons(sub_comm, write_output):
  n_vtx = 20

  # > dcube_generate create a NGon discretisation of a cube
  dist_tree = generate_dist_block(n_vtx, "Poly", sub_comm, origin=[0.,0.,0.], edge_length=1.)

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert PT.Zone.n_vtx(zone) == n_vtx ** 3

  assert PT.get_child_from_name(zone, 'NGonElements') is not None

  # > The cube include 6 boundary groups
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 6

  assert MT.getDistribution(zone) is not None
  # > Distribution dtype should be consistent with PDM
  assert MT.getDistribution(zone, 'Vertex')[1].dtype == maia.npy_pdm_gnum_dtype

  if write_output:
    out_dir = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    out_file = os.path.join(out_dir, f'dcube_ngon.hdf')
    maia.io.write_trees(dist_tree, out_file, sub_comm)
    maia.io.dist_tree_to_file(dist_tree, out_file, sub_comm)
    
@pytest.mark.parametrize("cgns_elmt_name", ["TRI_3", "QUAD_4", "TETRA_4", "PENTA_6", "HEXA_8"])
@mark_mpi_test([2])
def test_generate_dcube_elts(cgns_elmt_name, sub_comm, write_output):
  n_vtx = 20

  # > dcube_nodal_generate create an element discretisation of a cube. Several element type are supported
  dist_tree = generate_dist_block(n_vtx, cgns_elmt_name, sub_comm, origin=[0.,0.,0.], edge_length=1.)

  # 2D or 3D meshes can be generated, depending on the type of requested element
  dim = 2 if cgns_elmt_name in ["TRI_3", "QUAD_4"] else 3
  assert (PT.get_value(PT.get_all_CGNSBase_t(dist_tree)[0]) == [dim,3]).all()

  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert PT.Zone.n_vtx(zone) == n_vtx ** dim

  assert PT.get_child_from_name(zone, 'NGonElements') is None
  elem_nodes = PT.get_nodes_from_label(zone, 'Elements_t')
  # > Volumic + boundary elements are defined in the mesh (all boundary are merged)
  n_bnd_elem_node = 2 if cgns_elmt_name == "PENTA_6" else 1
  assert len(elem_nodes) == 1 + n_bnd_elem_node
  main_elem_n = [e for e in elem_nodes if PT.Element.CGNSName(e) == cgns_elmt_name]
  assert len(main_elem_n) == 1

  assert MT.getDistribution(zone) is not None
  # > Distribution dtype should be consistent with PDM
  assert MT.getDistribution(zone, 'Vertex')[1].dtype == maia.npy_pdm_gnum_dtype

  if write_output:
    out_dir = maia.utils.test_utils.create_pytest_output_dir(sub_comm)
    outfile = os.path.join(out_dir, 'dcube_elt.hdf')
    maia.io.dist_tree_to_file(dist_tree, outfile, sub_comm)

@pytest.mark.parametrize("random", [False, True])
@mark_mpi_test([3])
def test_generate_place_ngons(random, sub_comm):
  n_vtx = 20

  # > dplane_generate create a strange 2D discretisation with polygonal elements
  dist_tree = DPG.dplane_generate(xmin=0., xmax=1., ymin=0., ymax=1., \
      have_random=random, init_random=random, nx=n_vtx, ny=n_vtx, comm=sub_comm)

  assert (PT.get_value(PT.get_all_CGNSBase_t(dist_tree)[0]) == [2,2]).all()
  zones = PT.get_all_Zone_t(dist_tree)
  assert len(zones) == 1
  zone = zones[0]

  assert PT.get_child_from_name(zone, 'NGonElements') is not None

  # > The mesh include 4 boundary groups
  assert len(PT.get_nodes_from_label(zone, 'BC_t')) == 4

  assert MT.getDistribution(zone) is not None
  # > Distribution dtype should be consistent with PDM
  assert MT.getDistribution(zone, 'Vertex')[1].dtype == maia.npy_pdm_gnum_dtype
