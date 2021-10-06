import pytest
from   pytest_mpi_check._decorator import mark_mpi_test

import Converter.Internal   as I

from maia         import npy_pdm_gnum_dtype
from maia.sids    import sids
from maia.sids    import Internal_ext as IE

from maia.generate import dcube_generator as DCG
from maia.generate import dplane_generator as DPG

"""
Some regular meshes can be directly generated in their distributed version
"""

@mark_mpi_test([1,3])
def test_generate_dcube_ngons(sub_comm):
  n_vtx = 20

  # > dcube_generate create a NGon discretisation of a cube
  dist_tree = DCG.dcube_generate(n_vtx, edge_length=1., origin=[0.,0.,0.], comm=sub_comm)

  zones = I.getZones(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert sids.Zone.n_vtx(zone) == n_vtx ** 3

  assert I.getNodeFromName1(zone, 'NGonElements') is not None

  # > The cube include 6 boundary groups
  assert len(I.getNodesFromType(zone, 'BC_t')) == 6

  assert IE.getDistribution(zone) is not None
  # > Distribution dtype should be consistent with PDM
  assert IE.getDistribution(zone, 'Vertex')[1].dtype == npy_pdm_gnum_dtype

@pytest.mark.parametrize("cgns_elmt_name", ["TRI_3", "QUAD_4", "TETRA_4", "PENTA_6", "HEXA_8"])
@mark_mpi_test([2])
def test_generate_dcube_elts(cgns_elmt_name, sub_comm):
  n_vtx = 20

  # > dcube_nodal_generate create an element discretisation of a cube. Several element type are supported
  dist_tree = DCG.dcube_nodal_generate(n_vtx, edge_length=1., origin=[0.,0.,0.],\
      cgns_elmt_name=cgns_elmt_name, comm=sub_comm)

  # 2D or 3D meshes can be generated, depending on the type of requested element
  dim = 2 if cgns_elmt_name in ["TRI_3", "QUAD_4"] else 3
  assert (I.getVal(I.getBases(dist_tree)[0]) == [dim,dim]).all()

  zones = I.getZones(dist_tree)
  assert len(zones) == 1
  zone = zones[0]
  assert sids.Zone.n_vtx(zone) == n_vtx ** dim

  assert I.getNodeFromName1(zone, 'NGonElements') is None
  elem_nodes = I.getNodesFromType(zone, 'Elements_t')
  # > Volumic + boundary elements are defined in the mesh
  assert len(elem_nodes) == 1 + 2*dim
  main_elem_n = [e for e in elem_nodes if sids.ElementCGNSName(e) == cgns_elmt_name]
  assert len(main_elem_n) == 1

  assert IE.getDistribution(zone) is not None
  # > Distribution dtype should be consistent with PDM
  assert IE.getDistribution(zone, 'Vertex')[1].dtype == npy_pdm_gnum_dtype

@pytest.mark.parametrize("random", [False, True])
@mark_mpi_test([3])
def test_generate_place_ngons(random, sub_comm):
  n_vtx = 20

  # > dplane_generate create a strange 2D discretisation with polygonal elements
  dist_tree = DPG.dplane_generate(xmin=0., xmax=1., ymin=0., ymax=1., \
      have_random=random, init_random=random, nx=n_vtx, ny=n_vtx, comm=sub_comm)

  assert (I.getVal(I.getBases(dist_tree)[0]) == [2,2]).all()
  zones = I.getZones(dist_tree)
  assert len(zones) == 1
  zone = zones[0]

  assert I.getNodeFromName1(zone, 'NGonElements') is not None

  # > The mesh include 4 boundary groups
  assert len(I.getNodesFromType(zone, 'BC_t')) == 4

  assert IE.getDistribution(zone) is not None
  # > Distribution dtype should be consistent with PDM
  assert IE.getDistribution(zone, 'Vertex')[1].dtype == npy_pdm_gnum_dtype
