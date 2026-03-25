import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT

import Pypdm.Pypdm as PDM

from maia.algo.part import cgns_to_pdm_pmesh as PMESH

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("use_ordinal", [False, True])
def test_cgns_to_pdm_pmesh(use_ordinal, comm):
  tree  = maia.factory.generate_dist_block(6, 'TETRA_4', comm)
  ptree = maia.factory.partition_dist_tree(tree, comm)

  for i,name in enumerate(['Xmin', 'Ymin', 'Zmin', 'Xmax', 'Ymax', 'Zmax']):
    for zone in PT.get_all_Zone_t(ptree):
      if (bc := PT.get_node_from_name(zone, name)) is not None:
        PT.new_child(bc, 'Ordinal', 'Ordinal_t', i if name != 'Zmax' else 11)

  pmesh = PMESH.part_zones_to_pdm_pmesh_nodal(PT.get_all_Zone_t(ptree), comm, True, use_ordinal)

  try:
    assert pmesh.get_n_group(PDM._PDM_GEOMETRY_KIND_VOLUMIC) == 0
    assert pmesh.get_n_group(PDM._PDM_GEOMETRY_KIND_SURFACIC) == (12 if use_ordinal else 6)
  except AttributeError:
    pass # Missing API in old versions of PDM

def test_pmesh_to_cgns(comm):
  tree = maia.factory.generate_dist_block(11, 'TETRA_4', comm)
  dzone = PT.get_all_Zone_t(tree)[0]

  # Get a PMN from paradigm (here using MultiPart)
  from maia.algo.dist import cgns_to_pdm_dmesh as DMESH
  dmn = DMESH.cgns_dist_zone_to_pdm_dmesh_nodal(dzone, comm, True, True)
  mpart = PDM.MultiPart(1,
                        np.array([1], np.int32),
                        0,
                        PDM.MultiPart.HILBERT,
                        PDM.MultiPart.HOMOGENEOUS,
                        np.ones(1),
                        comm)
  mpart.dmesh_nodal_set(0, dmn)
  mpart.compute()
  pmn = mpart.part_mesh_nodal_get(0)
  
  pzones = PMESH.pdm_pmesh_nodal_to_part_zones(pmn, comm, zone_name='MyZone')
  assert isinstance(pzones, list) and len(pzones) == 1
  pzone = pzones[0]

  assert PT.get_name(pzone) == 'MyZone.P0.N0'

  assert PT.Zone.n_cell(pzone) == PT.Zone.n_cell(dzone)
  assert PT.Zone.n_vtx(pzone) == PT.Zone.n_vtx(dzone)
  
  elts = PT.get_nodes_from_label(pzone, 'Elements_t')
  assert len(elts) == 2 and [PT.get_name(e) for e in elts] == ['TETRA_4.0', 'TRI_3.0']
  assert (PT.Element.Range(elts[0]) == [1, 5000]).all() # TETRA
  assert (PT.Element.Range(elts[1]) == [5001, 6200]).all() # TRI

  bcs = PT.get_nodes_from_label(pzone, 'BC_t')
  assert len(bcs) == 6
  assert all(PT.Subset.GridLocation(bc) == 'FaceCenter' for bc in bcs)

def test_pmesh_to_cgns_2d(comm):
  tree = maia.factory.generate_dist_block(11, 'TRI_3', comm, origin=(0., 0.))
  dzone = PT.get_all_Zone_t(tree)[0]

  # Get a PMN from paradigm (here using MultiPart)
  from maia.algo.dist import cgns_to_pdm_dmesh as DMESH
  dmn = DMESH.cgns_dist_zone_to_pdm_dmesh_nodal(dzone, comm, True, True)
  mpart = PDM.MultiPart(1,
                        np.array([2], np.int32),
                        0,
                        PDM.MultiPart.HILBERT,
                        PDM.MultiPart.HOMOGENEOUS,
                        np.ones(1),
                        comm)
  mpart.dmesh_nodal_set(0, dmn)
  mpart.compute()
  pmn = mpart.part_mesh_nodal_get(0)
  pzones = PMESH.pdm_pmesh_nodal_to_part_zones(pmn, comm, phy_dim=2)
  assert isinstance(pzones, list) and len(pzones) == 2
  pz1, pz2 = pzones

  assert PT.get_name(pz1) == 'zone.P0.N0'
  assert PT.get_name(pz2) == 'zone.P0.N1'

  assert PT.Zone.n_cell(pz1) == 100 and PT.Zone.n_vtx(pz1) == 66
  assert PT.Zone.n_cell(pz2) == 100 and PT.Zone.n_vtx(pz2) == 66
  
  for pzone in pzones:
    assert PT.Zone.coordinates(pzone)[2] is None # No CZ because phy_dim=2

    elts = PT.get_nodes_from_label(pzone, 'Elements_t')
    assert len(elts) == 2 and [PT.get_name(e) for e in elts] == ['TRI_3.0', 'BAR_2.0']

    bcs = PT.get_nodes_from_label(pzone, 'BC_t')
    assert all(PT.Subset.GridLocation(bc) == 'EdgeCenter' for bc in bcs)
