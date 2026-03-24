import pytest
import pytest_parallel

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

  pmesh = PMESH.cgns_part_zones_to_pdm_pmesh_nodal(PT.get_all_Zone_t(ptree), comm, True, use_ordinal)

  try:
    assert pmesh.get_n_group(PDM._PDM_GEOMETRY_KIND_VOLUMIC) == 0
    assert pmesh.get_n_group(PDM._PDM_GEOMETRY_KIND_SURFACIC) == (12 if use_ordinal else 6)
  except AttributeError:
    pass # Missing API in old versions of PDM