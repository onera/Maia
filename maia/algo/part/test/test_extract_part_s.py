from mpi4py import MPI
import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT

from maia.algo.part import extract_part as EP

def sample_part_tree(comm, bc_loc):
  dist_tree = maia.factory.dcube_generator.dcube_struct_generate(10, 1., [0.,0.,0.], comm, bc_location=bc_loc)
  part_opts = maia.factory.partitioning.compute_regular_weights(dist_tree, comm, n_part=4)
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=part_opts)
  return part_tree

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("bc_loc" , ['Vertex','FaceCenter'])
@pytest.mark.parametrize("bc_name", ['Xmin','Xmax','Ymin','Ymax','Zmin','Zmax'])
def test_extract_part_simple(bc_loc, bc_name, comm):
  part_tree = sample_part_tree(comm, bc_loc)

  extract_part_tree = maia.algo.part.extract_part_s.extract_part_s_from_bc_name(part_tree, bc_name, comm)
  extract_dist_tree = maia.factory.recover_dist_tree(extract_part_tree, comm)
  extract_dist_zone = PT.get_all_Zone_t(extract_dist_tree)[0]
  assert PT.Zone.n_vtx( extract_dist_zone)==100
  assert PT.Zone.n_cell(extract_dist_zone)==81
