from mpi4py import MPI
import pytest
import pytest_parallel
import numpy as np

import maia
import maia.pytree as PT

from   maia.algo.part import extract_part as EP
from   maia.factory  import dist_from_part

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

  extract_part_tree = EP.extract_part_from_bc_name(part_tree, bc_name, comm)
  extract_dist_tree = maia.factory.recover_dist_tree(extract_part_tree, comm)
  extract_dist_zone = PT.get_all_Zone_t(extract_dist_tree)[0]
  assert PT.Zone.n_vtx( extract_dist_zone)==100
  assert PT.Zone.n_cell(extract_dist_zone)==81

@pytest_parallel.mark.parallel([1,3])
@pytest.mark.parametrize("bc_loc" , ['Vertex','FaceCenter'])
@pytest.mark.parametrize("bc_name", ['Xmin','Xmax','Ymin','Ymax','Zmin','Zmax'])
def test_exch_field(bc_loc, bc_name, comm):
  part_tree = sample_part_tree(comm, bc_loc)
  
  # > Initialize flow solution
  for part_zone in PT.get_all_Zone_t(part_tree):
    cx, cy, cz = PT.Zone.coordinates(part_zone)
    PT.new_FlowSolution('FlowSol#Vtx', loc='Vertex', fields={'cx':cx}, parent=part_zone)
  
  # > Get point range
  point_range = list()
  part_tree_per_dom = dist_from_part.get_parts_per_blocks(part_tree, comm)
  for domain, part_zones in part_tree_per_dom.items():
    point_range_domain = list()
    for part_zone in part_zones:
      bc_n = PT.get_node_from_name_and_label(part_zone, bc_name, 'BC_t')
      bc_pr = PT.get_value(PT.get_child_from_name(bc_n, 'PointRange')) if bc_n is not None else np.empty(0, np.int32)
      point_range_domain.append(bc_pr)
    point_range.append(point_range_domain)

  extractor = EP.Extractor(part_tree, point_range, bc_loc, comm)
  extractor.exchange_fields(['FlowSol#Vtx'])
  extract_part_tree = extractor.get_extract_part_tree()

  extract_dist_tree = maia.factory.recover_dist_tree(extract_part_tree, comm)
  extract_dist_zone = PT.get_all_Zone_t(extract_dist_tree)[0]
  assert PT.Zone.n_vtx( extract_dist_zone)==100
  assert PT.Zone.n_cell(extract_dist_zone)==81
  coord_x,_,_ = PT.Zone.coordinates(extract_dist_zone)
  field_x = PT.get_node_from_path(extract_dist_zone, 'FlowSol#Vtx/cx')[1]
  assert np.array_equal(coord_x, field_x)
