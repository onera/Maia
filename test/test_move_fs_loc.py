import os
import pytest
import pytest_parallel

import maia.pytree        as PT

import maia
import maia.utils.test_utils as TU

ref_dir = os.path.join(os.path.dirname(__file__), 'references')

@pytest_parallel.mark.parallel(2)
@pytest.mark.parametrize("jn_loc", ["Vertex", "FaceCenter"])
def test_centers_to_node(jn_loc, comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  # This feature is U only
  dist_tree = maia.algo.dist.convert_s_to_u(dist_tree, 'NGON_n', comm, subset_loc={'GC_t':jn_loc})

  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  # Create a Centers solution
  for part in PT.get_all_Zone_t(part_tree):
    cell_center = maia.algo.part.compute_cell_center(part)
    ccx = cell_center[0::3]
    ccy = cell_center[1::3]
    ccz = cell_center[2::3]
    PT.new_FlowSolution('FSol', loc='CellCenter', fields={'cX':ccx, 'cY':ccy, 'cZ':ccz}, parent=part)

  maia.algo.part.centers_to_nodes(part_tree, comm, ['FSol'])

  # Compare with reference
  maia.transfer.part_tree_to_dist_tree_all(dist_tree, part_tree, comm)
  ref_file  = os.path.join(ref_dir, f'U_twoblocks_centerstonodes.yaml')
  ref_tree = maia.io.file_to_dist_tree(ref_file, comm)
  for zone in PT.iter_all_Zone_t(dist_tree):
    ref_zone = PT.get_node_from_name(ref_tree, PT.get_name(zone))

    assert PT.is_same_tree(PT.get_child_from_name(zone, 'FSol#Vtx'),
                           PT.get_child_from_name(ref_zone, 'FSol#Vtx'), abs_tol=1E-14)
  
  if write_output:
    maia.algo.pe_to_nface(dist_tree, comm)
    out_dir = TU.create_pytest_output_dir(comm)
    maia.io.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'center_to_node.hdf'), comm)
