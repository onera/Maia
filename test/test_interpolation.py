import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import numpy as np
import os

import Converter.PyTree   as C
import Converter.Internal as I
import maia.pytree        as PT

import maia.utils    as MU
import maia.algo     as MA
import maia.factory  as MF
import maia.transfer as MT


def refine_mesh(tree, factor=1):
  """ On the fly (sequential) isotropic mesh refinement """
  import Intersector.PyTree as XOR
  tree = I.copyTree(tree)
  MA.seq.poly_new_to_old(tree)
  zones = I.getZones(tree) #Go old norm
  assert len(zones) == 1
  refined_tree = XOR.adaptCells(tree, factor*np.ones(PT.Zone.n_vtx(zones[0]), dtype=int), sensor_type=2)
  refined_tree = XOR.closeCells(refined_tree)
  I._adaptNFace2PE(refined_tree, remove=False)
  MA.seq.poly_old_to_new(refined_tree)
  return refined_tree

@mark_mpi_test([1])
@pytest.mark.parametrize("strategy", ["LocationAndClosest", "Location"])
def test_interpolation_non_overlaping_cubes(sub_comm, strategy, write_output):
  n_vtx_src       = 11
  origin_src      = [0., 0., 0.]
  n_vtx_tgt       = 11
  origin_tgt      = [0.84, 0.51, 0.02] #Chose wisely to avoid ties in mesh location

  # Generate meshes
  dist_tree_src    = MF.generate_dist_block(n_vtx_src, "Poly", sub_comm, origin_src)
  dist_tree_target = MF.generate_dist_block(n_vtx_tgt, "Poly", sub_comm, origin_tgt)

  # Remove some useless nodes
  PT.rm_nodes_from_label(dist_tree_src, 'ZoneBC_t')
  PT.rm_nodes_from_label(dist_tree_target, 'ZoneBC_t')

  # Create a field on the source mesh : we use gnum to have something independant of parallelism
  zone = I.getZones(dist_tree_src)[0]
  d_fs = I.newFlowSolution("FlowSolution#Init", gridLocation='CellCenter', parent=zone)
  distri = PT.maia.getDistribution(zone, 'Cell')[1]
  I.newDataArray("Density", np.arange(distri[0], distri[1], dtype=float)+1, parent=d_fs)

  # Create partition on the meshes. Source and destination can have a different partitionning !
  dzone_to_weighted_parts_src    = MF.partitioning.compute_regular_weights(dist_tree_src   , sub_comm, 2)
  dzone_to_weighted_parts_target = MF.partitioning.compute_regular_weights(dist_tree_target, sub_comm, 1)
  part_tree_src    = MF.partition_dist_tree(dist_tree_src   , sub_comm, zone_to_parts=dzone_to_weighted_parts_src   )
  part_tree_target = MF.partition_dist_tree(dist_tree_target, sub_comm, zone_to_parts=dzone_to_weighted_parts_target)

  # Transfert source flow sol on partitions
  MT.dist_tree_to_part_tree_all(dist_tree_src, part_tree_src, sub_comm)

  # Interpolation
  # With Location strategy, non located point will have a NaN sol. With LocationAndClosest,
  # a ClosestPoint algorithm is applied to the non located points
  MA.part.interpolate_from_part_trees(part_tree_src, part_tree_target, sub_comm,\
      containers_name=['FlowSolution#Init'], location='CellCenter', strategy=strategy) 

  if write_output:
    out_dir = MU.test_utils.create_pytest_output_dir(sub_comm)
    C.convertPyTree2File(part_tree_src   , os.path.join(out_dir, 'part_tree_src.hdf'))
    C.convertPyTree2File(part_tree_target, os.path.join(out_dir, 'part_tree_target.hdf'))

  # > Check results
  for tgt_part in I.getZones(part_tree_target):
    sol_n = I.getNodeFromPath(tgt_part, "FlowSolution#Init/Density")
    assert sol_n is not None
    sol = I.getValue(sol_n)
    # Expected sol can be recomputed using cell centers
    expected_sol = np.empty_like(sol)
    cell_center  = MA.part.compute_cell_center(tgt_part)
    for icell in range(sol.shape[0]):
      expected_sol[icell] = -int(cell_center[3*icell] < 0.95) + \
          10*min(int(10*cell_center[3*icell+1]+1), 10) + 100*int(10*cell_center[3*icell+2])
    if strategy == 'Location':
      for icell in range(sol.shape[0]):
        if cell_center[3*icell] > 1 or cell_center[3*icell+1] > 1:
          expected_sol[icell] = np.nan

    assert np.array_equal(expected_sol, sol, equal_nan=True)

@mark_mpi_test([2])
@pytest.mark.parametrize("n_part_tgt", [1,3,7])
def test_interpolation_refined(sub_comm, n_part_tgt, write_output):
  mesh_file = os.path.join(MU.test_utils.mesh_dir, 'U_ATB_45.yaml')

  # Load mesh and create a refined version with proc 0
  if sub_comm.Get_rank() == 0:
    with open(mesh_file, 'r') as f:
      tree = PT.yaml.parse_yaml_cgns.to_cgns_tree(f)
    # Simplify tree
    PT.rm_nodes_from_label(tree, 'ZoneBC_t')
    PT.rm_nodes_from_label(tree, 'ZoneGridConnectivity_t')
    zone = I.getZones(tree)[0]
    d_fs = I.newFlowSolution("FlowSolution#Centers", gridLocation='CellCenter', parent=zone)
    I.newDataArray("Density", np.arange(PT.Zone.n_cell(zone), dtype=float)+1, parent=d_fs)
    refined_tree = refine_mesh(tree)
  else:
    tree, refined_tree = None, None
  dist_tree_src = MF.distribute_tree(tree, sub_comm, owner=0)
  dist_tree_tgt = MF.distribute_tree(refined_tree, sub_comm, owner=0)

  zone = I.getZones(dist_tree_src)[0]
  sol = I.copyTree(PT.get_node_from_name(zone, 'FlowSolution#Centers'))
  I.setName(sol, 'FlowSolution#Init')
  I._addChild(zone, sol)

  # Create partition on the meshes
  dzone_to_weighted_parts_target = MF.partitioning.compute_regular_weights(dist_tree_tgt, sub_comm, n_part_tgt)
  part_tree_src = MF.partition_dist_tree(dist_tree_src, sub_comm)
  part_tree_tgt = MF.partition_dist_tree(dist_tree_tgt, sub_comm, zone_to_parts=dzone_to_weighted_parts_target)

  # Transfert source flow sol on partitions
  MT.dist_tree_to_part_tree_all(dist_tree_src, part_tree_src, sub_comm)
  MT.dist_tree_to_part_tree_all(dist_tree_tgt, part_tree_tgt, sub_comm)

  # Here we use the Interpolator API, who could allow us to redo an interpolation later
  interpolator = MA.part.create_interpolator_from_part_trees(part_tree_src, part_tree_tgt,\
      sub_comm, location='CellCenter', strategy='Location')
  interpolator.exchange_fields('FlowSolution#Init')

  # > Check results
  for tgt_part in I.getZones(part_tree_tgt):
    topo = I.getNodeFromPath(tgt_part, "FlowSolution#Centers/Density")
    geo  = I.getNodeFromPath(tgt_part, "FlowSolution#Init/Density")
    assert np.array_equal(topo[1], geo[1])

