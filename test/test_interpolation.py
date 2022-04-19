import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import numpy as np
import os


import Converter.PyTree       as C
import Converter.Internal     as I
import maia.sids.Internal_ext as IE
from maia.sids import sids

from maia.utils                       import test_utils              as TU
from maia.utils                       import parse_yaml_cgns
from maia.cgns_io                     import cgns_io_tree            as IOT
from maia.generate                    import dcube_generator         as DCG
from maia.distribution                import distribute_nodes        as DN
from maia.partitioning.load_balancing import setup_partition_weights as DBA
from maia.partitioning                import part                    as PPA
from maia.tree_exchange.dist_to_part  import data_exchange           as MBTP
from maia.geometry.geometry           import compute_cell_center

from maia.interpolation               import interpolate             as ITP

def refine_mesh(tree, factor=1):
  """ On the fly (sequential) isotropic mesh refinement """
  import Intersector.PyTree as XOR
  tree = I.fixNGon(tree)
  zones = I.getZones(tree) #Go old norm
  assert len(zones) == 1
  refined_tree = XOR.adaptCells(tree, factor*np.ones(sids.Zone.n_vtx(zones[0]), dtype=int), sensor_type=2)
  refined_tree = XOR.closeCells(refined_tree)
  I._createElsaHybrid(refined_tree, method=1)
  I._rmNodesByName(refined_tree, ':elsA#Hybrid')

  I._rmNodesByName(refined_tree, 'NFaceElements') #Convert back to new norm
  ngon = I.getNodeFromName(refined_tree, 'NGonElements')
  er_n = I.getNodeFromName1(ngon, 'ElementRange')
  ec_n = I.getNodeFromName1(ngon, 'ElementConnectivity')
  pe_n = I.getNodeFromName1(ngon, 'ParentElements')
  pe_n[1] += er_n[1][1] * (pe_n[1] > 0) #Cassiopee uses old indexing
  ec  = ec_n[1]
  eso = np.empty(sids.ElementSize(ngon)+1, np.int32)
  eso[0] = 0
  c = 0
  for i in range(sids.ElementSize(ngon)):
    eso[i+1] = eso[i] + ec[c]
    c += ec[c] + 1
  ec_n[1] = np.delete(ec, np.arange(sids.ElementSize(ngon)) + eso[:-1])
  I.newDataArray('ElementStartOffset', eso, parent=ngon)
  return refined_tree

@mark_mpi_test([1])
@pytest.mark.parametrize("strategy", ["LocationAndClosest", "Location"])
def test_interpolation_non_overlaping_cubes(sub_comm, strategy, write_output):
  n_vtx_src       = 11
  origin_src      = [0., 0., 0.]
  n_vtx_tgt       = 11
  origin_tgt      = [0.84, 0.51, 0.02] #Chose wisely to avoid ties in mesh location

  # Generate meshes
  dist_tree_src    = DCG.dcube_generate(n_vtx_src, 1., origin_src, sub_comm)
  dist_tree_target = DCG.dcube_generate(n_vtx_tgt, 1., origin_tgt, sub_comm)

  # Remove some useless nodes
  I._rmNodesByName(dist_tree_src, 'ZoneBC')
  I._rmNodesByName(dist_tree_target, 'ZoneBC')

  # Create a field on the source mesh : we use gnum to have something independant of parallelism
  zone = I.getZones(dist_tree_src)[0]
  d_fs = I.newFlowSolution("FlowSolution#Init", gridLocation='CellCenter', parent=zone)
  distri = IE.getDistribution(zone, 'Cell')[1]
  I.newDataArray("Density", np.arange(distri[0], distri[1], dtype=float)+1, parent=d_fs)

  # Create partition on the meshes. Source and destination can have a different partitionning !
  dzone_to_weighted_parts_src    = DBA.npart_per_zone(dist_tree_src   , sub_comm, 2)
  dzone_to_weighted_parts_target = DBA.npart_per_zone(dist_tree_target, sub_comm, 1)
  part_tree_src    = PPA.partitioning(dist_tree_src   , sub_comm, zone_to_parts=dzone_to_weighted_parts_src   )
  part_tree_target = PPA.partitioning(dist_tree_target, sub_comm, zone_to_parts=dzone_to_weighted_parts_target)

  # Transfert source flow sol on partitions
  MBTP.dist_sol_to_part_sol(zone, I.getZones(part_tree_src), sub_comm)

  # Interpolation
  # With Location strategy, non located point will have a NaN sol. With LocationAndClosest,
  # a ClosestPoint algorithm is applied to the non located points
  ITP.interpolate_from_part_trees(part_tree_src, part_tree_target, sub_comm,\
      containers_name=['FlowSolution#Init'], location='CellCenter', strategy=strategy) 

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    C.convertPyTree2File(part_tree_src   , os.path.join(out_dir, 'part_tree_src.hdf'))
    C.convertPyTree2File(part_tree_target, os.path.join(out_dir, 'part_tree_target.hdf'))

  # > Check results
  for tgt_part in I.getZones(part_tree_target):
    sol_n = I.getNodeFromPath(tgt_part, "FlowSolution#Init/Density")
    assert sol_n is not None
    sol = I.getValue(sol_n)
    # Expected sol can be recomputed using cell centers
    expected_sol = np.empty_like(sol)
    cell_center  = compute_cell_center(tgt_part)
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
  mesh_file = os.path.join(TU.mesh_dir, 'U_ATB_45.yaml')

  # Load mesh and create a refined version with proc 0
  if sub_comm.Get_rank() == 0:
    with open(mesh_file, 'r') as f:
      tree = parse_yaml_cgns.to_cgns_tree(f)
    # Simplify tree
    I._rmNodesByType(tree, 'ZoneBC_t')
    I._rmNodesByType(tree, 'ZoneGridConnectivity_t')
    zone = I.getZones(tree)[0]
    d_fs = I.newFlowSolution("FlowSolution#Centers", gridLocation='CellCenter', parent=zone)
    I.newDataArray("Density", np.arange(sids.Zone.n_cell(zone), dtype=float)+1, parent=d_fs)
    refined_tree = refine_mesh(tree)
  else:
    tree, refined_tree = None, None
  dist_tree_src = DN.distribute_tree(tree, sub_comm, owner=0)
  dist_tree_tgt = DN.distribute_tree(refined_tree, sub_comm, owner=0)

  zone = I.getZones(dist_tree_src)[0]
  sol = I.copyTree(I.getNodeFromName(zone, 'FlowSolution#Centers'))
  I.setName(sol, 'FlowSolution#Init')
  I._addChild(zone, sol)

  # Create partition on the meshes
  dzone_to_weighted_parts_target = DBA.npart_per_zone(dist_tree_tgt, sub_comm, n_part_tgt)
  part_tree_src = PPA.partitioning(dist_tree_src, sub_comm)
  part_tree_tgt = PPA.partitioning(dist_tree_tgt, sub_comm, zone_to_parts=dzone_to_weighted_parts_target)

  # Transfert source flow sol on partitions
  MBTP.dist_sol_to_part_sol(I.getZones(dist_tree_src)[0], I.getZones(part_tree_src), sub_comm)
  MBTP.dist_sol_to_part_sol(I.getZones(dist_tree_tgt)[0], I.getZones(part_tree_tgt), sub_comm)

  ITP.interpolate_from_part_trees(part_tree_src, part_tree_tgt, sub_comm,\
      containers_name=['FlowSolution#Init'], location='CellCenter')

  # > Check results
  for tgt_part in I.getZones(part_tree_tgt):
    topo = I.getNodeFromPath(tgt_part, "FlowSolution#Centers/Density")
    geo  = I.getNodeFromPath(tgt_part, "FlowSolution#Init/Density")
    assert np.array_equal(topo[1], geo[1])

