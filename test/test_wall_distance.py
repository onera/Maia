import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
import numpy as np

import Converter.Internal as I
import Converter.PyTree   as C

import maia
from maia.cgns_io                     import cgns_io_tree            as IOT
from maia.partitioning                import part                    as PPA
from maia.partitioning.load_balancing import setup_partition_weights as DBA
from maia.sids  import pytree     as PT
from maia.utils import test_utils as TU

import maia.tree_exchange as TE
from maia.tree_exchange.part_to_dist import data_exchange as PTD

from maia.geometry.wall_distance import wall_distance

ref_dir = os.path.join(os.path.dirname(__file__), 'references')

@pytest.mark.parametrize("method", ["cloud"])
@mark_mpi_test([1,4])
def test_wall_distance_S(method, sub_comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  ref_file  = os.path.join(ref_dir,     'S_twoblocks_walldist.yaml')

  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  # Families are not present in the tree, we need to add it
  base = I.getBases(dist_tree)[0]
  fam = I.newFamily('WALL', parent=base)
  I.newFamilyBC(parent=fam)
  for bc in I.getNodesFromType(dist_tree, 'BC_t'):
    if I.getValue(bc) == 'BCWall':
      I.setValue(bc, 'FamilySpecified')
      I.createChild(bc, 'FamilyName', 'FamilyName_t', 'WALL')

  # Partitioning
  part_tree = PPA.partitioning(dist_tree, sub_comm, graph_part_tool='ptscotch')

  # Wall distance computation
  families = ['WALL']
  wall_distance(part_tree, sub_comm, method=method, families=families)

  # Save file and compare
  for d_base in I.getBases(dist_tree):
    for d_zone in I.getZones(d_base):
      p_zones = TE.utils.get_partitioned_zones(part_tree, I.getName(d_base) + '/' + I.getName(d_zone))
      PTD.part_sol_to_dist_sol(d_zone, p_zones, sub_comm)

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    C.convertPyTree2File(part_tree, os.path.join(out_dir, f'parttree_out_{sub_comm.Get_rank()}.hdf'))
    IOT.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'result.hdf'), sub_comm)

  # Compare to reference solution
  refence_solution = IOT.file_to_dist_tree(ref_file, sub_comm)
  for d_base in I.getBases(dist_tree):
    for d_zone in I.getZones(d_base):
      zone_path = '/'.join([I.getName(d_base), I.getName(d_zone)])
      ref_wall_dist = I.getNodeFromPath(refence_solution, zone_path + '/WallDistance')
      assert PT.is_same_tree(ref_wall_dist, I.getNodeFromName1(d_zone, 'WallDistance'), abs_tol=1E-14)

wall_dist_methods = ["cloud"]
if maia.pdma_enabled:
  wall_dist_methods.append("propagation")
@pytest.mark.parametrize("method", wall_dist_methods)
@mark_mpi_test([1, 3])
def test_wall_distance_U(method, sub_comm, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'U_ATB_45.yaml')
  ref_file  = os.path.join(ref_dir,     'U_ATB_45_walldist.yaml')

  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  #Let WALL family be autodetected by setting its type to wall:
  wall_family = I.getNodeFromName2(dist_tree, 'WALL')
  family_bc = I.getNodeFromName1(wall_family, 'FamilyBC')
  I.setValue(family_bc, 'BCWall')

  # Partitioning
  zone_to_parts = DBA.npart_per_zone(dist_tree, sub_comm, 3)
  part_tree = PPA.partitioning(dist_tree, sub_comm, zone_to_parts=zone_to_parts)

  # Wall distance computation
  wall_distance(part_tree, sub_comm, method=method)

  # Save file and compare
  for d_base in I.getBases(dist_tree):
    for d_zone in I.getZones(d_base):
      p_zones = TE.utils.get_partitioned_zones(part_tree, I.getName(d_base) + '/' + I.getName(d_zone))
      PTD.part_sol_to_dist_sol(d_zone, p_zones, sub_comm)

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    IOT.dist_tree_to_file(dist_tree, os.path.join(out_dir, 'result.hdf'), sub_comm)

  # Compare to reference solution
  refence_solution = IOT.file_to_dist_tree(ref_file, sub_comm)
  for d_base in I.getBases(dist_tree):
    for d_zone in I.getZones(d_base):
      zone_path = '/'.join([I.getName(d_base), I.getName(d_zone)])
      ref_wall_dist = I.getNodeFromPath(refence_solution, zone_path + '/WallDistance')
      assert PT.is_same_tree(ref_wall_dist, I.getNodeFromName1(d_zone, 'WallDistance'), abs_tol=1E-14)

