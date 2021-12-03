import pytest
from   pytest_mpi_check._decorator import mark_mpi_test
import os
from mpi4py import MPI

import Converter.Internal as I

import maia
from   maia.utils            import test_utils as TU

from maia.cgns_io            import cgns_io_tree                     as IOT
from maia.cgns_io            import save_part_tree                   as SPT
from maia.partitioning       import part                             as PPA
from maia.partitioning.load_balancing import setup_partition_weights as SPW


@mark_mpi_test([3])
def test_load_balancing(sub_comm):
  """
  An important input of partioning method is the dictionnary
  zone_to_parts. This dictionnary allow to control the number of partitions
  produced on each rank, as well as their size (if parmetis is used)

  The dictionnary map the name of the initial block to a list : the size
  of the list is the number of partitions to create for this rank, and
  the value are the wanted weight (in % of cell) of each partition. Empty
  list means this rank will hold no partition for this block.
  The sum of weights for a given block should be 1.

  In this example, we have 3 procs and two blocks
  """

  # > This mesh has two blocks of different sizes (768 & 192 cells)
  mesh_file = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  """ zone_to_parts dict can be specified by hand eg in this way : """
  zone_to_parts = {}
  if sub_comm.Get_rank() == 0:
    zone_to_parts['Small'] = []
    zone_to_parts['Large'] = [.25]
  if sub_comm.Get_rank() == 1:
    zone_to_parts['Small'] = [1.] #Small block will not be cutted
    zone_to_parts['Large'] = [.25]
  if sub_comm.Get_rank() == 2:
    zone_to_parts['Small'] = []
    zone_to_parts['Large'] = [.1, .1, .2, .1] #Large block will be cutted in several parts !
  assert sub_comm.allreduce(sum(zone_to_parts['Small']), MPI.SUM) == 1. and \
         sub_comm.allreduce(sum(zone_to_parts['Large']), MPI.SUM) == 1.

  """ Maia provide some helpers to automatically compute zone_to_parts dict :

  n_part_per_zone function make each rank to request n (usually one) partition
  on each block. Weights will thus be homogeous : """
  zone_to_parts = SPW.npart_per_zone(dist_tree, sub_comm, 1)

  for zone in I.getZones(dist_tree):
    assert len(zone_to_parts[I.getName(zone)]) == 1
    assert zone_to_parts[I.getName(zone)][0] == 1. / sub_comm.Get_size()

  """ Note that n_part can be different for each proc :"""
  n_part = 3 if sub_comm.Get_rank == 1 else 1
  zone_to_parts = SPW.npart_per_zone(dist_tree, sub_comm, n_part)

  for zone in I.getZones(dist_tree):
    assert len(zone_to_parts[I.getName(zone)]) == n_part
    assert zone_to_parts[I.getName(zone)][0] == 1. / sub_comm.allreduce(n_part, MPI.SUM)

  """ the other function balance_multizone_tree try to equilibrate the total number of
  partitioned cells for each rank while minimizing the number of partitions : """

  zone_to_parts = SPW.balance_multizone_tree(dist_tree, sub_comm)

  """ We can see that small zone will not be cutted """
  assert sub_comm.allreduce(len(zone_to_parts['Small']), MPI.SUM) == 1 and \
         sub_comm.allreduce(len(zone_to_parts['Large']), MPI.SUM) == 3
  assert sub_comm.allreduce(sum(zone_to_parts['Small']), MPI.SUM) == 1. and \
         sub_comm.allreduce(sum(zone_to_parts['Large']), MPI.SUM) == 1.

@mark_mpi_test([3])
def test_part_S(sub_comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  zone_to_parts = SPW.npart_per_zone(dist_tree, sub_comm, 2)
  part_tree = PPA.partitioning(dist_tree, sub_comm, zone_to_parts=zone_to_parts)
  assert len(I.getZones(part_tree)) == 2*2

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    SPT.save_part_tree(part_tree, os.path.join(out_dir, 'part_tree'), sub_comm)

@pytest.mark.parametrize("graph_part_tool", ["parmetis", "ptscotch"])
@mark_mpi_test([2])
def test_part_elements(sub_comm, graph_part_tool, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  #Note : zone_to_parts defaults to 1part_per_zone
  part_tree = PPA.partitioning(dist_tree, sub_comm, graph_part_tool=graph_part_tool)
  assert len(I.getZones(part_tree)) == 1

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    SPT.save_part_tree(part_tree, os.path.join(out_dir, 'part_tree'), sub_comm)


renum_methods_to_test = ["NONE", "HILBERT"]
if maia.pdma_enabled:
  renum_methods_to_test.append("CACHEBLOCKING2")
@pytest.mark.parametrize("cell_renum_method", renum_methods_to_test)
@mark_mpi_test([2])
def test_part_NGon(sub_comm, cell_renum_method, write_output):

  #Todo : replace with naca3 zones
  mesh_file = os.path.join(TU.mesh_dir, 'U_ATB_45.yaml')
  dist_tree = IOT.file_to_dist_tree(mesh_file, sub_comm)

  zone_to_parts = SPW.balance_multizone_tree(dist_tree, sub_comm)

  #Different reordering methods can be applied after partitioning
  reordering = {'cell_renum_method' : cell_renum_method,
                'face_renum_method' : 'NONE',
                'n_cell_per_cache'  : 0,
                'n_face_per_pack'   : 0,
                'graph_part_tool'   : 'parmetis' }

  part_tree = PPA.partitioning(dist_tree, sub_comm, zone_to_parts=zone_to_parts, reordering=reordering)

  assert len(I.getZones(part_tree)) == sum([len(zone_to_parts[zone]) for zone in zone_to_parts])

  if write_output:
    out_dir = TU.create_pytest_output_dir(sub_comm)
    SPT.save_part_tree(part_tree, os.path.join(out_dir, 'part_tree'), sub_comm)

