import pytest
import pytest_parallel
import os
from mpi4py import MPI

import maia
from maia.utils   import test_utils     as TU

from maia.factory import partitioning   as PPA

PART_TOOLS = []
if maia.pdm_has_parmetis:
  PART_TOOLS.append("parmetis")
if maia.pdm_has_ptscotch:
  PART_TOOLS.append("ptscotch")

@pytest_parallel.mark.parallel([3])
def test_load_balancing(comm):
  """
  An important input of partioning method is the dictionnary
  zone_to_parts. This dictionnary allow to control the number of partitions
  produced on each rank, as well as their size (if parmetis is used)

  The dictionnary maps the path of the initial block to a list : the size
  of the list is the number of partitions to create for this rank, and
  the values are the desired weight (in % of cell) of each partition. Empty
  list means this rank will hold no partition for this block.
  The sum of weights on all ranks for a given block should be 1.

  In this example, we have 3 procs and two blocks
  """

  # > This mesh has two blocks of different sizes (768 & 192 cells)
  mesh_file = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)
  zone_paths = maia.pytree.predicates_to_paths(dist_tree, 'CGNSBase_t/Zone_t')

  """ zone_to_parts dict can be specified by hand this way : """
  zone_to_parts = {}
  if comm.Get_rank() == 0:
    zone_to_parts['Base/Small'] = []
    zone_to_parts['Base/Large'] = [.25]
  if comm.Get_rank() == 1:
    zone_to_parts['Base/Small'] = [1.] #Small block will not be cut
    zone_to_parts['Base/Large'] = [.25]
  if comm.Get_rank() == 2:
    zone_to_parts['Base/Small'] = []
    zone_to_parts['Base/Large'] = [.1, .1, .2, .1] #Large block will be cut in several parts !
  assert comm.allreduce(sum(zone_to_parts['Base/Small']), MPI.SUM) == 1. and \
         comm.allreduce(sum(zone_to_parts['Base/Large']), MPI.SUM) == 1.

  """ Maia provides some helpers to automatically compute zone_to_parts dict :

  the n_part_per_zone function makes each rank request n (usually one) partitions
  on each block. Weights will thus be homogenous : """
  zone_to_parts = PPA.compute_regular_weights(dist_tree, comm, 1)

  for zone_path in zone_paths:
    assert len(zone_to_parts[zone_path]) == 1
    assert zone_to_parts[zone_path][0] == 1. / comm.Get_size()

  """ Note that n_part can be different for each proc :"""
  n_part = 3 if comm.Get_rank == 1 else 1
  zone_to_parts = PPA.compute_regular_weights(dist_tree, comm, n_part)

  for zone_path in zone_paths:
    assert len(zone_to_parts[zone_path]) == n_part
    assert zone_to_parts[zone_path][0] == 1. / comm.allreduce(n_part, MPI.SUM)

  """ the other function balance_multizone_tree try to balance the total number of
  partitioned cells for each rank while minimizing the number of partitions : """

  zone_to_parts = PPA.compute_balanced_weights(dist_tree, comm)

  # """ We can see that the small zone will not be cut """
  assert comm.allreduce(len(zone_to_parts.get('Base/Small', [])), MPI.SUM) == 1 and \
         comm.allreduce(len(zone_to_parts.get('Base/Large', [])), MPI.SUM) == 3
  assert comm.allreduce(sum(zone_to_parts.get('Base/Small', [])), MPI.SUM) == 1. and \
         comm.allreduce(sum(zone_to_parts.get('Base/Large', [])), MPI.SUM) == 1.

@pytest_parallel.mark.parallel([3])
def test_part_H(comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'H_elt_and_s.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  part_tree = PPA.partition_dist_tree(dist_tree, comm)

  assert comm.allreduce(len(maia.pytree.get_all_Zone_t(part_tree)), MPI.SUM) == 4
  n_original_jn = len(maia.pytree.get_nodes_from_name(part_tree, '1to1Connection:dom*'))
  assert comm.allreduce(n_original_jn, MPI.SUM) == 4

  if write_output:
    out_dir = TU.create_pytest_output_dir(comm)
    maia.io.part_tree_to_file(part_tree, os.path.join(out_dir, 'part_tree.hdf'), comm)

@pytest_parallel.mark.parallel([3])
def test_part_S(comm, write_output):
  mesh_file = os.path.join(TU.mesh_dir, 'S_twoblocks.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  zone_to_parts = PPA.compute_regular_weights(dist_tree, comm, 2)
  part_tree = PPA.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts)
  assert len(maia.pytree.get_all_Zone_t(part_tree)) == 2*2

  if write_output:
    out_dir = TU.create_pytest_output_dir(comm)
    maia.io.part_tree_to_file(part_tree, os.path.join(out_dir, 'part_tree.hdf'), comm)

@pytest.mark.parametrize("graph_part_tool", PART_TOOLS)
@pytest_parallel.mark.parallel([2])
def test_part_elements(comm, graph_part_tool, write_output):

  mesh_file = os.path.join(TU.mesh_dir, 'Uelt_M6Wing.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  #Note : zone_to_parts defaults to 1part_per_zone
  part_tree = PPA.partition_dist_tree(dist_tree, comm, graph_part_tool=graph_part_tool)
  assert len(maia.pytree.get_all_Zone_t(part_tree)) == 1

  if write_output:
    out_dir = TU.create_pytest_output_dir(comm)
    maia.io.part_tree_to_file(part_tree, os.path.join(out_dir, 'part_tree.hdf'), comm)


renum_methods_to_test = ["NONE", "HILBERT"]
if maia.pdma_enabled:
  renum_methods_to_test.append("CACHEBLOCKING2")
@pytest.mark.parametrize("cell_renum_method", renum_methods_to_test)
@pytest_parallel.mark.parallel([2])
def test_part_NGon(comm, cell_renum_method, write_output):

  #Todo : replace with naca3 zones
  mesh_file = os.path.join(TU.mesh_dir, 'U_ATB_45.yaml')
  dist_tree = maia.io.file_to_dist_tree(mesh_file, comm)

  zone_to_parts = PPA.compute_balanced_weights(dist_tree, comm)

  #Different reordering methods can be applied after partitioning
  reordering = {'cell_renum_method' : cell_renum_method,
                'face_renum_method' : 'NONE',
                'n_cell_per_cache'  : 0,
                'n_face_per_pack'   : 0,
                'graph_part_tool'   : 'parmetis' }

  part_tree = PPA.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts, reordering=reordering)

  assert len(maia.pytree.get_all_Zone_t(part_tree)) == sum([len(zone_to_parts[zone]) for zone in zone_to_parts])

  if write_output:
    out_dir = TU.create_pytest_output_dir(comm)
    maia.io.part_tree_to_file(part_tree, os.path.join(out_dir, 'part_tree.hdf'), comm)

