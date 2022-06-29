def test_generate_dist_block():
  #generate_dist_block@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT

  dist_tree = maia.factory.generate_dist_block(10, 'Poly', MPI.COMM_WORLD)
  zone = PT.getNodeFromType(dist_tree, 'Zone_t')
  assert PT.Element.CGNSName(PT.getNodeFromType(zone, 'Elements_t')) == 'NGON_n'

  dist_tree = maia.factory.generate_dist_block(10, 'TETRA_4', MPI.COMM_WORLD)
  zone = PT.getNodeFromType(dist_tree, 'Zone_t')
  assert PT.Element.CGNSName(PT.getNodeFromType(zone, 'Elements_t')) == 'TETRA_4'
  #generate_dist_block@end

def test_partition_dist_tree():
  #partition_dist_tree@start
  from mpi4py import MPI
  import maia

  comm = MPI.COMM_WORLD
  i_rank, n_rank = comm.Get_rank(), comm.Get_size()
  dist_tree  = maia.factory.generate_dist_block(10, 'Poly', comm)

  #Basic use
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm)

  #Crazy partitioning where each proc get as many partitions as its rank
  n_part_tot = n_rank * (n_rank + 1) // 2
  part_tree = maia.factory.partition_dist_tree(dist_tree, comm, \
      zone_to_parts={'Base/zone' : [1./n_part_tot for i in range(i_rank+1)]})
  assert len(maia.pytree.get_all_Zone_t(part_tree)) == i_rank+1
  #partition_dist_tree@end

def test_compute_regular_weights():
  #compute_regular_weights@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  from   maia.factory import partitioning as mpart

  dist_tree = maia.io.file_to_dist_tree(mesh_dir+'/S_twoblocks.yaml', MPI.COMM_WORLD)

  zone_to_parts = mpart.compute_regular_weights(dist_tree, MPI.COMM_WORLD)
  if MPI.COMM_WORLD.Get_size() == 2:
    assert zone_to_parts == {'Base/Large': [0.5], 'Base/Small': [0.5]}
  #compute_regular_weights@end

def test_compute_balanced_weights():
  #compute_balanced_weights@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  from   maia.factory import partitioning as mpart

  comm = MPI.COMM_WORLD
  dist_tree = maia.io.file_to_dist_tree(mesh_dir+'/S_twoblocks.yaml', comm)

  zone_to_parts = mpart.compute_balanced_weights(dist_tree, comm)
  if comm.Get_size() == 2 and comm.Get_rank() == 0:
    assert zone_to_parts == {'Base/Large': [0.375], 'Base/Small': [1.0]}
  if comm.Get_size() == 2 and comm.Get_rank() == 1:
    assert zone_to_parts == {'Base/Large': [0.625]}
  #compute_balanced_weights@end
