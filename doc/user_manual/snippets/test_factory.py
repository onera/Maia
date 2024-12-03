def test_generate_dist_points():
  #generate_dist_points@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT

  dist_tree = maia.factory.generate_dist_points(10, 'Unstructured', MPI.COMM_WORLD)
  zone = PT.get_node_from_label(dist_tree, 'Zone_t')
  assert PT.Zone.n_vtx(zone) == 10**3
  #generate_dist_points@end

def test_generate_dist_block():
  #generate_dist_block@start
  from mpi4py import MPI
  comm = MPI.COMM_WORLD

  from maia.factory import generate_dist_block

  # 3D unstructured polyedric zone (basic)
  dist_tree = generate_dist_block(11, 'Poly', comm)
  # 3D structured zone, different number of vertices in each direction
  dist_tree = generate_dist_block((11,21,11), 'Structured', comm)
  # 3D unstructured zone, different length in each direction
  dist_tree = generate_dist_block(11, 'TETRA_4', comm, length=(1.,1.,10.))
  
  # 2D structured zone, PhyDim=3, variable nb. of vertices, custom origin
  dist_tree = generate_dist_block((11,21), 'S', comm, origin=[2.,3.,1.5])
  # 2D unstructured zone, PhyDim=2, custom length direction (not yet implemented)
  #dist_tree = generate_dist_block(6, 'QUAD_4', comm, origin=[0., 0.], length=[[0.707,0.707], [-0.707,0.707]])

  # 1D unstructured zone, PhyDim=3, custom origin and end point
  dist_tree = generate_dist_block(6, 'BAR_2', comm, origin=[0.25, 0.25, 0.], length=[[1,0.,-1.]])
  #generate_dist_block@end

def test_generate_dist_sphere():
  #generate_dist_sphere@start
  from mpi4py import MPI
  import maia
  import maia.pytree as PT

  dist_tree = maia.factory.generate_dist_sphere(10, 'TRI_3', MPI.COMM_WORLD)
  assert PT.Element.CGNSName(PT.get_node_from_label(dist_tree, 'Elements_t')) == 'TRI_3'
  #generate_dist_sphere@end

def test_full_to_dist_tree():
  #full_to_dist_tree@start
  from   mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  comm = MPI.COMM_WORLD

  if comm.Get_rank() == 0:
    tree = maia.io.read_tree(mesh_dir/'S_twoblocks.yaml', comm)
  else:
    tree = None
  dist_tree = maia.factory.full_to_dist_tree(tree, comm, owner=0)
  #full_to_dist_tree@end

def test_dist_to_full_tree():
  #dist_to_full_tree@start
  from   mpi4py import MPI
  import maia
  comm = MPI.COMM_WORLD

  dist_tree = maia.factory.generate_dist_sphere(10, 'TRI_3', comm)
  full_tree = maia.factory.dist_to_full_tree(dist_tree, comm, target=0)

  if comm.Get_rank() == 0:
    assert maia.pytree.get_node_from_name(full_tree, ":CGNS#Distribution") is None
  else:
    assert full_tree is None
  #dist_to_full_tree@end

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

def test_compute_nosplit_weights():
  #compute_nosplit_weights@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  from   maia.factory import partitioning as mpart

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'S_twoblocks.yaml', MPI.COMM_WORLD)

  zone_to_parts = mpart.compute_nosplit_weights(dist_tree, MPI.COMM_WORLD)
  if MPI.COMM_WORLD.Get_size() == 2:
    assert len(zone_to_parts) == 1
  #compute_nosplit_weights@end

def test_compute_regular_weights():
  #compute_regular_weights@start
  from mpi4py import MPI
  import maia
  from   maia.utils.test_utils import mesh_dir
  from   maia.factory import partitioning as mpart

  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'S_twoblocks.yaml', MPI.COMM_WORLD)

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
  dist_tree = maia.io.file_to_dist_tree(mesh_dir/'S_twoblocks.yaml', comm)

  zone_to_parts = mpart.compute_balanced_weights(dist_tree, comm)
  if comm.Get_size() == 2 and comm.Get_rank() == 0:
    assert zone_to_parts == {'Base/Large': [0.375], 'Base/Small': [1.0]}
  if comm.Get_size() == 2 and comm.Get_rank() == 1:
    assert zone_to_parts == {'Base/Large': [0.625]}
  #compute_balanced_weights@end

def test_recover_dist_tree():
  #recover_dist_tree@start
  from mpi4py import MPI
  import maia
  comm = MPI.COMM_WORLD

  dist_tree_bck  = maia.factory.generate_dist_block(5, 'TETRA_4', comm)
  part_tree = maia.factory.partition_dist_tree(dist_tree_bck, comm)

  dist_tree = maia.factory.recover_dist_tree(part_tree, comm)
  assert maia.pytree.is_same_tree(dist_tree, dist_tree_bck)
  #recover_dist_tree@end
