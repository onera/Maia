def test_file_to_dist_tree_full():
  #file_to_dist_tree_full@start
  from mpi4py import MPI
  import maia

  # Generate a sample tree
  dist_tree = maia.factory.generate_dist_block(10, "Poly", MPI.COMM_WORLD)
  # Write
  maia.io.dist_tree_to_file(dist_tree, "tree.cgns", MPI.COMM_WORLD)
  # Read
  tree = maia.io.file_to_dist_tree("tree.cgns", MPI.COMM_WORLD)
  #file_to_dist_tree_full@end

def test_file_to_dist_tree_filter():
  #file_to_dist_tree_filter@start
  from mpi4py import MPI
  import maia

  # Generate a sample tree
  dist_tree = maia.factory.generate_dist_block(10, "Poly", MPI.COMM_WORLD)

  # Remove the nodes we do not want to write
  maia.pytree.rm_nodes_from_name(dist_tree, 'CoordinateZ') #This is a DataArray
  maia.pytree.rm_nodes_from_name(dist_tree, 'Zm*') #This is some BC nodes
  # Write
  maia.io.dist_tree_to_file(dist_tree, "tree.cgns", MPI.COMM_WORLD)

  # Read
  from maia.io.cgns_io_tree import load_collective_size_tree, fill_size_tree
  dist_tree = load_collective_size_tree("tree.cgns", MPI.COMM_WORLD)
  #For now dist_tree only contains sizes : let's filter it
  maia.pytree.rm_nodes_from_name(dist_tree, 'CoordinateY') #This is a DataArray
  maia.pytree.rm_nodes_from_name(dist_tree, 'Ym*') #This is some BC nodes
  fill_size_tree(dist_tree, "tree.cgns", MPI.COMM_WORLD)
  #file_to_dist_tree_filter@end

def test_save_part_tree():
  #save_part_tree@start
  from mpi4py import MPI
  import maia

  dist_tree = maia.factory.generate_dist_block(10, "Poly", MPI.COMM_WORLD)
  part_tree = maia.factory.partition_dist_tree(dist_tree, MPI.COMM_WORLD)

  maia.io.part_tree_to_file(part_tree, 'part_tree.cgns', MPI.COMM_WORLD)
  #save_part_tree@end

def test_write_tree():
  #write_tree@start
  from mpi4py import MPI
  import maia

  dist_tree = maia.factory.generate_dist_block(10, "Poly", MPI.COMM_WORLD)
  if MPI.COMM_WORLD.Get_rank() == 0:
    maia.io.write_tree(dist_tree, "tree.cgns")
  #write_tree@end

def test_write_trees():
  #write_trees@start
  from mpi4py import MPI
  import maia

  dist_tree = maia.factory.generate_dist_block(10, "Poly", MPI.COMM_WORLD)
  maia.io.write_trees(dist_tree, "tree.cgns", MPI.COMM_WORLD)
  #write_trees@end
