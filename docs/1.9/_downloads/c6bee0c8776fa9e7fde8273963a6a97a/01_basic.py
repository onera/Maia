# Imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
print(f"Hello from rank {comm.Get_rank()} out of size {comm.Get_size()}")

import maia
import maia.pytree as PT

# File reading
tree = maia.io.file_to_dist_tree('naca_2d.cgns', comm)

# Algo module
maia.algo.dist.extrude(tree, (0,0,1), comm)

# Factory module
ptree = maia.factory.partition_dist_tree(tree, comm)

maia.algo.part.compute_wall_distance(ptree, comm)

# Transfer module
maia.transfer.part_tree_to_dist_tree_all(tree, ptree, comm)

print(PT.get_node_from_name(tree, 'WallDistance') is not None)

# Final write and visualization
maia.io.dist_tree_to_file(tree, 'naca_3d.cgns', comm)
