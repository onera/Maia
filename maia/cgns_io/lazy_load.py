from mpi4py import MPI

def warm_up_dist_tree(dist_tree, data_shape):
  """
  """




def load_collective_pruned_tree(filename, comm):
  """
  """
  # Reprendre load pypart
  ts1 = C.convertFile2PyTree(filename,
                             skeletonData=[3, 5],
                             dataShape=data_shape,
                             format='bin_hdf')

  # ps1[path] -> ( ..., shape)

  # warm_up_dist_tree(dist_tree, data_shape)

  # > Noeud user defined : PointList#Shape

