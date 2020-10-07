from   mpi4py             import MPI
import Converter.PyTree   as     C
import Converter.Internal as     I

def warm_up_dist_tree(dist_tree, data_shape):
  """
  """




def load_collective_pruned_tree(filename, comm, skeleton_depth=3, skeleton_n_data=5):
  """
  """
  rank = comm.Get_rank()
  size = comm.Get_size()

  # > In order to avoid filesystem overload only 1 proc read the squeleton
  if(rank == 0):
    # Reprendre load pypart
    data_shape = dict()
    dist_tree = C.convertFile2PyTree(filename,
                                     skeletonData=[skeleton_depth, skeleton_n_data],
                                     dataShape=data_shape,
                                     format='bin_hdf')

    # > Well we warn_up the with data_shape before send
    warm_up_dist_tree(dist_tree, data_shape)

    I.printTree(dist_tree)

  else:
    dist_tree = None

  # ps1[path] -> ( ..., shape)
  # warm_up_dist_tree(dist_tree, data_shape)

  # > Noeud user defined : PointList#Shape

