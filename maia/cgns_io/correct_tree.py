from   mpi4py             import MPI
import Converter.PyTree   as     C
import Converter.Internal as     I

def correct_point_range(size_tree, size_data):
  """
  Performs adaptation or correction in order to be properly setup for other algorithm
  """

  print("correct_point_range")
