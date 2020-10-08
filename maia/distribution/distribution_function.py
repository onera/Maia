import numpy              as NPY
import Converter.Internal as I

def uniform_distribution(n_elemt, comm):
  """
  """
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()

  step      = n_elemt // n_rank
  remainder = n_elemt %  n_rank

  proc_indices = NPY.empty( 3, dtype=type(n_elemt), order='c')

  if(i_rank < remainder):
    proc_indices[0] = i_rank * (step + 1)
    proc_indices[1] = proc_indices[0] + step + 1
  else:
    proc_indices[0] = i_rank * step + remainder
    proc_indices[1] = proc_indices[0] + step

  proc_indices[2] = n_elemt

  return proc_indices

def create_distribution_node(n_elemt, comm, name, parent_node):
  """
  Helper class to setup pyCGNS node with distribution
  """
  distrib = uniform_distribution(n_elemt, comm)
  I.newDataArray(name, value=distrib, parent=parent_node)
