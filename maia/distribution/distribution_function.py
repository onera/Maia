import numpy              as np
import Converter.Internal as I

def uniform_distribution_at(n_elt, i, n_interval):
  """
  """
  step      = n_elt // n_interval
  remainder = n_elt %  n_interval

  if i < remainder:
    inf = i * (step + 1)
    sup = inf + step + 1
  else:
    inf = i * step + remainder
    sup = inf + step

  return inf,sup

def uniform_distribution(n_elt, comm):
  """
  """
  int_type = type(n_elt)
  i_rank = int_type(comm.Get_rank())
  n_rank = int_type(comm.Get_size())
  u_dist = uniform_distribution_at(n_elt,i_rank,n_rank)
  proc_indices = np.empty(3, dtype=type(n_elt))
  proc_indices[0] = u_dist[0]
  proc_indices[1] = u_dist[1]
  proc_indices[2] = n_elt
  return proc_indices

def create_distribution_node(n_elt, comm, name, parent_node):
  """
  setup CGNS node with distribution
  """
  distrib = uniform_distribution(n_elt, comm)
  I.newDataArray(name, value=distrib, parent=parent_node)
