import numpy              as np
import Converter.Internal as I
import maia.sids.Internal_ext as IE

from maia import npy_pdm_gnum_dtype

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
  i_rank = comm.Get_rank()
  n_rank = comm.Get_size()
  u_dist = uniform_distribution_at(n_elt,i_rank,n_rank)
  proc_indices = np.empty(3, dtype=npy_pdm_gnum_dtype)
  proc_indices[0] = u_dist[0]
  proc_indices[1] = u_dist[1]
  proc_indices[2] = n_elt
  return proc_indices

def create_distribution_node(n_elt, comm, name, parent_node):
  """
  setup CGNS node with distribution
  """
  distrib    = uniform_distribution(n_elt, comm)
  create_distribution_node_from_distrib(name, parent_node, distrib)

def create_distribution_node_from_distrib(name, parent_node, distrib):
  """
  setup CGNS node with distribution
  """
  distrib_ud = IE.newDistribution(parent=parent_node)
  I.newDataArray(name, value=distrib, parent=distrib_ud)
