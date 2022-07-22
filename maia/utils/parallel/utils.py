from mpi4py import MPI
import numpy as np
import Converter.Internal as I

from maia.utils import py_utils
from maia       import npy_pdm_gnum_dtype

def uniform_distribution(n_elt, comm):
  """
  """
  u_dist = py_utils.uniform_distribution_at(n_elt, comm.Get_rank(), comm.Get_size())
  proc_indices = np.empty(3, dtype=npy_pdm_gnum_dtype)
  proc_indices[0] = u_dist[0]
  proc_indices[1] = u_dist[1]
  proc_indices[2] = n_elt
  return proc_indices

def dn_to_distribution(dn_elt, comm):
  """
  """
  distri_full = gather_and_shift(dn_elt, comm, npy_pdm_gnum_dtype)
  distri      = full_to_partial_distribution(distri_full, comm)
  return distri

def partial_to_full_distribution(partial_distrib, comm):
  """
  Compute the full distribution array from the partials distribution
  arrays. 
  Full distribution store data for all procs, ie is Np+1 sized array
  [0, dn_0, dn_0+dn_1, ..., sum_{j=0,i-1}dn_j, sum_{j=0...Np-1}dn_j := total_size]
  Partial distribution is reduced for each proc i to the 3 values array
  [start_i, end_i, total_size] = [sum_{j=0,i-1}dn_j, sum_{j=0,i}dn_j, total_size]
  Input and output must be numpy arrays
  """
  dn_elmt = partial_distrib[1] - partial_distrib[0]
  full_distrib = np.empty((comm.Get_size() + 1), dtype=partial_distrib.dtype)
  full_distrib_view = full_distrib[1:]
  #Fill full_distri[1:], then full_distrib[0] with 0
  comm.Allgather(dn_elmt, full_distrib_view)
  full_distrib[0] = 0
  #Compute cumulated sum
  np.cumsum(full_distrib, out=full_distrib)
  return full_distrib

def full_to_partial_distribution(full_distrib, comm):
  return full_distrib[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

def gather_and_shift(value, comm, dtype=None):
  if dtype is None:
    value = np.asarray(value)
    dtype = value.dtype
  else:
    value = np.asarray(value, dtype=dtype)
  distrib = np.empty(comm.Get_size()+1, dtype)
  distrib_view = distrib[1:]
  comm.Allgather(value, distrib_view)
  distrib[0]   = 0
  np.cumsum(distrib, out=distrib)
  return distrib

def exists_anywhere(trees, node_path, comm):
  exists_loc = False
  for tree in trees:
    exists_loc = exists_loc or (I.getNodeFromPath(tree, node_path) is not None)
  return comm.allreduce(exists_loc, op=MPI.LOR)

def exists_everywhere(trees, node_path, comm):
  exists_loc = True #Allow True if list is empty
  for tree in trees:
    exists_loc = exists_loc and (I.getNodeFromPath(tree, node_path) is not None)
  return comm.allreduce(exists_loc, op=MPI.LAND)

