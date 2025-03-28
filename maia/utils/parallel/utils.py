from mpi4py import MPI
import numpy as np
from maia.typing import *
import maia.pytree as PT

from maia.utils import py_utils
from maia       import npy_pdm_gnum_dtype

from Pypdm.Pypdm import compute_weighted_distribution

T = TypeVar('T')

def gathering_distribution(i_rank: int, n_elt: int, comm: MPIComm) -> ArrayLike:
  """
  """
  if   comm.Get_rank()  < i_rank: distrib = np.array([0    , 0    , n_elt ], dtype=npy_pdm_gnum_dtype)
  elif comm.Get_rank() == i_rank: distrib = np.array([0    , n_elt, n_elt ], dtype=npy_pdm_gnum_dtype)
  else                          : distrib = np.array([n_elt, n_elt, n_elt ], dtype=npy_pdm_gnum_dtype)
  return distrib

def uniform_distribution(n_elt: int, comm: MPIComm) -> ArrayLike:
  """
  """
  u_dist = py_utils.uniform_distribution_at(n_elt, comm.Get_rank(), comm.Get_size())
  proc_indices = np.empty(3, dtype=npy_pdm_gnum_dtype)
  proc_indices[0] = u_dist[0]
  proc_indices[1] = u_dist[1]
  proc_indices[2] = n_elt
  return proc_indices

def dn_to_distribution(dn_elt: int, comm: MPIComm) -> ArrayLike:
  """
  """
  distri = np.zeros(3, dtype=npy_pdm_gnum_dtype)
  comm.Exscan(np.array([dn_elt], dtype=npy_pdm_gnum_dtype), distri[0:1])
  
  distri[1:] = distri[0] + dn_elt
  comm.Bcast(distri[2:], root=comm.Get_size()-1)
  return distri

def distribution_from_gnum(gnum_list: List[ArrayLike], 
                           comm: MPIComm, 
                           weights: bool = False,
                           full: bool = False) -> ArrayLike:
  """
  Create a distribution including all the provided gnums. 
  If weights=True, the distribution try to put the same number of ids on each rank.
  Otherwise, it is uniform.
  If full=True, a full (size = comm.size+1) distribution is returned. Otherwise, a partial
  distribution (size = 3)
  """
  if weights:
    from maia.utils import as_pdm_gnum # Cyclic import ...
    _gnum_list = [as_pdm_gnum(gn)  for gn in gnum_list]
    _weights   = [np.ones(gn.size) for gn in gnum_list]
    distri_f = compute_weighted_distribution(_gnum_list, _weights, comm)
    return distri_f if full else full_to_partial_distribution(distri_f, comm)

  else:
    global_max = arrays_max(gnum_list, comm)
    distri = uniform_distribution(global_max, comm)
    return partial_to_full_distribution(distri, comm) if full else distri

def partial_to_full_distribution(partial_distrib: ArrayLike, comm: MPIComm) -> ArrayLike:
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

def full_to_partial_distribution(full_distrib: ArrayLike, comm: MPIComm) -> ArrayLike:
  return full_distrib[[comm.Get_rank(), comm.Get_rank()+1, comm.Get_size()]]

def gather_and_shift(value: Union[int, float, ArrayLike],
                     comm: MPIComm, 
                     dtype: Optional[np.dtype] = None) -> ArrayLike:
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

def arrays_max(array_list: List[ArrayLike], comm: MPIComm) -> int:
  if len(array_list) > 0:
    local_max = max([array.max(initial=0) for array in array_list])
  else:
    local_max = 0
  return comm.allreduce(local_max, MPI.MAX)

def any_true(L: List[T], f: Callable[[T], bool], comm: MPIComm) -> bool:
  return comm.allreduce(py_utils.any_true(L, f), op=MPI.LOR)

def all_true(L: List[T], f: Callable[[T], bool], comm: MPIComm) -> bool:
  return comm.allreduce(py_utils.all_true(L, f), op=MPI.LAND)

def exists_anywhere(trees: List[CGNSTree], node_path: str, comm: MPIComm) -> bool:
  return any_true(trees, 
                  lambda t: PT.get_node_from_path(t, node_path) is not None,
                  comm)

def exists_everywhere(trees: List[CGNSTree], node_path: str, comm: MPIComm) -> bool:
  exists_loc = True #Allow True if list is empty
  for tree in trees:
    exists_loc = exists_loc and (PT.get_node_from_path(tree, node_path) is not None)
  return comm.allreduce(exists_loc, op=MPI.LAND)

