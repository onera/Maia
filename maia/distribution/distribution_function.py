import numpy as NPY

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

