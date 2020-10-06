cimport cython
cimport numpy       as      NPY

# MANDATORY :
# ---------
NPY.import_array()


# --------------------------------------------------------------------------
def compute_proc_indexes(NPY.ndarray[NPY.int32_t, ndim=1, mode='fortran'] proc_indexes,
                         int n_element, int i_rank, int n_rank):
  """
  """
  # ************************************************************************
  # > Declaration
  cdef int step      = n_element//(n_rank)
  cdef int remainder = n_element%(n_rank)
  # ************************************************************************

  if (i_rank < remainder):
    proc_indexes[0] = i_rank * (step + 1)
    proc_indexes[1] = proc_indexes[0] + step + 1
  else:
    proc_indexes[0] = i_rank * step + remainder
    proc_indexes[1] = proc_indexes[0] + step
