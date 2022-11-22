import numpy as np
from mpi4py import MPI


def equal_array_comparison(comm = MPI.COMM_SELF):
  # TODO parallel reduce
  def impl(x, ref):
    if np.array_equal(x, ref):
      return ''
    else:
      return str(x) + ' <> ' + str(ref)
  return impl


def sq_norm(x):
  return np.inner(x,x)

def norm(x, comm):
  #res = np.array([sq_norm(x)])
  #comm.Allreduce(MPI.IN_PLACE, res, MPI.SUM)
  #return np.sqrt(res[0])
  res = comm.allreduce(sq_norm(x), MPI.SUM)
  return np.sqrt(res)

def _close_in_relative_norm(x, ref, tol, comm):
  x   = np.array(x)
  ref = np.array(ref)
  diff = np.abs(x-ref)

  norm_diff = norm(diff, comm)
  norm_ref  = norm(ref , comm)

  # floating point value closest to 0 before loosing precision (see 'denormal numbers')
  smallest_normal = np.finfo(np.float64).smallest_normal

  if norm_ref >= smallest_normal:
    denorm = False
    is_ok = (norm_diff/norm_ref) <= tol
  else:
    denorm = False
    is_ok = np.array_equal(x, ref) # when the reference itself is extremely small, require the values to be exactly equal
                                  # Notes:
                                  #   - more strict than `norm_diff == 0` (rounding effects)
                                  #   - not the same as `np.array_equal(diff, 0.)` (numpy bug)
                                  #   - numpy correctly compares 0. and -0. as equal

  return is_ok, denorm, norm_diff, norm_ref

def close_in_relative_norm(x, ref, tol, comm):
  return _close_in_relative_norm(x, ref, tol, comm)[0]

def relative_norm_comparison(tol, comm):
  def impl(x, ref):
    is_ok, denorm, norm_diff, norm_ref = _close_in_relative_norm(x, ref, tol, comm)

    if is_ok:
      return ''
    else:
      N = len(ref)
      msg = f'mean diff: {norm_diff/N:.3e}, mean: {norm_ref/N:.3e}, rel error: {norm_diff/norm_ref:.3e}'
      if denorm:
        msg += ' -- WARNING: imprecise comparison because of small reference'
      return msg
  return impl


def field_comparison(tol, comm):
  def impl(x, ref):
    if x is not None and not isinstance(x,str) and x.dtype.kind == 'f':
      return relative_norm_comparison(tol, comm)(x, ref)
    else:
      return equal_array_comparison(comm)(x, ref)
  return impl
