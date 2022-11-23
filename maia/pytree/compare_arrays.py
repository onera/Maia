import numpy as np
from mpi4py import MPI


def equal_array_comparison(comm = MPI.COMM_SELF):
  # TODO parallel reduce
  def impl(x, ref):
    if np.array_equal(x, ref):
      return True, '', ''
    else:
      return False, str(x) + ' <> ' + str(ref), ''
  return impl


def sq_norm(x):
  return np.inner(x,x)

def norm(x, comm):
  res = comm.allreduce(sq_norm(x), MPI.SUM)
  return np.sqrt(res)

def _close_in_relative_norm(x, ref, tol, comm):
  x   = np.array(x)
  ref = np.array(ref)

  if np.array_equal(x, ref):
    return {'exact_eq':True, 'within_tol': True}
  else:
    norm_ref  = norm(ref , comm)
    diff = np.abs(x-ref)
    norm_diff = norm(diff, comm)

    # FIXME because we compare scalar fields, not tensors
    sqN = np.sqrt(len(ref))
    if norm_ref/sqN <= 1e-12:
      denorm = False
      within_tol = True

    # floating point value closest to 0 before loosing precision (see 'denormal numbers')
    else:
      smallest_normal = np.finfo(np.float64).smallest_normal
      if norm_ref >= smallest_normal:
        denorm = False
        within_tol = (norm_diff/norm_ref) <= tol

      else:
        denorm = True
        within_tol = False # when the reference itself is extremely small, require the values to be exactly equal
                      # Notes:
                      #   - more strict than `norm_diff == 0` (rounding effects)
                      #   - not the same as `np.array_equal(diff, 0.)` (numpy bug?)
                      #   - numpy correctly compares 0. and -0. as equal

    return {
      'exact_eq': False,
      'within_tol': within_tol,
      'denorm': denorm,
      'norm_diff': norm_diff,
      'norm_ref': norm_ref,
    }

def close_in_relative_norm(x, ref, tol, comm):
  return _close_in_relative_norm(x, ref, tol, comm)['within_tol']

def relative_norm_comparison(tol, comm):
  def impl(x, ref):
    info = _close_in_relative_norm(x, ref, tol, comm)

    err_msg = ''
    warn_msg = ''
    if not info['exact_eq']:
      norm_diff = info['norm_diff']
      norm_ref  = info['norm_ref']
      sqN = np.sqrt(len(ref))
      msg = f'mean diff: {norm_diff/sqN:.3e}, mean: {norm_ref/sqN:.3e}, rel error: {norm_diff/norm_ref:.3e}'
      if info['denorm']:
        msg += ' -- WARNING: imprecise comparison because of small reference'

      if info['within_tol']:
        warn_msg = msg
      else:
        err_msg = msg

    return info['within_tol'], err_msg, warn_msg
  return impl

def field_comparison(tol, comm):
  def impl(x, ref):
    if x is not None and not isinstance(x,str) and x.dtype.kind == 'f':
      return relative_norm_comparison(tol, comm)(x, ref)
    else:
      return equal_array_comparison(comm)(x, ref)
  return impl
