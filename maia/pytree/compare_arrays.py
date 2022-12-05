import numpy as np
from mpi4py import MPI
import maia.pytree as PT


def equal_array_report(x, ref, comm):
  equal_arrays = np.array_equal(x, ref)
  equal_arrays_tot = comm.allreduce(equal_arrays, MPI.LAND)
  if equal_arrays_tot:
    return True, '', ''
  else:
    sz_tot = comm.allreduce(len(x), MPI.SUM)
    if sz_tot < 10:
      xs   = comm.gather(x  , root=0)
      refs = comm.gather(ref, root=0)

      if comm.Get_rank() == 0:
        x_tot   = np.concatenate(xs)
        ref_tot = np.concatenate(refs)

        return False, str(x_tot) + ' <> ' + str(ref_tot), ''
      else:
        return False, '', ''
    else:
      eq = np.equal(x, ref)
      n_eq = np.count_nonzero(eq)
      n_eq_tot = comm.reduce(n_eq, op=MPI.SUM, root=0)
      if comm.Get_rank() == 0:
        n_not_eq = sz_tot - n_eq_tot
        return False, f'{n_not_eq} values are different', ''
      else:
        return False, '', ''

def equal_array_comparison(comm = MPI.COMM_SELF):
  def impl(nodes_stack):
    node_x,node_ref = nodes_stack[-1]
    x   = PT.get_value(node_x)
    ref = PT.get_value(node_ref)
    return equal_array_report(x, ref, comm)
  return impl


def sq_norm(x):
  return np.inner(x,x)

def norm(x, comm):
  res = comm.allreduce(sq_norm(x), MPI.SUM)
  return np.sqrt(res)

def _close_in_relative_norm(x, ref, tol, comm):
  x   = np.array(x)
  ref = np.array(ref)

  equal_arrays = equal_array_report(x, ref, comm)[0]
  if equal_arrays:
    return {'exact_eq':True, 'within_tol': True}
  else:
    norm_ref  = norm(ref , comm)
    diff = np.abs(x-ref)
    norm_diff = norm(diff, comm)

    # FIXME because we compare scalar fields, not tensors
    sqN = np.sqrt(len(ref))
    #if norm_ref/sqN <= 1e-12:
    if False:
      denorm = False
      within_tol = True

    # floating point value closest to 0 before loosing precision (see 'denormal numbers')
    else:
      try:
        smallest_normal = np.finfo(np.float64).smallest_normal
      except AttributeError:
        smallest_normal = np.float64(2.2250738585072014e-308)
        import warnings
        warnings.warn(f'`np.finfo(np.float64).smallest_normal` ' \
                      f'does not exist with your NumPy version. ' \
                      f'using {smallest_normal}', DeprecationWarning)

      if norm_ref >= smallest_normal:
        denorm = False
        within_tol = (norm_diff/norm_ref) <= tol

      else:
        if norm_ref == 0.:
          denorm = False
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

def relative_norm_comparison(tol, comm, n_dim=1):
  def impl(x, ref):
    info = _close_in_relative_norm(x, ref, tol, comm)

    err_msg = ''
    warn_msg = ''
    if not info['exact_eq']:
      norm_diff = info['norm_diff']
      norm_ref  = info['norm_ref']
      sqN = np.sqrt(len(ref)/n_dim)
      with np.errstate(divide='ignore'): # do not warn if `norm_ref == 0.`
        msg = f'RMS mean diff: {norm_diff/sqN:.3e}, RMS ref mean: {norm_ref/sqN:.3e}, rel error: {norm_diff/norm_ref:.3e}'
      if info['denorm']:
        msg += ' -- WARNING: imprecise comparison because of small reference'

      if info['within_tol']:
        warn_msg = msg
      else:
        err_msg = msg

    return info['within_tol'], err_msg, warn_msg
  return impl


def field_comparison(tol, comm):
  def impl(nodes_stack):
    node_x,node_ref = nodes_stack[-1]
    x   = PT.get_value(node_x,raw=True)
    ref = PT.get_value(node_ref,raw=True)
    if x is not None and not isinstance(x,str) and x.dtype.kind == 'f':
      return relative_norm_comparison(tol, comm)(x, ref)
    else:
      return equal_array_comparison(comm)(nodes_stack)
  return impl


def _relative_norm_comparison(tol, comm, tensor_name, suffixes, x, ref):
  x_val   = [PT.get_value(PT.get_node_from_name(x  , tensor_name+suffix)) for suffix in suffixes]
  ref_val = [PT.get_value(PT.get_node_from_name(ref, tensor_name+suffix)) for suffix in suffixes]

  x_cat   = np.concatenate(x_val)
  ref_cat = np.concatenate(ref_val)

  return relative_norm_comparison(tol, comm, n_dim=len(suffixes))(x_cat, ref_cat)

def relative_norm_comparison_rank_1(tol, comm, tensor_name, x, ref):
  suffixes = ['X','Y','Z']
  return _relative_norm_comparison(tol, comm, tensor_name, suffixes, x, ref)

def relative_norm_comparison_rank_2(tol, comm, tensor_name, x, ref):
  suffixes = ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']
  return _relative_norm_comparison(tol, comm, tensor_name, suffixes, x, ref)


def tensor_field_comparison(tol, comm):
  class impl:
    @staticmethod
    def __call__(nodes_stack):
      node_x,node_ref = nodes_stack[-1]
      name_x = PT.get_name(node_x)

      x   = PT.get_value(node_x,raw=True)
      ref = PT.get_value(node_ref,raw=True)
      if x is not None and not isinstance(x,str) and PT.get_label(node_x) == 'DataArray_t' and x.dtype.kind == 'f':
        parent_x,parent_ref = nodes_stack[-2]
        if name_x[-2:-1] == 'XX':
          tensor_name = name_x[:-2]
          return relative_norm_comparison_rank_2(tol, comm, tensor_name, parent_x, parent_ref)
        elif name_x[-1] == 'X' and name_x[-2] != 'Y' and name_x[-2] != 'Z':
          tensor_name = name_x[:-1]
          return relative_norm_comparison_rank_1(tol, comm, tensor_name, parent_x, parent_ref)
        elif name_x[-1] == 'Y' or  name_x[-1] == 'Z':
          return True, '', '' # Tested within 'X' or 'XX'
        else:
          return relative_norm_comparison(tol, comm)(x, ref)
      else:
        return equal_array_comparison(comm)(nodes_stack)

    @staticmethod
    def modify_name(path): # Ugly hack around CGNS being retarded
      suffixes = ['X','Y','Z']
      if path[-1] in suffixes:
        path = path[:-1]
      # remove a second time in case of a tensor
      if path[-1] in suffixes:
        path = path[:-1]
      return path
  return impl()
