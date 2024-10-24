from mpi4py import MPI
import numpy as np
import maia.pytree as PT

def sq_norm(x):
  return np.sum(x*x)

def norm(x, comm):
  res = comm.allreduce(sq_norm(x), MPI.SUM)
  return np.sqrt(res)

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

class EqualArray:
  def __init__(self, comm=MPI.COMM_SELF):
    self.comm = comm
  def __call__(self, stack1, stack2):
    node_x,node_ref = stack1[-1], stack2[-1]
    x   = PT.get_value(node_x)
    ref = PT.get_value(node_ref)
    return equal_array_report(x, ref, self.comm)


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

    # floating point value closest to 0 before loosing precision (see 'denormal numbers')
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
        # Report field mean information. We use the RMS (root mean square) instead of the arithmetic mean
        # Because it it the one coherent with the L2 norm (arithmetic mean would be coherent to the L1 norm)
        msg = f'RMS mean diff: {norm_diff/sqN:.3e}, RMS ref mean: {norm_ref/sqN:.3e}, rel error: {norm_diff/norm_ref:.3e}'
      if info['denorm']:
        msg += ' -- WARNING: imprecise comparison because of small reference'

      if info['within_tol']:
        warn_msg = msg
      else:
        err_msg = msg

    return info['within_tol'], err_msg, warn_msg
  return impl


class FieldComparison(EqualArray):
  """ Creates a function to compare scalar fields with a relative tolerance

  Args:
    tol (Float): tolerance
    comm (MPIComm): MPI communicator on which to call the collective comparison
  """
  def __init__(self, tol, comm):
    EqualArray.__init__(self, comm)
    self.tol = tol
  def __call__(self, stack1, stack2):
    node_x,node_ref = stack1[-1], stack2[-1]
    x   = PT.get_value(node_x,raw=True)
    ref = PT.get_value(node_ref,raw=True)
    if x.dtype.kind == 'f':
      return relative_norm_comparison(self.tol, self.comm)(x, ref)
    else:
      return EqualArray.__call__(self, stack1, stack2)


def _relative_tensor_norm_comparison(tol, comm, x_nodes, ref_nodes, tensor_rank):
  x_val   = [PT.get_value(x_node  ) for x_node   in x_nodes  ]
  ref_val = [PT.get_value(ref_node) for ref_node in ref_nodes]

  x_cat   = np.concatenate(x_val)
  ref_cat = np.concatenate(ref_val)

  return relative_norm_comparison(tol, comm, n_dim=len(x_nodes))(x_cat, ref_cat)

def _sym_to_full_rank_2_tensor(flds, dim):
  if dim == 2:
    return [flds[0],flds[1],
            flds[1],flds[2]]
  elif dim == 3:
    return [flds[0],flds[1],flds[2],
            flds[1],flds[3],flds[4],
            flds[2],flds[4],flds[5]]
  else:
    raise AssertionError(f'dimension {dim} is not implemented')

suffixes = {
  ('rank_1','2D'): ['X','Y'],
  ('rank_1','3D'): ['X','Y','Z'],
  ('rank_2','2D'): ['XX','XY','YX','YY'],
  ('rank_2','3D'): ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ'],
}
suffixes_sym = {
  '2D' : ['XX','XY','YY'],
  '3D' : ['XX','XY','XZ',  'YY','YZ',  'ZZ'],
}

def find_and_check_tensor_fields(node, tensor_name, tensor_rank):
  fields = [PT.get_node_from_name(node, tensor_name+suffix) for suffix in suffixes[(f'rank_{tensor_rank}','3D')]]
  fields = [f for f in fields if f is not None]

  fld_suffs = [PT.get_name(f)[-tensor_rank:] for f in fields]

  sorted_pairs = sorted(zip(fld_suffs, fields))
  fld_suffs = [pair[0] for pair in sorted_pairs]
  fields    = [pair[1] for pair in sorted_pairs]

  dim = 3 if any('Z' in node for node in fld_suffs) else 2

  possible_suffs     = suffixes[(f'rank_{tensor_rank}',f'{dim}D')]
  possible_suffs_sym = suffixes_sym[f'{dim}D']
  if tensor_rank == 1:
    if len(fld_suffs)==len(possible_suffs) and fld_suffs == possible_suffs:
      return fields
    else:
      err_msg = f'Tensor field \'{tensor_name}\' of rank {tensor_rank} in dimension {dim}: found components {fld_suffs}.\n' \
                f'It does not match components {possible_suffs} (vector in {dim}D).\n'
      raise RuntimeError(err_msg)
  elif tensor_rank == 2:
    if len(fld_suffs)==len(possible_suffs) and fld_suffs == possible_suffs:
      return fields
    elif len(fld_suffs)==len(possible_suffs_sym) and fld_suffs == possible_suffs_sym:
      return _sym_to_full_rank_2_tensor(fields, dim)
    else:
      err_msg = f'Tensor field \'{tensor_name}\' of rank {tensor_rank} in dimension {dim}: found components {fld_suffs}.\n' \
                f'It does not match components {possible_suffs} (full {tensor_rank}-tensor in {dim}D),\n' \
                f'or components {possible_suffs_sym} (full {tensor_rank}-tensor in {dim}D)\n'
      raise RuntimeError(err_msg)
  else:
    raise AssertionError(f'tensor_rank {tensor_rank} is not implemented')


def _tensor_info(name):
  if name[-2:] in suffixes[('rank_2','3D')]: # If the name matches order 2 tensors
    tensor_rank = 2
    tensor_name = name[:-2]
    is_first_component = name[-2:] == 'XX'
  elif name[-1:] in suffixes[('rank_1','3D')]: # If the name matches order 1 tensors
    tensor_rank = 1
    tensor_name = name[:-1]
    is_first_component = name[-1:] == 'X'
  else:
    tensor_rank = 0
    tensor_name = ''
    is_first_component = True
  return tensor_rank, is_first_component, tensor_name


class TensorFieldComparison(EqualArray):
  """ Creates a function to compare tensor fields with a relative tolerance

  To identify tensors, the functions looks at the name of the current field.
  If it ends with 'X' or 'XX', then it will look for 'Y'/'Z' or 'XY'/... sibling nodes,
  reconstruct a tensor field from them, and then do the comparison on them

  Tensor of rank 0 (i.e. scalar field), 1 and 2 are supported.
  Rank-2 tensors that only have components ['XX','XY','YY'] or ['XX','XY','XZ','YY','YZ','ZZ'] are interpreted as symmetric tensors.
  Missing components (e.g. having 'VelocityZ' without 'VelocityX/Y') will result in a error.

  Args:
    tol (Float): tolerance
    comm (MPIComm): MPI communicator on which to call the collective comparison
  """
  @staticmethod
  def modify_name(path): # Ugly hack around CGNS being retarded
    suffixes = ['X','Y','Z']
    if path[-1] in suffixes:
      path = path[:-1]
    # remove a second time in case of a tensor
    if path[-1] in suffixes:
      path = path[:-1]
    return path

  def __init__(self, tol, comm):
    EqualArray.__init__(self, comm)
    self.tol = tol

  def __call__(self, stack1, stack2):
    node_x,node_ref = stack1[-1], stack2[-1]
    name_x = PT.get_name(node_x)
    x   = PT.get_value(node_x,raw=True)
    ref = PT.get_value(node_ref,raw=True)
    if PT.get_label(node_x) == 'DataArray_t' and x.dtype.kind == 'f':
      parent_x,parent_ref = stack1[-2], stack2[-2]
      tensor_rank, is_first_component, tensor_name = _tensor_info(name_x)
      if tensor_rank>0:
        x_nodes   = find_and_check_tensor_fields(parent_x  , tensor_name, tensor_rank)
        ref_nodes = find_and_check_tensor_fields(parent_ref, tensor_name, tensor_rank)
        if is_first_component:
          return _relative_tensor_norm_comparison(self.tol, self.comm, x_nodes, ref_nodes, tensor_rank)
        else:
          return True, '', '' # Other component are actually tested within by the first component
      else: # scalar
        return relative_norm_comparison(self.tol, self.comm)(x, ref)
    else:
      return EqualArray.__call__(self, stack1, stack2)
