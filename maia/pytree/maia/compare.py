from mpi4py import MPI
import numpy as np
from math import prod

import maia.pytree as PT

from maia.utils import par_utils
from maia.transfer import protocols as EP

def is_distributed(stack):
  last = stack[-1]
  label = PT.get_label(last)
  if label == 'IndexArray_t':
    return True
  elif label == 'DataArray_t':
    parent_label = PT.get_label(stack[-2])
    return parent_label in ['GridCoordinates_t', 'FlowSolution_t', 'DiscreteData_t', \
        'ZoneSubRegion_t', 'Elements_t', 'ArbitraryGridMotion_t', 'BCData_t']
  return False

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
  """
  A callable object generating a report for :func:`~maia.pytree.diff_tree`,
  using an exact point-to-point comparison.

  This is the extension of :func:`~maia.pytree.compare.EqualArray` for distributed trees.

  Args:
    comm (MPIComm): MPI communicator
  Example:
    >>> comp = MT.compare.EqualArray(comm)
    >>> sol1 = PT.new_FlowSolution(fields={'Density' : [1., 1.002, 1.]})
    >>> sol2 = PT.new_FlowSolution(fields={'Density' : [1., 1.001, 1.]})
    >>> PT.diff_tree(sol1, sol2, comp=comp)
    DiffReport(
      status=False,
      errors='/FlowSolution/Density -- Values differ: [1.    1.002 1.   ] <> [1.    1.001 1.   ]\\n',
      warnings=''
      )
  """

  def __init__(self, comm):
    self.comm = comm

  def is_same_value_shape(self, stack1, stack2):
    shape1 = PT.get_np_value(stack1[-1]).shape
    shape2 = PT.get_np_value(stack2[-1]).shape
    if is_distributed(stack1):
      size1 = par_utils.dn_to_distribution(prod(shape1), self.comm)[-1]
      size2 = par_utils.dn_to_distribution(prod(shape2), self.comm)[-1]
      last = stack1[-1]
      if PT.get_name(last) == 'ParentElements' and PT.get_label(stack1[-2]) == 'Elements_t':
        shape1 = (size1 // 2, 2)
        shape2 = (size2 // 2, 2)
      elif PT.get_label(last) == 'IndexArray_t':
        shape1 = (shape1[0], size1 // shape1[0])
        shape2 = (shape2[0], size2 // shape2[0])
      else:
        shape1 = (size1,)
        shape2 = (size2,)
    if shape1 != shape2:
      return False, f'{shape1} <> {shape2}', ''
    return True, '', ''

  def redistribute_value_as(self, x, ref):
    x_distri   = par_utils.dn_to_distribution(x.size, self.comm)
    ref_distri = par_utils.dn_to_distribution(ref.size, self.comm)
    x_distri_f   = par_utils.partial_to_full_distribution(x_distri, self.comm)
    ref_distri_f = par_utils.partial_to_full_distribution(ref_distri, self.comm)
    if not (x_distri_f == ref_distri_f).all():
      # Redistribute X to compare
      x = EP.block_to_block(x, x_distri_f, ref_distri_f, self.comm).reshape(ref.shape, order='F')
    return x, ref


  def __call__(self, stack1, stack2):
    node_x, node_ref = stack1[-1], stack2[-1]
    x   = PT.get_np_value(node_x)
    ref = PT.get_np_value(node_ref)
    
    if len(stack1) > 1 and PT.get_name(stack1[-2]) == ':CGNS#Distribution':
      # Distribution index itself : compare full value
      x = par_utils.partial_to_full_distribution(x, self.comm)
      ref = par_utils.partial_to_full_distribution(ref, self.comm)
      return equal_array_report(x, ref, MPI.COMM_SELF)
    elif not is_distributed(stack1):
      # Standard non distributed array --> all ranks have all data
      return equal_array_report(x, ref, MPI.COMM_SELF)
    else:
      # Distributed array -> ensure distribution is same before parallel comparison
      x, ref = self.redistribute_value_as(x, ref)
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
  """ A comparison object for :func:`~maia.pytree.diff_tree` that
  compare arrays with a relative tolerance.

  Floating points arrays are considered equal if :math:`||a-b|| \leq \mathrm{tol}\ ||b||`,
  where :math:`||\cdot||` is the :math:`L^2` norm,
  while integer arrays fallback to :func:`EqualArray` comparison.

  This function operate on distributed trees.

  Args:
    tol (float) : tolerance
    comm (MPIComm) : MPI communicator
  Exemple:
    >>> comp = MT.compare.FieldComparison(1E-2, comm)
    >>> sol1 = PT.new_FlowSolution(fields={'Density' : [1., 1.002, 1.]})
    >>> sol2 = PT.new_FlowSolution(fields={'Density' : [1., 1.001, 1.]})
    >>> PT.diff_tree(sol1, sol2, comp=comp)
    DiffReport(
      status=True,
      errors='',
      warnings='/FlowSolution/Density -- Values differ: RMS mean diff: 5.773e-04, RMS ref mean: 1.000e+00, rel error: 5.771e-04\\n'
    )
  """
  def __init__(self, tol, comm):
    EqualArray.__init__(self, comm)
    self.tol = tol
  def __call__(self, stack1, stack2):
    node_x,node_ref = stack1[-1], stack2[-1]
    x   = PT.get_np_value(node_x)
    ref = PT.get_np_value(node_ref)
    if x.dtype.kind == 'f':
      if is_distributed(stack1):
        x, ref = self.redistribute_value_as(x, ref)
        return relative_norm_comparison(self.tol, self.comm)(x, ref)
      else:
        return relative_norm_comparison(self.tol, MPI.COMM_SELF)(x, ref)
    else: # Redistribution is done in EqualArray if necessary
      return EqualArray.__call__(self, stack1, stack2)


def _relative_tensor_norm_comparison(tol, comm, x_val, ref_val, tensor_rank):

  x_cat   = np.concatenate(x_val)
  ref_cat = np.concatenate(ref_val)

  return relative_norm_comparison(tol, comm, n_dim=len(x_val))(x_cat, ref_cat)

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
  """ A comparison object for :func:`~maia.pytree.diff_tree` that
  compare tensorial fields with a relative tolerance.
  
  This comparison method is similar to :func:`FieldComparison`,
  but tensors fields components are treated together.

  Tensors of rank 0 (i.e. scalar field), 1 (components ending with ``X``, ``Y`` and ``Z``)
  and 2 (components ending with ``XX``, ``XY``, ..., ``ZZ``) are supported.
  Rank-2 tensors that only have components ``[XX, XY, YY]`` or ``[XX, XY, XZ, YY, YZ, ZZ]``
  are interpreted as symmetric tensors.
  Missing components (e.g. having ``VelocityZ`` without ``VelocityX/Y``) will result in a error.

  Args:
    tol (float) : tolerance
    comm (MPIComm): MPI communicator
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

    if PT.get_label(node_x) == 'DataArray_t' and PT.get_value_kind(node_x) == 'R':
      parent_x,parent_ref = stack1[-2], stack2[-2]
      tensor_rank, is_first_component, tensor_name = _tensor_info(PT.get_name(node_x))
      if tensor_rank>0:
        x_nodes   = find_and_check_tensor_fields(parent_x  , tensor_name, tensor_rank)
        ref_nodes = find_and_check_tensor_fields(parent_ref, tensor_name, tensor_rank)
        if is_first_component:
          x = [PT.get_np_value(n) for n in x_nodes]
          ref = [PT.get_np_value(n) for n in ref_nodes]
          if is_distributed(stack1):
            out = (self.redistribute_value_as(_x, _ref) for _x,_ref in zip(x, ref))
            x, ref = zip(*out) # Unzip output
            return _relative_tensor_norm_comparison(self.tol, self.comm, x, ref, tensor_rank)
          else:
            return _relative_tensor_norm_comparison(self.tol, MPI.COMM_SELF, x, ref, tensor_rank)
        else:
          return True, '', '' # Other component are actually tested within by the first component
      else: # scalar
        x   = PT.get_np_value(node_x)
        ref = PT.get_np_value(node_ref)
        if is_distributed(stack1):
          x, ref = self.redistribute_value_as(x, ref)
          return relative_norm_comparison(self.tol, self.comm)(x, ref)
        else:
          return relative_norm_comparison(self.tol, MPI.COMM_SELF)(x, ref)
    else:
      return EqualArray.__call__(self, stack1, stack2)
