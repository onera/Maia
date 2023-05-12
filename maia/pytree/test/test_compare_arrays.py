import pytest
import os
import numpy as np

from mpi4py import MPI

import maia.pytree as PT
from maia.pytree.yaml   import parse_yaml_cgns
from maia.pytree.compare_arrays import close_in_relative_norm, equal_array_report
import pytest_parallel

def test_close_in_relative_norm():
  tol = 1.e-12

# Scalar cases
  # unambiguous same
  ref = 1.
  val = 1.
  assert close_in_relative_norm([val], [ref], tol, MPI.COMM_SELF) == True

  # unambiguous different
  ref = 1.
  val = 2.
  assert close_in_relative_norm([val], [ref], tol, MPI.COMM_SELF) == False

  # seems close, but not relatively speaking
  ref = 1.e-20
  val = 2.e-20
  assert close_in_relative_norm([val], [ref], tol, MPI.COMM_SELF) == False

  # seems far, but not relatively speaking
  ref = 1.e20
  val = ref + 1000
  assert close_in_relative_norm([val], [ref], tol, MPI.COMM_SELF) == True

# The reference is close or equal to 0.
  # case the reference is exactly 0. :
  ref = 0.
  val = 0. # if the value is also 0, then they are equal
  assert close_in_relative_norm([val], [ref], tol, MPI.COMM_SELF) == True
  val = 1.e-20 # this is too much even with very loose tolerance
  assert close_in_relative_norm([val], [ref], tol = 10., comm = MPI.COMM_SELF) == False

  # note that the behavior is not symmetrical, which is OK: we compare a value to a reference, not two indistinct values
  ref = 1.e-20
  val = 0.
  assert close_in_relative_norm([val], [ref], tol = 10., comm = MPI.COMM_SELF) == True

  # when the values are too small to keep the floating point precision, we compare as if tol=0.
  try:
    smallest_normal = np.finfo(np.float64).smallest_normal
  except AttributeError:
    smallest_normal = np.float64(2.2250738585072014e-308)
    import warnings
    warnings.warn(f'`np.finfo(np.float64).smallest_normal` ' \
                  f'does not exist with your NumPy version. ' \
                  f'using {smallest_normal}', DeprecationWarning)
  very_small = smallest_normal / 10
  ref = very_small
  val = ref/10
  assert close_in_relative_norm([val], [ref], tol = 1., comm = MPI.COMM_SELF) == False
  assert close_in_relative_norm([val], [ref], tol = 0., comm = MPI.COMM_SELF) == False # same in this case

# Fields
  # unambiguous same
  ref = [1.,2.]
  val = [1.,2.]
  assert close_in_relative_norm(val, ref, tol, MPI.COMM_SELF) == True

  # unambiguous different
  ref = [1.,2.]
  val = [1.,3.]
  assert close_in_relative_norm(val, ref, tol, MPI.COMM_SELF) == False

  # for Bruno ;)
  ref = [1.,2.]
  val = [2.,1.]
  assert close_in_relative_norm(val, ref, tol, MPI.COMM_SELF) == False

  # the first values are far in isolation, but the field is globally similar
  ref = [1.,1.e20]
  val = [2.,1.e20]
  assert close_in_relative_norm(val, ref, tol, MPI.COMM_SELF) == True

  # the first values seem close, but the field only has small absolute values
  ref = [1.e-20,1.e-20]
  val = [2.e-20,1.e-20]
  assert close_in_relative_norm(val, ref, tol, MPI.COMM_SELF) == False


@pytest_parallel.mark.parallel(2)
def test_equal_array_report(comm):
  # Equal
  if comm.Get_rank() == 0:
    x   = np.array([0,1,2])
    ref = np.array([0,1,2])
  else:
    x   = np.array([3,4])
    ref = np.array([3,4])
  assert equal_array_report(x, ref, comm) == (True, '', '')

  # Not equal on rank 0
  if comm.Get_rank() == 0:
    x   = np.array([0,1,2])
    ref = np.array([0,1,9])
  else:
    x   = np.array([3,4])
    ref = np.array([3,4])
  is_same, report, _ = equal_array_report(x, ref, comm)
  if comm.Get_rank() == 0:
    assert is_same == False
    assert report == '[0 1 2 3 4] <> [0 1 9 3 4]'
  else:
    assert is_same == False
    assert report == ''

  # Not equal on rank 1
  if comm.Get_rank() == 0:
    x   = np.array([0,1,2])
    ref = np.array([0,1,2])
  else:
    x   = np.array([3,4])
    ref = np.array([3,9])
  is_same, report, _ = equal_array_report(x, ref, comm)
  if comm.Get_rank() == 0:
    assert is_same == False
    assert report == '[0 1 2 3 4] <> [0 1 2 3 9]'
  else:
    assert is_same == False
    assert report == ''


  # Not equal and array too big to be printed
  if comm.Get_rank() == 0:
    x   = np.array([0,1,2,4,5,6,7])
    ref = np.array([9,1,2,4,5,6,7])
  else:
    x   = np.array([8,9,10,11,12,13])
    ref = np.array([8,9,10,11,99,99])
  is_same, report, _ = equal_array_report(x, ref, comm)
  if comm.Get_rank() == 0:
    assert is_same == False
    assert report == '3 values are different'
  else:
    assert is_same == False
    assert report == ''
