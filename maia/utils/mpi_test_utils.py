import pytest
from decorator import decorate

from mpi4py import MPI

def exit_if_null_comm(tested_fun):
  def exit_if_null_comm_impl(_,sub_comm,*args,**kwargs):
    if sub_comm == MPI.COMM_NULL :
      return
    tested_fun(sub_comm,*args,**kwargs)

  # preserve the tested_fun named arguments signature for use by other pytest decorator
  tested_fun_replica = decorate(tested_fun,exit_if_null_comm_impl)

  return tested_fun_replica

def mark_mpi_test(n_proc_list):
  if type(n_proc_list)==int:
    n_proc_list = [n_proc_list]
  max_n_proc = max(n_proc_list)
  def mark_mpi_test_impl(tested_fun):
    return pytest.mark.mpi(min_size=max_n_proc) (
        pytest.mark.parametrize("sub_comm", n_proc_list, indirect=['sub_comm']) (
          exit_if_null_comm(tested_fun)
        )
      )
  return mark_mpi_test_impl
