import pytest
import os
from mpi4py import MPI


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):

  comm = MPI.COMM_WORLD
  if comm.Get_rank() == 0:
    if not os.path.exists('reports'):
      os.makedirs('reports')
    if not os.path.exists('reports/assets'):
      os.makedirs('reports/assets')
  comm.barrier()

  #Only proc 0 holds test results, others are empty
  if comm.Get_rank() == 0:
    config.option.xmlpath  = 'reports/' + "report_unit_test.xml"
