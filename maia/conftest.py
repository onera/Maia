import pytest
import os
from mpi4py import MPI


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
  #Only proc 0 holds test results, others are empty
  if MPI.COMM_WORLD.Get_rank() == 0:
    config.option.xmlpath = "junit_unit.xml"
    
