import pytest
import os
import glob
from mpi4py import MPI

import maia.io
from maia.utils                              import test_utils as TU
from maia.utils.py_utils                     import uniform_distribution_at

from maia.conftest import is_master_process, rewrite_junit_report

def generate_cgns_files(comm):
  """
  Generate the CGNS files from the yaml files before launching the tests
  """
  yaml_folder = os.path.join(TU.mesh_dir)
  filenames = glob.glob(yaml_folder + '/*.yaml')
  start, end = uniform_distribution_at(len(filenames), comm.Get_rank(), comm.Get_size())
  for filename in filenames[start:end]:
    tree = maia.io.file_to_dist_tree(filename, MPI.COMM_SELF)
    maia.io.dist_tree_to_file(tree, os.path.splitext(filename)[0] + '.hdf', MPI.COMM_SELF)

def pytest_addoption(parser):
  parser.addoption("--gen_hdf", dest='gen_hdf', action='store_true')
  parser.addoption("--write_output", dest='write_output', action='store_true')

@pytest.fixture
def write_output(request):
  """ This get the value of command line argument write_ouput before
  launching the tests
  """
  return request.config.getoption("--write_output")

@pytest.hookimpl(tryfirst=True) # False ?
def pytest_configure(config):
  comm = MPI.COMM_WORLD

  #Only master process holds test results, others are empty
  if is_master_process(config):
    config.option.xmlpath = "junit_maia_func.xml"

  if config.getoption('gen_hdf'):
    generate_cgns_files(comm)
  comm.barrier()

def pytest_sessionfinish(session, exitstatus):
  if is_master_process(session.config):
    rewrite_junit_report(session.config.option.xmlpath, 'Functional')