import pytest
import os
import glob
from mpi4py import MPI

import maia.io
from maia.utils                              import test_utils as TU
from maia.utils.py_utils                     import uniform_distribution_at

def rename_reports(config, comm):
  if comm.Get_rank() == 0:
    if not os.path.exists('reports'):
      os.makedirs('reports', exist_ok=True)
    if not os.path.exists('reports/assets'):
      os.makedirs('reports/assets', exist_ok=True)
  comm.barrier()

  #Only proc 0 holds test results, others are empty
  if comm.Get_rank() == 0:
    config.option.xmlpath  = 'reports/' + "report_func_test.xml"

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

  rename_reports(config, comm)
  if config.getoption('gen_hdf'):
    generate_cgns_files(comm)
  comm.barrier()
