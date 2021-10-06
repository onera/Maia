import pytest
import os
import glob
from mpi4py import MPI

from pytest_mpi_check import assert_mpi

from maia.utils                              import test_utils as TU
from maia.distribution.distribution_function import uniform_distribution_at
from maia.cgns_io                            import cgns_io_tree as IOT
# pytest.register_assert_rewrite("pytest_check.check")

def rename_reports(config, comm):
# https://stackoverflow.com/questions/59577426/how-to-rename-the-title-of-the-html-report-generated-by-pytest-html-plug-in
  if comm.Get_rank() == 0:
    if not os.path.exists('reports'):
      os.makedirs('reports')
    if not os.path.exists('reports/assets'):
      os.makedirs('reports/assets')
  comm.barrier()

  #Only proc 0 holds test results, others are empty
  if comm.Get_rank() == 0:
    config.option.htmlpath = 'reports/' + "report_func_test.html"
    config.option.xmlpath  = 'reports/' + "report_func_test.xml"

def generate_cgns_files(comm):
  """
  Generate the CGNS files from the yaml files before launching the tests
  """
  yaml_folder = os.path.join(TU.mesh_dir)
  filenames = glob.glob(yaml_folder + '/*.yaml')
  start, end = uniform_distribution_at(len(filenames), comm.Get_rank(), comm.Get_size())
  for filename in filenames[start:end]:
    tree = IOT.file_to_dist_tree(filename, MPI.COMM_SELF)
    IOT.dist_tree_to_file(tree, os.path.splitext(filename)[0] + '.hdf', MPI.COMM_SELF)

def pytest_addoption(parser):
  parser.addoption("--gen_hdf", dest='gen_hdf', action='store_true')

@pytest.hookimpl(tryfirst=True) # False ?
def pytest_configure(config):
  comm = MPI.COMM_WORLD

  rename_reports(config, comm)
  if config.getoption('gen_hdf'):
    generate_cgns_files(comm)
  comm.barrier()

  pytest.assert_mpi = assert_mpi

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):

  pytest_html = item.config.pluginmanager.getplugin('html')
  outcome = yield
  report = outcome.get_result()
  extra = getattr(report, 'extra', [])
  if report.when == 'call':
    # always add url to report
    extra.append(pytest_html.extras.url('http://www.example.com/'))
    xfail = hasattr(report, 'wasxfail')
    if (report.skipped and xfail) or (report.failed and not xfail):
      # only add additional html on failure
      extra.append(pytest_html.extras.html('<div>Additional HTML</div>'))
    report.extra = extra

