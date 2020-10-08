import pytest
import os
from mpi4py import MPI

# --------------------------------------------------------------------------
@pytest.fixture
def sub_comm(request):
  """
  """
  comm = MPI.COMM_WORLD

  nproc = request.param

  # Groups communicator creation
  gprocs   = [i for i in range(nproc)]
  group    = comm.Get_group()
  subgroup = group.Incl(gprocs)
  subcomm  = comm.Create(subgroup)

  comm.Barrier()

  return subcomm

# --------------------------------------------------------------------------
def assert_mpi(comm, rank, cond ):
  if(comm.rank == rank):
    print("assert_mpi_maia --> ", cond)
    assert(cond == True)
  else:
    pass

# https://stackoverflow.com/questions/59577426/how-to-rename-the-title-of-the-html-report-generated-by-pytest-html-plug-in
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
  # to remove environment section
  config._metadata = None

  if not os.path.exists('reports'):
    os.makedirs('reports')

  comm = MPI.COMM_WORLD
  config.option.htmlpath = 'reports/' + "report_unit_test_{0}.html".format(comm.rank)

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
