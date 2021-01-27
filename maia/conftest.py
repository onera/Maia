import pytest
import os
from pytest_check import *
from mpi4py import MPI
import sys

pytest.register_assert_rewrite("pytest_check.check")

# --------------------------------------------------------------------------
def setup_subcomm(comm, n_proc):
  """
  """
  # Groups communicator creation
  gprocs   = [i for i in range(n_proc)]
  group    = comm.Get_group()
  subgroup = group.Incl(gprocs)
  subcomm  = comm.Create(subgroup)
  return subcomm


# --------------------------------------------------------------------------
@pytest.fixture
def make_sub_comm(request):
  """
  """
  # print(" request:: ", dir(request._pyfuncitem))
  return request._pyfuncitem._sub_comm

# --------------------------------------------------------------------------
# @pytest.fixture
# def make_sub_comm(request):
#   """
#   """
#   comm = MPI.COMM_WORLD

#   print(" request:: ", dir(request._pyfuncitem))
#   # print(" toto:: ", comm.rank, request._pyfuncitem.toto)
#   n_proc = request.param

#   # if(request._pyfuncitem.toto == comm.rank):
#   #   pytest.skip()
#   #   return None

#   return None

#   if(n_proc > comm.size):
#     pytest.skip()
#     return None

#   # Groups communicator creation
#   gprocs   = [i for i in range(n_proc)]
#   group    = comm.Get_group()
#   subgroup = group.Incl(gprocs)
#   # print(dir(comm))
#   # subcomm  = comm.Create(subgroup)      # Lock here - Collection among all procs
#   subcomm  = comm.Create_group(subgroup)  # Ok - Only collective amont groups

#   if(subcomm == MPI.COMM_NULL):
#     pytest.skip()

#   return subcomm

# --------------------------------------------------------------------------
@pytest.fixture
def sub_comm(request):
  """
  """
  comm = MPI.COMM_WORLD

  n_proc = request.param

  # Groups communicator creation
  gprocs   = [i for i in range(n_proc)]
  group    = comm.Get_group()
  subgroup = group.Incl(gprocs)
  subcomm  = comm.Create(subgroup)

  comm.Barrier()

  return subcomm

# --------------------------------------------------------------------------
def assert_mpi(comm, rank, cond ):
  if(comm.rank == rank):
    #print("assert_mpi_maia --> ", cond)
    assert(cond == True)
  else:
    pass

# --------------------------------------------------------------------------
@pytest.mark.trylast
def pytest_sessionstart(session):
  """
  """
  pass
  #print("pytest_session_start")
  # print(dir(session.session))

# --------------------------------------------------------------------------
# @pytest.mark.tryfirst
# def pytest_runtestloop(session):
#   """
#   """
#   comm = MPI.COMM_WORLD
#   print("pytest_runtestloop", comm.rank)
#   # print(dir(session.session))
#   for i, item in enumerate(session.items):
#     # if(comm.rank == i):
#       # print(i, item, dir(item))
#       print("launch on {0} item : {1}".format(comm.rank, item))
#       item.toto = i
#       # Create sub_comm !
#       # print(i, item)
#       item.config.hook.pytest_runtest_protocol(item=item, nextitem=None)
#       if session.shouldfail:
#         raise session.Failed(session.shouldfail)
#       if session.shouldstop:
#         raise session.Interrupted(session.shouldstop)
#   return True

# --------------------------------------------------------------------------
# def pytest_runtest_logreport(report):
#   """
#   """
#   comm = MPI.COMM_WORLD
#   print(f"[{comm.rank}] - pytest_runtest_logreport", report.nodeid)

# --------------------------------------------------------------------------
def pytest_runtest_setup(item):
  """
  """
  # print("my_setup", dir(item))
  # print("my_setup", dir(item.keywords))
  # print("my_setup")
  # print("item.iter_markers(name=mpi_test)", item.iter_markers(name="mpi_test"))
  for mark in item.iter_markers(name="mpi_test"):
    if mark.args:
      raise ValueError("mpi mark does not take positional args")
    try:
      from mpi4py import MPI
    except ImportError:
      pytest.fail("MPI tests require that mpi4py be installed")
    comm = MPI.COMM_WORLD
    try:
      n_rank_test = mark.kwargs.get('comm_size')
      #print("*"*100, n_rank_test)

      # > Si n_rank_test > comm.size() --> Skip
      if(n_rank_test > comm.size):
        pytest.skip(
            "Test requires {} MPI processes, only {} MPI "
            "processes specified, skipping "
            "test".format(n_rank_test, comm.size)
            )
      else: # Skip only for rank that not necessary for the test
        sub_comm    = setup_subcomm(comm, n_rank_test)

        if(sub_comm == MPI.COMM_NULL):
          pytest.skip("Test need {} MPI processes, only {} MPI "
                      "processes specified, skipping "
                      "test".format(n_rank_test, comm.size)
                      )
    except KeyError:
      pass

# --------------------------------------------------------------------------
# def pytest_runtest_call(item):
#   """
#   """
#   print("Mygod")
#   item.runtest()

# --------------------------------------------------------------------------
def pytest_runtest_teardown(item):
  """
  """
  from mpi4py import MPI
  # comm = MPI.COMM_WORLD
  # # print("pytest_runtest_teardown", comm.rank)
  # a = comm.allreduce(comm.rank)
  # comm.Barrier()

# https://stackoverflow.com/questions/59577426/how-to-rename-the-title-of-the-html-report-generated-by-pytest-html-plug-in
@pytest.hookimpl(tryfirst=True) # False ?
def pytest_configure(config):
  # to remove environment section
  config._metadata = None

  comm = MPI.COMM_WORLD
  if comm.Get_rank() == 0:
    if not os.path.exists('reports'):
      os.makedirs('reports')
    if not os.path.exists('reports/assets'):
      os.makedirs('reports/assets')
  comm.barrier()

  config.option.htmlpath = 'reports/' + "report_unit_test_{0}.html".format(comm.rank)

  pytest.assert_mpi = assert_mpi
  # temp_ouput_file = 'maia_'
  # sys.stdout = open(f"{temp_ouput_file(args.test_name)}_{MPI.COMM_WORLD.Get_rank()}", "w")
  # print(dir(config))
  # exit(2)


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
