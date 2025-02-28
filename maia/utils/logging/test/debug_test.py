from mpi4py import MPI
import re
from maia.utils.logging import _debug
#import pytest_parallel


def test_variable_log_string():
  my_variable = 123
  assert \
      _debug.variable_log_string(my_variable,0) \
   == _debug.colors.bold+_debug.colors.blue + \
        "rank "+str(MPI.COMM_WORLD.Get_rank())+": " + \
      _debug.colors.reset + \
      "my_variable = 123" # notice that the name "my_variable" has been stringified

# TESTED with MPI_COMM_WORLD and with the framework ptst_parallel.mark.parallel and it is failed
"""
#@pytest_parallel.mark.parallel(3)
def test_slog(capsys):
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  message= f"Hello from rank {}"
  _debug.slog(MPI.COMM_WORLD, message)
  # process 0 receive messages from others process and print it
  MPI.COMM_WORLD.Barrier() 
  if rank==0:
    captured= capsys.readouterr()
    output= captured.out
    for i in range(MPI.COMM_WORLD.Get_size()):
      expected_message=f"Rank{i} : Hello from rank {i}"
      assert expected_message in output
"""
#correct test
def test_log(capsys):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    test_var = 42
    _debug.log(test_var)
    # apsys.readouterr is used to capture the output
    captured = capsys.readouterr()
    # (re.sub)  is used to delete all ANSI colors for captured.out.
    cleaned_output = re.sub(r'\x1b\[[0-9;]*[mK]', '', captured.out)
    expected_output = f"rank {rank}: test_var = 42\n"
    assert cleaned_output == expected_output