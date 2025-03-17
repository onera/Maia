from mpi4py import MPI
import re
from maia.utils.logging import _debug
import pytest_parallel


def test_variable_log_string():
  my_variable = 123
  assert \
      _debug.variable_log_string(my_variable,0) \
   == _debug.colors.bold+_debug.colors.blue + \
        "rank "+str(MPI.COMM_WORLD.Get_rank())+": " + \
      _debug.colors.reset + \
      "my_variable = 123" # notice that the name "my_variable" has been stringified 

@pytest_parallel.mark.parallel(2)
def test_slog(capsys, comm):
  rank = comm.Get_rank()
  message= f"Hello from rank {rank}" 
  _debug.slog(comm, message)

  # process 0 receive messages from others process and print it
  captured= capsys.readouterr()
  output= captured.out
  if rank==0:
    expected = '\x1b[34mRank 0: \x1b[0mHello from rank 0\n\x1b[34mRank 1: \x1b[0mHello from rank 1\n'
  else:
    expected = ''
  assert output == expected


@pytest_parallel.mark.parallel(2)
def test_log(capsys, comm):
    rank = comm.Get_rank()
    test_var = 42

    _debug.log(test_var)
    captured = capsys.readouterr().out
    
    assert captured == f'\x1b[1m\x1b[34mrank {rank}: \x1b[0mtest_var = 42\n'