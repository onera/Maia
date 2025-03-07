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

# TESTED focntionnel with framework ptst_parallel.mark.parallel but it 

@pytest_parallel.mark.parallel(2)
def test_slog(capsys, comm):
  rank = comm.Get_rank()
  message= f"Hello from rank {rank}"
  _debug.slog(comm, message)
  #assert False
  # process 0 receive messages from others process and print it
  if rank==0:
    captured= capsys.readouterr()
    output= captured.out
    for i in range(comm.Get_size()):
      expected_message=f"Hello from rank {i}"
      assert expected_message in output

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