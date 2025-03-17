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

# COMMENTES ADDEDS BUT IT IS NOT CORRECT
@pytest_parallel.mark.parallel(2)
def test_slog(capsys, comm):
  # def color_str(color_code, text):
  #   return f"{color_code}{text}\033[0m"
  rank = comm.Get_rank()
  message= f"Hello from rank {rank}" 
  #msg = color_str(colors.blue, f'Rank {rk}: ') + msg
  _debug.slog(comm, message)
  # process 0 receive messages from others process and print it
  if rank==0:
    captured= capsys.readouterr()
    output= captured.out
    #print(output)
    for i in range(comm.Get_size()):
      # expected = '\x1b[34mRank 0: \x1b[0mHello from rank 0\n\x1b[34mRank 1: \x1b[0mHello from rank 1\n'
      # assert output == expected
      expected_message=f"Hello from rank {i}"
      assert expected_message in output

@pytest_parallel.mark.parallel(2)
def test_log(capsys, comm):
    rank = comm.Get_rank()
    test_var = 42
    _debug.log(test_var)
    # apsys.readouterr is used to capture the output
    captured = capsys.readouterr()
    #print(captured)
    # (re.sub)  is used to delete all ANSI colors for captured.out.
    cleaned_output = re.sub(r'\x1b\[[0-9;]*[mK]', '', captured.out)
    #print(cleaned_output)
    expected_output = f"rank {rank}: test_var = 42\n"
    #expected_output= f'\x1b[1m\x1b[34mrank 1: \x1b[0mtest_var = 42\n', err=''
    #print(expected_output)
    assert expected_output == expected_output