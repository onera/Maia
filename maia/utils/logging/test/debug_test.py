from mpi4py import MPI
from maia.utils.logging import _debug

def test_variable_log_string():
  my_variable = 123
  assert \
      _debug.variable_log_string(my_variable,0) \
   == _debug.colors.bold+_debug.colors.blue + \
        "rank "+str(MPI.COMM_WORLD.Get_rank())+": " + \
      _debug.colors.reset + \
      "my_variable = 123" # notice that the name "my_variable" has been stringified
