from mpi4py import MPI
from maia.utils.logging import debug

def test_variable_log_string():
  my_variable = 123
  assert \
      debug.variable_log_string(my_variable,0) \
   == debug.colors.bold+debug.colors.blue + \
        "rank "+str(MPI.COMM_WORLD.Get_rank())+": " + \
      debug.colors.reset + \
      "my_variable = 123" # notice that the name "my_variable" has been stringified
