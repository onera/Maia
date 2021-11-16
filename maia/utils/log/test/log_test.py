from mpi4py import MPI
from maia.utils.log import log

def test_variable_log_string():
  my_variable = 123
  assert \
      log.variable_log_string(my_variable,0) \
   == log.bold+log.blue + \
        "rank "+str(MPI.COMM_WORLD.Get_rank())+": " + \
      log.reset + \
      "my_variable = 123" # notice that the name "my_variable" has been stringified
