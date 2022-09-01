import sys
from mpi4py import MPI

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback): 

  rank = MPI.COMM_WORLD.Get_rank()
  err_mssg = f"Your application aborted because of an uncaught exception on rank {rank}:\n\n"

  sys.stderr.write(err_mssg)
  sys_excepthook(type, value, traceback) 
  sys.stderr.write('\n')
  sys.stdout.flush()
  sys.stderr.flush()

  MPI.COMM_WORLD.Abort() 

def enable_mpi_excepthook():
  sys.excepthook = mpi_excepthook
def disable_mpi_excepthook():
  sys.excepthook = sys_excepthook
