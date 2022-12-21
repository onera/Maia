import inspect
from mpi4py import MPI

class colors:
  #https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html#8-colors
  black   = "\u001b[30m"
  red     = "\u001b[31m"
  green   = "\u001b[32m"
  yellow  = "\u001b[33m"
  blue    = "\u001b[34m"
  magenta = "\u001b[35m"
  cyan    = "\u001b[36m"
  white   = "\u001b[37m"

  bold    = "\u001b[1m"
  underl  = "\u001b[4m"
  revers  = "\u001b[7m"

  reset   = "\u001b[0m"


def color_str(col,s):
  return col + str(s) + colors.reset


def slog(comm, *args):
  """ Synchronized log. Needs to be called collectively from all procs in `comm` """
  rk = comm.Get_rank()
  n_rk = comm.Get_size()
  msg = ''.join([str(arg) for arg in args])
  msg = color_str(f'Rank {rk}: ',colors.blue) + msg;

  comm.Barrier()
  if rk == 0:
    print(msg)
    for i in range(1,n_rk):
      recv_msg = comm.recv(source=i, tag=i)
      print(recv_msg)
  else:
    comm.send(msg, dest=0, tag=rk)
  comm.Barrier()


# https://stackoverflow.com/a/18425523/1583122
def retrieve_name(var,unwinding_steps):
  """ Use this to get the name of the variable in the code """
  callers_local_vars = inspect.currentframe()
  for i in range(unwinding_steps+1):
    callers_local_vars = callers_local_vars.f_back
  local_vars = callers_local_vars.f_locals.items()
  return [var_name for var_name, var_val in local_vars if var_val is var]

def variable_log_string(var,unwinding_steps):
  return color_str(colors.bold+colors.blue,"rank "+str(MPI.COMM_WORLD.Get_rank())+": ") \
    + retrieve_name(var,unwinding_steps+1)[0]+" = "+str(var)

def log(var):
  """ To be used as a shorthand in a debugging context """
  print(variable_log_string(var,1)) # unwind one time since we want the name of var where ELOG is called
