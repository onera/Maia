import inspect
from mpi4py import MPI

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
  return col + str(s) + reset

# https://stackoverflow.com/a/18425523/1583122
def retrieve_name(var,unwinding_steps):
  """ Use this to get the name of the variable in the code """
  callers_local_vars = inspect.currentframe()
  for i in range(unwinding_steps+1):
    callers_local_vars = callers_local_vars.f_back
  local_vars = callers_local_vars.f_locals.items()
  return [var_name for var_name, var_val in local_vars if var_val is var]

def variable_log_string(var,unwinding_steps):
  return color_str(bold+blue,"rank "+str(MPI.COMM_WORLD.Get_rank())+": ") \
    + retrieve_name(var,unwinding_steps+1)[0]+" = "+str(var)

def DLOG(var):
  """ To be used as a shorthand in a debugging context """
  print(variable_log_string(var,1)) # unwind one time since we want the name of var where ELOG is called
