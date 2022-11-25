class colors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


def colored(s, color):
  return color + s + colors.ENDC


def slog(comm, *args):
  """
   synchronized log
  """
  rk = comm.Get_rank()
  n_rk = comm.Get_size()
  msg = ''.join([str(arg) for arg in args])
  msg = colored(f'Rank {rk}: ',colors.OKBLUE) + msg;

  comm.Barrier()
  if rk == 0:
    print(msg)
    for i in range(1,n_rk):
      recv_msg = comm.recv(source=i, tag=i)
      print(recv_msg)
  else:
    comm.send(msg, dest=0, tag=rk)
  comm.Barrier()

