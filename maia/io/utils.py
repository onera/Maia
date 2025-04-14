def create_parent_folder(comm,filename):
  """create_parent_folder(comm,filename)

  Creation of parent folder of filename

  Args:
    filename(str) : name of the file
    comm     (MPIComm) : MPI communicator

  """

  from pathlib import Path
  rank = comm.Get_rank()
  if rank == 0:
    parent_folder = Path(filename).parent
    Path.mkdir(parent_folder, exist_ok=True)
  comm.barrier()