from pathlib import Path

def create_parent_folder(filename, comm):
  """ Creation of parent folder of filename """

  rank = comm.Get_rank()
  if rank == 0:
    parent_folder = Path(filename).parent
    Path.mkdir(parent_folder, parents = True, exist_ok=True)
  comm.barrier()