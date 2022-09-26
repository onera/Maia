from mpi4py import MPI
from h5py   import h5p, h5f, h5fd

import maia.pytree as PT

from .hdf._hdf_cgns import open_from_path
from .hdf._hdf_cgns import load_lazy_wrapper, write_lazy_wrapper
from .hdf._hdf_cgns import load_data_partial, write_data_partial
from .fix_tree      import fix_point_ranges

def skip_data(pname_and_label, name_and_label):
  """ Function used to determine if the data are heavy or not """
  if pname_and_label[1] == 'UserDefinedData_t':
    return False
  if name_and_label[1] in ['IndexArray_t']:
    return True
  if name_and_label[1] in ['DataArray_t']:
    if pname_and_label[1] not in ['Periodic_t', 'ReferenceState_t']:
      return True
  return False

def load_collective_size_tree(filename, comm):

  if comm.Get_rank() == 0:
    size_tree = load_lazy_wrapper(filename, skip_data)
    fix_point_ranges(size_tree)
  else:
    size_tree = None

  size_tree = comm.bcast(size_tree, root=0)

  return size_tree

def load_partial(filename, dist_tree, hdf_filter):
  fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDONLY)

  for path, filter in hdf_filter.items():
    if isinstance(filter, (list, tuple)):
      node = PT.get_node_from_path(dist_tree, path[1:]) #! Path has '/'
      gid = open_from_path(fid, path[1:])
      node[1] = load_data_partial(gid, filter)

def write_partial(filename, dist_tree, hdf_filter, comm):

  if comm.Get_rank() == 0:
    write_lazy_wrapper(dist_tree, filename, skip_data)
  comm.barrier()

  fapl = h5p.create(h5p.FILE_ACCESS)
  fapl.set_driver(h5fd.MPIO)
  fapl.set_fapl_mpio(comm, MPI.Info())
  fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR, fapl)

  for path, filter in hdf_filter.items():
    array = PT.get_node_from_path(dist_tree, path[1:])[1] #! Path has '/'
    gid = open_from_path(fid, path[1:])
    write_data_partial(gid, array, filter)
    gid.close()
  
  fid.close()

def read_full(filename):
  return load_lazy_wrapper(filename, lambda X,Y: False)

def write_full(filename, dist_tree):
  write_lazy_wrapper(dist_tree, filename, lambda X,Y: False)

