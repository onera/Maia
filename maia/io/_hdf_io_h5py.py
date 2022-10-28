from mpi4py import MPI
from h5py   import h5p, h5f, h5fd

import maia.pytree as PT

from .hdf._hdf_cgns import open_from_path
from .hdf._hdf_cgns import load_tree_partial, write_tree_partial
from .hdf._hdf_cgns import load_data_partial, write_data_partial
from .hdf._hdf_cgns import write_link
from .fix_tree      import fix_point_ranges

def load_data(pname_and_label, name_and_label):
  """ Function used to determine if the data is heavy or not """
  if pname_and_label[1] == 'UserDefinedData_t':
    return True
  if name_and_label[1] in ['IndexArray_t']:
    return False
  if name_and_label[1] in ['DataArray_t']:
    if pname_and_label[1].endswith('Model_t'): #Stuff related to FlowEquationSet:
      return True
    if pname_and_label[1] not in ['BaseIterativeData_t', 'Periodic_t', 'ReferenceState_t']:
      return False
  return True

def load_collective_size_tree(filename, comm):

  if comm.Get_rank() == 0:
    size_tree = load_tree_partial(filename, load_data)
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
    write_tree_partial(dist_tree, filename, load_data)
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
  return load_tree_partial(filename, lambda X,Y: True)

def write_full(filename, dist_tree, links=[]):
  write_tree_partial(dist_tree, filename, lambda X,Y: True)

  # Add links if any
  fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR)
  for link in links:
    target_dir, target_file, target_node, local_node = link
    parent_node_path = PT.path_head(local_node)
    local_node_name  = PT.path_tail(local_node)
    gid = open_from_path(fid, parent_node_path)
    write_link(gid, local_node_name, target_file, target_node)
  fid.close()

