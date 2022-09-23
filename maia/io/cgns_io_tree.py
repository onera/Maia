import os
import maia.pytree        as PT

from .distribution_tree         import add_distribution_info, clean_distribution_info
from .load_collective_size_tree import load_collective_size_tree
from .hdf.tree                  import create_tree_hdf_filter
from .fix_tree                  import ensure_PE_global_indexing, _enforce_pdm_dtype

from maia.factory     import distribute_tree
from maia.pytree.yaml import parse_yaml_cgns

def load_partial_cassiopee(filename, dist_tree, hdf_filter, comm):
  import Converter.PyTree   as C
  partial_dict_load = C.convertFile2PartialPyTreeFromPath(filename, hdf_filter, comm)

  for path, data in partial_dict_load.items():
    if path.startswith('/'):
      path = path[1:]
    node = PT.get_node_from_path(dist_tree, path)
    node[1] = data

def write_partial_cassiopee(filename, dist_tree, hdf_filter, comm):
  import Converter.PyTree   as C
  C.convertPyTree2FilePartial(dist_tree, filename, comm, hdf_filter, ParallelHDF=True)

def load_partial_h5py(filename, dist_tree, hdf_filter):
  from h5py import h5f
  from .hdf._hdf_cgns import open_from_path, load_data_partial
  fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDONLY)

  for path, filter in hdf_filter.items():
    if isinstance(filter, (list, tuple)):
      node = PT.get_node_from_path(dist_tree, path[1:]) #! Path has '/'
      gid = open_from_path(fid, path[1:])
      node[1] = load_data_partial(gid, filter)

def write_partial_h5py(filename, dist_tree, hdf_filter, comm):
  from mpi4py import MPI
  from h5py import h5p, h5f, h5fd
  from .hdf._hdf_cgns import write_lazy_wrapper, write_data_partial, open_from_path

  if comm.Get_rank() == 0:

    def skip_data(pname_and_label, name_and_label):
      if pname_and_label[1] == 'UserDefinedData_t':
        return False
      if name_and_label[1] in ['IndexArray_t']:
        return True
      if name_and_label[1] in ['DataArray_t']:
        if pname_and_label[1] not in ['Periodic_t', 'ReferenceState_t']:
          return True
      return False

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

def load_tree_from_filter(filename, dist_tree, comm, hdf_filter, legacy):
  """
  """
  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() \
      if isinstance(value, (list, tuple))}

  if legacy:
    load_partial_cassiopee(filename, dist_tree, hdf_filter_with_dim, comm)
  else:
    load_partial_h5py(filename, dist_tree, hdf_filter_with_dim)

  # > Match with callable
  hdf_filter_with_func = {key: value for (key, value) in hdf_filter.items() \
      if not isinstance(value, (list, tuple))}
  unlock_at_least_one = True
  while(len(hdf_filter_with_func) > 0 and unlock_at_least_one ):
    # Update if you can
    next_hdf_filter = dict()
    unlock_at_least_one = False
    for key, f in hdf_filter_with_func.items():
      try:
        f(next_hdf_filter)
        unlock_at_least_one = True
      except RuntimeError: # Not ready yet
        pass

    if legacy:
      load_partial_cassiopee(filename, dist_tree, next_hdf_filter, comm)
    else:
      load_partial_h5py(filename, dist_tree, next_hdf_filter)

    hdf_filter_with_func = {key: value for (key, value) in next_hdf_filter.items() \
        if not isinstance(value, (list, tuple))}

  if(unlock_at_least_one is False):
    raise RuntimeError("Something strange in the loading process")

  ensure_PE_global_indexing(dist_tree)


def save_tree_from_filter(filename, dist_tree, comm, hdf_filter, legacy):
  """
  """
  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() if isinstance(value, list)}
  hdf_filter_with_func = {key: value for (key, value) in hdf_filter.items() if not isinstance(value, list)}

  next_hdf_filter = dict()
  for key, f in hdf_filter_with_func.items():
    f(hdf_filter_with_dim)

  #Dont save distribution info, but work on a copy to keep it for further use
  saving_dist_tree = PT.shallow_copy(dist_tree)
  clean_distribution_info(saving_dist_tree)

  if legacy:
    write_partial_cassiopee(filename, saving_dist_tree, hdf_filter_with_dim, comm)
  else:
    write_partial_h5py(filename, saving_dist_tree, hdf_filter_with_dim, comm)

def file_to_dist_tree(filename, comm, distribution_policy='uniform', legacy=False):
  """
  Distributed load of filename. Return a dist_tree.
  """
  filename = str(filename)
  if os.path.splitext(filename)[1] == '.yaml':
    if comm.Get_rank() == 0:
      with open(filename, 'r') as f:
        tree = parse_yaml_cgns.to_cgns_tree(f)
        _enforce_pdm_dtype(tree)  
    else:
      tree = None
    dist_tree = distribute_tree(tree, comm, owner=0) 

  else:
    dist_tree = load_collective_size_tree(filename, comm, legacy)
    add_distribution_info(dist_tree, comm, distribution_policy)

    hdf_filter = create_tree_hdf_filter(dist_tree)

    # Coords#Size appears in dict -> remove it
    if not legacy:
      hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}

    load_tree_from_filter(filename, dist_tree, comm, hdf_filter, legacy)
    if not legacy:
      PT.rm_nodes_from_name(dist_tree, '*#Size')

  return dist_tree

def dist_tree_to_file(dist_tree, filename, comm, hdf_filter = None, legacy=False):
  """
  Distributed write of cgns_tree into filename.
  """
  filename = str(filename)
  if hdf_filter is None:
    hdf_filter = create_tree_hdf_filter(dist_tree)
  save_tree_from_filter(filename, dist_tree, comm, hdf_filter, legacy)
