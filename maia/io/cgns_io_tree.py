import os
import maia.pytree        as PT

from .distribution_tree         import add_distribution_info, clean_distribution_info
from .hdf.tree                  import create_tree_hdf_filter
from .fix_tree                  import ensure_PE_global_indexing, _enforce_pdm_dtype

from maia.factory     import distribute_tree
from maia.pytree.yaml import parse_yaml_cgns

def load_collective_size_tree(filename, comm, legacy=False):
  if legacy:
    from ._hdf_io_cass import load_collective_size_tree
  else:
    from ._hdf_io_h5py import load_collective_size_tree
  return load_collective_size_tree(filename, comm)

def load_partial(filename, dist_tree, hdf_filter, comm, legacy):
  if legacy:
    from ._hdf_io_cass import load_partial
    load_partial(filename, dist_tree, hdf_filter, comm)
  else:
    from ._hdf_io_h5py import load_partial
    load_partial(filename, dist_tree, hdf_filter)

def write_partial(filename, dist_tree, hdf_filter, comm, legacy):
  if legacy:
    from ._hdf_io_cass import write_partial
  else:
    from ._hdf_io_h5py import write_partial
  write_partial(filename, dist_tree, hdf_filter, comm)

def write_tree(tree, filename, links=[], legacy=False):
  """Sequential write to a CGNS file.

  Args:
    tree (CGNSTree) : Tree to write
    filename (str) : Path of the file
    links   (list) : List of links to create (see SIDS-to-Python guide)

  Example:
      .. literalinclude:: snippets/test_io.py
        :start-after: #write_tree@start
        :end-before: #write_tree@end
        :dedent: 2
  """
  if legacy:
    from ._hdf_io_cass import write_full
  else:
    from ._hdf_io_h5py import write_full
  write_full(filename, tree, links=links)

def read_tree(filename, legacy=False):
  """Sequential load of a CGNS file. 

  Args:
    filename (str) : Path of the file
  Returns:
    CGNSTree: Full (not distributed) CGNS tree
  """
  if legacy:
    from ._hdf_io_cass import read_full
  else:
    from ._hdf_io_h5py import read_full
  return read_full(filename)



def load_tree_from_filter(filename, dist_tree, comm, hdf_filter, legacy):
  """
  """
  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() \
      if isinstance(value, (list, tuple))}

  load_partial(filename, dist_tree, hdf_filter_with_dim, comm, legacy)

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

    load_partial(filename, dist_tree, next_hdf_filter, comm, legacy)

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

  write_partial(filename, saving_dist_tree, hdf_filter_with_dim, comm, legacy)

def fill_size_tree(tree, filename, comm, legacy=False):
  add_distribution_info(tree, comm)
  hdf_filter = create_tree_hdf_filter(tree)
  # Coords#Size appears in dict -> remove it
  if not legacy:
    hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}

  load_tree_from_filter(filename, tree, comm, hdf_filter, legacy)
  if not legacy:
    PT.rm_nodes_from_name(tree, '*#Size')


def file_to_dist_tree(filename, comm, legacy=False):
  """Distributed load of a CGNS file.

  Args:
    filename (str) : Path of the file
    comm     (MPIComm) : MPI communicator
  Returns:
    CGNSTree: Distributed CGNS tree
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
    fill_size_tree(dist_tree, filename, comm, legacy)

  return dist_tree

def dist_tree_to_file(dist_tree, filename, comm, legacy=False):
  """Distributed write to a CGNS file.

  Args:
    dist_tree (CGNSTree) : Distributed tree to write
    filename (str) : Path of the file
    comm     (MPIComm) : MPI communicator
  """
  filename = str(filename)
  hdf_filter = create_tree_hdf_filter(dist_tree)
  save_tree_from_filter(filename, dist_tree, comm, hdf_filter, legacy)

def write_trees(tree, filename, comm, legacy=False):
  """Sequential write to CGNS files.

  Write separate trees for each process. Rank id will be automatically
  inserted in the filename.

  Args:
    tree (CGNSTree) : Tree to write
    filename (str) : Path of the file
    comm     (MPIComm) : MPI communicator

  Example:
      .. literalinclude:: snippets/test_io.py
        :start-after: #write_trees@start
        :end-before: #write_trees@end
        :dedent: 2
  """
  # Give to each process a filename
  base_name, extension = os.path.splitext(filename)
  base_name += f"_{comm.Get_rank()}"
  _filename = base_name + extension
  write_tree(tree, _filename, links=[], legacy=legacy)
