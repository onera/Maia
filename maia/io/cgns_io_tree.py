_LEGACY_IO  = False
import os
import time
import mpi4py.MPI as MPI

import maia.pytree        as PT
import maia.pytree.maia   as MT
import maia.utils.logging as mlog

from .distribution_tree         import add_distribution_info, clean_distribution_info
from .hdf.tree                  import create_tree_hdf_filter
from .fix_tree                  import ensure_PE_global_indexing, ensure_signed_nface_connectivity
from .utils                     import create_parent_folder

if _LEGACY_IO:
  from . import _hdf_io_cass as _hdf_io
else:
  from . import _hdf_io_h5py as _hdf_io

from maia.factory     import full_to_dist

def load_size_tree(filename, comm):
  return _hdf_io.load_size_tree(filename, comm)

def load_partial(filename, dist_tree, hdf_filter, comm):
  if _LEGACY_IO:
    _hdf_io.load_partial(filename, dist_tree, hdf_filter, comm)
  else:
    _hdf_io.load_partial(filename, dist_tree, hdf_filter)

def write_tree(tree, filename, links=[]):
  """write_tree(tree, filename, links=[])

  Sequential write to a CGNS file.

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
  create_parent_folder(filename, MPI.COMM_SELF)
  filename = str(filename)
  _hdf_io.write_full(filename, tree, links=links)

def read_tree(filename):
  """read_tree(filename)

  Sequential load of a CGNS file.

  Args:
    filename (str) : Path of the file
  Returns:
    CGNSTree: Full (not distributed) CGNS tree
  """
  filename = str(filename)
  if os.path.splitext(filename)[1] == '.yaml':
    with open(filename, 'r') as f:
      tree = PT.yaml.to_cgns_tree(f)
    return tree
  else:
    return _hdf_io.read_full(filename)

def read_links(filename):
  """read_links(filename)

  Detect the links embedded in a CGNS file.

  Links information are returned as described in sids-to-python. Note that
  no data are loaded and the tree structure is not even built.

  Args:
    filename (str) : Path of the file
  Returns:
    list: Links description
  """
  filename = str(filename)

  return _hdf_io.read_links(filename)

def load_tree_from_filter(filename, dist_tree, comm, hdf_filter):
  """
  """
  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() \
      if isinstance(value, (list, tuple))}

  load_partial(filename, dist_tree, hdf_filter_with_dim, comm)

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

    load_partial(filename, dist_tree, next_hdf_filter, comm)

    hdf_filter_with_func = {key: value for (key, value) in next_hdf_filter.items() \
        if not isinstance(value, (list, tuple))}

  if(unlock_at_least_one is False):
    raise RuntimeError("Something strange in the loading process")

  n_shifted = ensure_PE_global_indexing(dist_tree)
  if n_shifted > 0 and comm.Get_rank() == 0:
    mlog.error(f"ParentElements arrays of NGON_n elements have been recomputed "\
               f"because they were wrongly defined (local indexing)")
  n_shifted = ensure_signed_nface_connectivity(dist_tree, comm)
  if n_shifted > 0 and comm.Get_rank() == 0:
    mlog.error(f"ElementConnectivity arrays of NFACE_n elements have been recomputed "\
               f"because they were wrongly defined (missing orientations)")

def save_tree_from_filter(filename, dist_tree, comm, hdf_filter, links):
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

  _hdf_io.write_partial(filename, saving_dist_tree, hdf_filter_with_dim, links, comm)

def fill_size_tree(tree, filename, comm):
  filename = str(filename)
  add_distribution_info(tree, comm)
  hdf_filter = create_tree_hdf_filter(tree)
  # Coords#Size appears in dict -> remove it
  hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}

  load_tree_from_filter(filename, tree, comm, hdf_filter)
  PT.rm_nodes_from_name(tree, '*#Size')


def file_to_dist_tree(filename, comm):
  """file_to_dist_tree(filename, comm)

  Distributed load of a CGNS file.

  Args:
    filename (str) : Path of the file
    comm     (MPIComm) : MPI communicator
  Returns:
    CGNSTree: Distributed CGNS tree
  """
  mlog.info(f"Distributed read of file {filename}...")
  start = time.time()
  filename = str(filename)
  if os.path.splitext(filename)[1] == '.yaml':
    if comm.Get_rank() == 0:
      with open(filename, 'r') as f:
        tree = PT.yaml.to_cgns_tree(f)
    else:
      tree = None
    dist_tree = full_to_dist.full_to_dist_tree(tree, comm, owner=0)

  else:
    dist_tree = load_size_tree(filename, comm)
    fill_size_tree(dist_tree, filename, comm)

  end = time.time()
  dt_size     = sum(MT.metrics.dtree_nbytes(dist_tree))
  all_dt_size = comm.allreduce(dt_size, MPI.SUM)
  mlog.info(f"Read completed ({end-start:.2f} s) --"
            f" Size of dist_tree for current rank is {mlog.bsize_to_str(dt_size)}"
            f" (Σ={mlog.bsize_to_str(all_dt_size)})")
  return dist_tree

def dist_tree_to_file(dist_tree, filename, comm, links=[]):
  """dist_tree_to_file(dist_tree, filename, comm, links=[])

  Distributed write to a CGNS file.

  If links are used, the link description list must be identiqual on all ranks.

  Args:
    dist_tree (CGNSTree) : Distributed tree to write
    filename (str) : Path of the file
    links   (list) : List of links to create (see SIDS-to-Python guide)
    comm     (MPIComm) : MPI communicator
  """
  if links:
    dist_tree = PT.shallow_copy(dist_tree)
    for link in links: # Links override data, so delete data
      PT.rm_node_from_path(dist_tree, link[3])

  dt_size     = sum(MT.metrics.dtree_nbytes(dist_tree))
  all_dt_size = comm.allreduce(dt_size, MPI.SUM)
  mlog.info(f"Distributed write of a {mlog.bsize_to_str(dt_size)} dist_tree"
            f" (Σ={mlog.bsize_to_str(all_dt_size)})...")
  start = time.time()
  filename = str(filename)

  # Check if folder exists
  create_parent_folder(filename, comm)

  hdf_filter = create_tree_hdf_filter(dist_tree)
  save_tree_from_filter(filename, dist_tree, comm, hdf_filter, links)
  end = time.time()
  mlog.info(f"Write completed [{filename}] ({end-start:.2f} s)")

def write_trees(tree, filename, comm, links=[]):
  """write_trees(tree, filename, comm, links=[])

  Sequential write to CGNS files.

  Write separate trees for each process. Rank id will be automatically
  inserted in the filename. If links are used, each rank must provide its own
  link description list.

  Args:
    tree (CGNSTree) : Tree to write
    filename (str) : Path of the file
    links   (list) : List of links to create (see SIDS-to-Python guide)
    comm     (MPIComm) : MPI communicator

  Example:
      .. literalinclude:: snippets/test_io.py
        :start-after: #write_trees@start
        :end-before: #write_trees@end
        :dedent: 2
  """
  # Give to each process a filename
  create_parent_folder(filename, comm)
  filename = str(filename)
  base_name, extension = os.path.splitext(filename)
  base_name += f"_{comm.Get_rank()}"
  _filename = base_name + extension
  write_tree(tree, _filename, links)
