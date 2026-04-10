_LEGACY_IO  = False
import os
import time
import mpi4py.MPI as MPI
import numpy as np

from maia.typing import *
import maia.pytree        as PT
import maia.pytree.maia   as MT
from   maia.pytree.maia   import metrics
import maia.utils.logging as mlog

from .distribution_tree         import add_distribution_info, clean_distribution_info
from .hdf.tree                  import create_tree_hdf_filter
from .fix_tree                  import ensure_PE_global_indexing, ensure_signed_nface_connectivity
from .utils                     import create_parent_folder
import hashlib

if _LEGACY_IO:
  from . import _hdf_io_cass as _hdf_io
else:
  from . import _hdf_io_h5py as _hdf_io #type:ignore[no-redef]

FULL_NAME_NODE_NAME = 'FullNameLongerThan32CharsLimit'

from maia.factory     import full_to_dist

def recompute_ec_size(tree, comm):
  # In write mode, retrieve ElementConnectivity#Size to feed hdf dataspaces
  pred = PT.pred.label_is('Elements_t') & PT.pred.has_child_of_name('ElementStartOffset')
  elts = PT.get_children_from_predicates(tree, ['CGNSBase_t', 'Zone_t', pred])
  sizes = np.array([PT.get_np_value(PT.find_child_from_name(e, 'ElementConnectivity')).size for e in elts])
  gsizes = np.zeros_like(sizes)
  comm.Allreduce(sizes, gsizes)
  for e,s in zip(elts, gsizes):
    PT.new_DataArray('ElementConnectivity#Size', s, parent=e)

def load_size_tree(filename: Union[str, PathLike], 
                   comm: MPIComm) -> CGNSTree:
  return _hdf_io.load_size_tree(str(filename), comm)

def load_partial(filename: str, 
                 dist_tree: CGNSTree, 
                 hdf_filter: Dict[str, Any], 
                 comm: MPIComm) -> None:
  if _LEGACY_IO:
    _hdf_io.load_partial(filename, dist_tree, hdf_filter, comm)
  else:
    _hdf_io.load_partial(filename, dist_tree, hdf_filter) #type:ignore[call-arg] #(signature mismatch)

def write_tree(tree: CGNSTree, 
               filename: Union[str, PathLike],
               links: List[List[str]] = []) -> None:
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

def read_tree(filename: Union[str, PathLike]) -> CGNSTree:
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

def read_links(filename: Union[str, PathLike]) -> List[List[str]]:
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

  assert not _LEGACY_IO, "Not implemented for legacy IO"
  return _hdf_io.read_links(filename) #type:ignore[attr-defined] #(Only av. on h5py io)

def load_tree_from_filter(filename: str,
                          dist_tree: CGNSTree, 
                          comm: MPIComm, 
                          hdf_filter: Dict[str, Any]) -> None:
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
    next_hdf_filter:Dict[str, Any] = dict()
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

def _unambiguous_short_names(names):
  """ Find shorter names that are:
  - less than 32 chars
  - unambiguous (two different original names should have two different short names)
  - human-readable as much as possible """
  if len(set(names)) < len(names):
    raise RuntimeError(f"There are two siblings of the same name among {names}")
  short_names = [PT.node.short_name(n) for n in names]

  perm = np.argsort(short_names)
  names       = np.array(names      )[perm]
  short_names = np.array(short_names)[perm]

  group_idces = np.unique(short_names, return_index=True)[1][1:]
  names       = np.split(names      , group_idces)
  short_names = np.split(short_names, group_idces)

  unamb_short_names = []
  for name_group, short_name_group in zip(names, short_names):
    if len(name_group) == 1: # no ambiguity: use the short name (and make sure it is at most 32 chars long)
      short_name = short_name_group[0]
      unamb_short_names.append(short_name[:32])
    else: # several short names are equal: complete with a 8-char hash
      for name, short_name in zip(name_group, short_name_group):
        name =  name.encode('ascii')
        hash = hashlib.sha256(name).hexdigest()[:8]
        unamb_short_names.append(short_name[:24]+hash)

  inv_perm = np.empty_like(perm)
  inv_perm[perm] = np.arange(perm.size)
  return list(np.array(unamb_short_names)[inv_perm])

def _create_full_name_children(tree):
  def _create_full_name_child(node):
    name_children = _unambiguous_short_names([child[0] for child in node[2]])
    for child,name_child in zip(node[2],name_children):
      if len(child[0]) > 32:
        full_name_node = PT.new_UserDefinedData(FULL_NAME_NODE_NAME, child[0])
        PT.get_children(child).append(full_name_node)
        child[0] = name_child
  PT.scan(tree, _create_full_name_child)

def save_tree_from_filter(filename: str,
                          saving_dist_tree: CGNSDistTree, 
                          comm: MPIComm, 
                          hdf_filter: Dict[str, Any], 
                          links: List[List[str]]) -> None:
  """
  """
  hdf_filter_with_dim  = {key: value for (key, value) in hdf_filter.items() if isinstance(value, list)}
  hdf_filter_with_func = {key: value for (key, value) in hdf_filter.items() if not isinstance(value, list)}

  next_hdf_filter:Dict[str, Any] = dict()
  for key, f in hdf_filter_with_func.items():
    f(hdf_filter_with_dim)

  clean_distribution_info(saving_dist_tree)

  _hdf_io.write_partial(filename, saving_dist_tree, hdf_filter_with_dim, links, comm)

def fill_size_tree(tree: CGNSTree, 
                   filename: Union[str, PathLike], 
                   comm: MPIComm) -> None:
  filename = str(filename)
  add_distribution_info(tree, comm)
  hdf_filter = create_tree_hdf_filter(tree)
  # Coords#Size appears in dict -> remove it
  hdf_filter = {key:val for key,val in hdf_filter.items() if not key.endswith('#Size')}

  load_tree_from_filter(filename, tree, comm, hdf_filter)

  n_shifted = ensure_PE_global_indexing(tree)
  if n_shifted > 0 and comm.Get_rank() == 0:
    mlog.error(f"ParentElements arrays of NGON_n elements have been recomputed "\
               f"because they were wrongly defined (local indexing)")
  n_shifted = ensure_signed_nface_connectivity(tree, comm)
  if n_shifted > 0 and comm.Get_rank() == 0:
    mlog.error(f"ElementConnectivity arrays of NFACE_n elements have been recomputed "\
               f"because they were wrongly defined (missing orientations)")

  PT.rm_nodes_from_name(tree, '*#Size')

def _replace_with_full_names(dist_tree):
  def _replace_with_full_name(node):
    if full_name_node := PT.get_child_from_name(node, FULL_NAME_NODE_NAME):
      node[0] = PT.get_value(full_name_node)
      PT.rm_children_from_name(node, FULL_NAME_NODE_NAME)
  PT.scan(dist_tree, _replace_with_full_name)

def file_to_dist_tree(filename: Union[str, PathLike], comm: MPIComm, handle_long_names: bool = True) -> CGNSDistTree:
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
    size_tree = load_size_tree(filename, comm)
    fill_size_tree(size_tree, filename, comm)
    dist_tree = CGNSDistTree(size_tree)
    if handle_long_names:
      _replace_with_full_names(dist_tree)

  end = time.time()
  dt_size     = sum(metrics.dtree_nbytes(dist_tree))
  all_dt_size = comm.allreduce(dt_size, MPI.SUM)
  mlog.info(f"Read completed ({end-start:.2f} s) --"
            f" Size of dist_tree for current rank is {mlog.bsize_to_str(dt_size)}"
            f" (Σ={mlog.bsize_to_str(all_dt_size)})")
  return dist_tree

def dist_tree_to_file(dist_tree: CGNSDistTree, 
                      filename: Union[str, PathLike], 
                      comm: MPIComm, 
                      links: List[List[str]] = []) -> None:
  """dist_tree_to_file(dist_tree, filename, comm, links=[])

  Distributed write to a CGNS file.

  If links are used, the link description list must be identiqual on all ranks.

  Args:
    dist_tree (CGNSDistTree) : Distributed tree to write
    filename (str)           : Path of the file
    links   (list)           : List of links to create (see SIDS-to-Python guide)
    comm     (MPIComm)       : MPI communicator
  """
  MT.check_cgns_dist_tree(dist_tree)

  # work on a copy that we may alter for our specific needs
  saving_dist_tree = PT.shallow_copy(dist_tree)
  _create_full_name_children(saving_dist_tree)

  if links:
    for link in links: # Links override data, so delete data
      PT.rm_node_from_path(saving_dist_tree, link[3])

  dt_size     = sum(metrics.dtree_nbytes(saving_dist_tree))
  all_dt_size = comm.allreduce(dt_size, MPI.SUM)
  mlog.info(f"Distributed write of a {mlog.bsize_to_str(dt_size)} dist_tree"
            f" (Σ={mlog.bsize_to_str(all_dt_size)})...")
  start = time.time()
  filename = str(filename)

  create_parent_folder(filename, comm)

  recompute_ec_size(saving_dist_tree, comm)
  hdf_filter = create_tree_hdf_filter(saving_dist_tree, mode='write')
  save_tree_from_filter(filename, saving_dist_tree, comm, hdf_filter, links)
  end = time.time()
  mlog.info(f"Write completed [{filename}] ({end-start:.2f} s)")

def write_trees(tree: CGNSTree, 
                filename: Union[str, PathLike], 
                comm: MPIComm, 
                links: List[List[str]] = []) -> None:
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
