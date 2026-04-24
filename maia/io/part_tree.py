import os

import maia
from maia.typing import *
from maia.pytree.typing import Predicate
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.pytree.utils  as PTu
import maia.utils.logging as mlog
from maia.factory.dist_from_part import discover_nodes_from_matching
from maia.factory.partitioning import compute_nosplit_weights
from .cgns_io_tree import _LEGACY_IO
from .cgns_io_tree import write_tree
from .cgns_io_tree import replace_long_names
from .utils        import create_parent_folder

from maia.pytree.node import name_utils as NU

def get_str_value(node:CGNSTree) -> str:
  assert isinstance(value := PT.get_value(node), str)
  return value

if _LEGACY_IO:
  import Converter.Filter as Filter
  from Converter.Distributed import writeZones
else:
  from h5py import h5f
  from ._hdf_io_h5py  import _write_links
  from .hdf._hdf_cgns import open_from_path, load_tree_partial, _load_node_partial, _write_node_partial

def enforce_maia_naming(part_tree: CGNSPartTree, 
                        comm: MPIComm) -> None:
  """Rename the zones and joins of a partitioned tree such that maia
  convention are respected
  """
  old_to_new = {}
  count = {}
  for zone_path in PT.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
    prefix = MT.conv.get_part_prefix(zone_path)
    if not prefix in count:
      count[prefix] = 0
    part_id = count[prefix]
    count[prefix] += 1
    old_to_new[zone_path] = MT.conv.add_part_suffix(prefix, comm.Get_rank(), part_id)

  # Unsplitted joins should not be renamed (since zone prefix does not change) 
  # --> protect them
  is_unsplit_gc:Predicate = lambda n : PT.get_label(n) == 'GridConnectivity_t' and get_str_value(n).endswith('P?.N?')
  gc_predicates = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', is_unsplit_gc]
  unsplit_gcs = PT.get_children_from_predicates(part_tree, gc_predicates)
  for gc in unsplit_gcs:
    PT.set_label(gc, 'UserDefinedData_t')

  MT.rename_zones(part_tree, old_to_new, comm)

  for gc in unsplit_gcs:
    PT.set_label(gc, 'GridConnectivity_t')

  # Update JNs name for internal joins
  is_intra_gc = MT.pred.is_gc_of_kind(is_intra=True)
  gc_predicates = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', is_intra_gc]
  for _, zone, _, gc in PT.get_children_from_predicates(part_tree, gc_predicates, ancestors=True):
    cur_proc, cur_part = MT.conv.get_part_suffix(PT.get_name(zone))
    opp_proc, opp_part = MT.conv.get_part_suffix(get_str_value(gc))
    PT.set_name(gc, MT.conv.name_intra_gc(cur_proc, cur_part, opp_proc, opp_part))
    donor_name = PT.get_node_from_name(gc, 'GridConnectivityDonorName')
    if donor_name is not None:
      PT.set_value(donor_name, MT.conv.name_intra_gc(opp_proc, opp_part, cur_proc, cur_part))


def _read_part_from_name(tree: CGNSTree, 
                         filename: Union[str, PathLike], 
                         comm: MPIComm) -> List[CGNSPath]:
  zones_path = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  max_proc = max([MT.conv.get_part_suffix(path)[0] for path in zones_path]) + 1
  if max_proc != comm.Get_size():
    mlog.error(f"Reading with {comm.Get_size()} procs file {filename} written for {max_proc} procs")
  return [path for path in zones_path if MT.conv.get_part_suffix(path)[0] == comm.Get_rank()]

def _read_part_from_size(tree: CGNSTree, 
                         filename: Union[str, PathLike],
                         comm: MPIComm) -> List[CGNSPath]:
  zones_path = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  max_proc = max([MT.conv.get_part_suffix(path)[0] for path in zones_path]) + 1
  mlog.warning(f"Ignoring procs affectation when reading file {filename} written for {max_proc} procs")
  return [path for path in compute_nosplit_weights(tree, comm)]


def file_to_part_tree(filename: Union[str, PathLike], 
                      comm: MPIComm, 
                      redispatch: bool = False) -> CGNSPartTree:
  """file_to_part_tree(filename, comm, redispatch=False)
  
  Read the partitioned zones from a hdf container and affect them
  to the ranks.
  
  If ``redispatch == False``, the CGNS zones are affected to the
  rank indicated in their name. 
  The size of the MPI communicator must thus be equal to the highest id
  appearing in partitioned zone names.

  If ``redispatch == True``, the CGNS zones are dispatched over the
  available processes, and renamed to follow maia's conventions.

  Important:
    This function **does not** perfom the partitioning operation; input file is supposed
    to contain an already partitioned tree, eg. saved with :func:`part_tree_to_file`.

  Args:
    filename (str) : Path of the file
    comm     (MPIComm) : MPI communicator
    redispatch (bool) : Controls the affectation of the partitions to the available ranks (see above).
      Defaults to False.
  Returns:
    CGNSTree: Partitioned CGNS tree

  """
  # Skeleton
  filename = str(filename)
  if _LEGACY_IO:
    tree = Filter.convertFile2SkeletonTree(filename, maxDepth=2)
  else:
    if comm.Get_rank() == 0:
      dont_load_zone = lambda N, labels, S : labels[-1] == 'Zone_t' or not 'Zone_t' in labels
      size_tree = load_tree_partial(filename, dont_load_zone)
      PT.rm_nodes_from_predicate(size_tree, PT.pred.label_is('DataArray_t') & PT.pred.name_matches('*#Size'))
    else:
      size_tree = None

    tree = comm.bcast(size_tree, root=0)

  if redispatch:
    zones_to_read = _read_part_from_size(tree, filename, comm)
  else:
    zones_to_read = _read_part_from_name(tree, filename, comm)

  # Data
  if _LEGACY_IO:
    to_read = list() #Read owned zones and metadata at Base level
    for zone_path in PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t'):
      if zone_path in zones_to_read:
        to_read.append(zone_path)
    PT.rm_nodes_from_label(tree, 'Zone_t')
    for other_path in PT.predicates_to_paths(tree, 'CGNSBase_t/*'):
      to_read.append(other_path)
    for base in PT.get_children_from_label(tree, 'CGNSBase_t'):
      PT.set_children(base, []) #Remove Base children to append it (with data) after

    nodes =  Filter.readNodesFromPaths(filename, to_read)
    for path, node in zip(to_read, nodes):
      base = PT.find_node_from_path(tree, PTu.path_head(path))
      PT.add_child(base, node)
  else:
    # Remove zones not going to this rank
    for base in PT.get_children_from_label(tree, 'CGNSBase_t'):
      _zones_to_read = [PTu.path_tail(zpath) for zpath in zones_to_read if PTu.path_head(zpath) == PT.get_name(base)]
      PT.rm_children_from_predicate(base, PT.pred.label_is('Zone_t') & ~PT.pred.name_in(_zones_to_read))

    # Now load full data of affected zones
    fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDONLY)
    for base in PT.get_children_from_label(tree, 'CGNSBase_t'):
      zone_names = [PT.get_name(n) for n in PT.get_children_from_label(base, 'Zone_t')]
      PT.rm_children_from_label(base, 'Zone_t')
      for zone_name in zone_names:
        gid = open_from_path(fid, f'{PT.get_name(base)}/{zone_name}')
        _load_node_partial(gid, base, lambda X,Y,s:True, None, ([],[]))
        gid.close()
    fid.close()

  # Remove empty bases
  PT.rm_children_from_predicate(tree, lambda n: PT.get_label(n) == 'CGNSBase_t' \
          and len(PT.get_children_from_label(n, 'Zone_t')) == 0)

  NU.unhash_long_names(tree)

  if redispatch:
    enforce_maia_naming(tree, comm)

  return tree


def part_tree_to_file(part_tree: CGNSPartTree, 
                      filename: Union[str, PathLike], 
                      comm: MPIComm, 
                      single_file: bool = False, 
                      links: List[List[str]] = []) -> None:
  """part_tree_to_file(part_tree, filename, comm, single_file=False, links=[])
  
  Gather the partitioned zones managed by all the processes and write it in a unique
  hdf container.

  If ``single_file`` is True, one file named *filename* storing all the partitioned
  zones is written. Otherwise, hdf links are used to produce a main file *filename*
  linking to additional subfiles.
  
  Args:
    part_tree (CGNSPartTree) : Partitioned tree
    filename (str)           : Path of the output file
    comm     (MPIComm)       : MPI communicator
    single_file (bool)       : Produce a unique file if True; use CGNS links otherwise.
    links (list)             : List of links to create (see SIDS-to-Python guide). Each rank must provide
      only the links related to one of its partitions.

  Example:
      .. literalinclude:: snippets/test_io.py
        :start-after: #save_part_tree@start
        :end-before: #save_part_tree@end
        :dedent: 2
  """
  MT.check_cgns_part_tree(part_tree)
  rank = comm.Get_rank()
  filename = str(filename)
  base_name, extension = os.path.splitext(filename)
  subfilename = base_name + f'_sub_{rank}' + extension

  # Get meta data nodes, this allows custom nodes located at tree top level (see #108)
  glob_nodes = PT.get_children_from_predicate(part_tree, ~PT.pred.label_is('CGNSBase_t'))
  top_tree = PT.new_node('CGNSTree', 'CGNSTree_t', children=glob_nodes)
  # Recover base data and families
  is_not_zone = ~PT.pred.label_is('Zone_t')
  discover_nodes_from_matching(top_tree, [part_tree], 'CGNSBase_t', comm, get_value='all', child_list=[is_not_zone])

  create_parent_folder(filename, comm)

  if single_file:
    # Sequential write seems to be faster than collective io -- see 01d84da7 for other methods
    # Create file and write Bases
    if rank == 0:
      write_tree(top_tree, filename)
    comm.barrier()
    for i in range(comm.Get_size()):
      if i == rank:
        if _LEGACY_IO:
          writeZones(part_tree, filename, proc=-1)
        else:
          tree, links = replace_long_names(part_tree, links)

          fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR)
          for zone_path in maia.pytree.predicates_to_paths(tree, 'CGNSBase_t/Zone_t'):
            zone = PT.find_node_from_path(tree, zone_path)
            gid = open_from_path(fid, zone_path.split('/')[0])
            _write_node_partial(gid, zone, lambda X,Y,s: True, ([],[]))
            gid.close()
          fid.close()
          _write_links(filename, links)
      comm.barrier()

  else:
    zone_links      = []
    for zone_path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
      zone_links += [['', subfilename, zone_path, zone_path]]

    write_tree(part_tree, subfilename, links) #Use direct API to manage name

    _zone_links = comm.gather(zone_links, root=0)
    if rank == 0:
      assert _zone_links is not None #For mypy
      zone_links  = [l for proc_links in _zone_links for l in proc_links] #Flatten gather result
      
      for zone_link in zone_links:
        b_name, z_name = zone_link[3].split('/')
        PT.new_child(PT.find_child_from_name(top_tree, b_name), z_name, 'Zone_t')

      write_tree(top_tree, filename, links=zone_links)

