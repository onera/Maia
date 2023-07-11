import os
import maia
import maia.pytree        as PT
import maia.pytree.maia   as MT

import maia.utils.logging as mlog
from maia.factory.dist_from_part import discover_nodes_from_matching
from maia.factory.partitioning import compute_nosplit_weights

from .cgns_io_tree import write_tree

def enforce_maia_naming(part_tree, comm):
  """Rename the zones and joins of a partitionned tree such that maia
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

  MT.rename_zones(part_tree, old_to_new, comm)

  # Update JNs name for internal joins
  is_intra_gc = lambda n : PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
                 and MT.conv.is_intra_gc(PT.get_name(n))
  gc_predicates = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', is_intra_gc]
  for _, zone, _, gc in PT.get_children_from_predicates(part_tree, gc_predicates, ancestors=True):
    cur_proc, cur_part = MT.conv.get_part_suffix(PT.get_name(zone))
    opp_proc, opp_part = MT.conv.get_part_suffix(PT.get_value(gc))
    PT.set_name(gc, MT.conv.name_intra_gc(cur_proc, cur_part, opp_proc, opp_part))
    donor_name = PT.get_node_from_name(gc, 'GridConnectivityDonorName')
    if donor_name is not None:
      PT.set_value(donor_name, MT.conv.name_intra_gc(opp_proc, opp_part, cur_proc, cur_part))


def _read_part_from_name(tree, filename, comm):
  zones_path = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  max_proc = max([PT.maia.conv.get_part_suffix(path)[0] for path in zones_path]) + 1
  if max_proc != comm.Get_size():
    mlog.error(f"Reading with {comm.Get_size()} procs file {filename} written for {max_proc} procs")
  return [path for path in zones_path if PT.maia.conv.get_part_suffix(path)[0] == comm.Get_rank()]

def _read_part_from_size(tree, filename, comm):
  zones_path = PT.predicates_to_paths(tree, 'CGNSBase_t/Zone_t')
  max_proc = max([PT.maia.conv.get_part_suffix(path)[0] for path in zones_path]) + 1
  mlog.warning(f"Ignoring procs affectation when reading file {filename} written for {max_proc} procs")
  return [path for path in compute_nosplit_weights(tree, comm)]


def read_part_tree(filename, comm, redispatch=False, legacy=False):
  """Read the partitioned zones from a hdf container and affect them
  to the ranks.
  
  If ``redispatch == False``, the CGNS zones are affected to the
  rank indicated in their name. 
  The size of the MPI communicator must thus be equal to the highest id
  appearing in partitioned zone names.

  If ``redispatch == True``, the CGNS zones are dispatched over the
  available processes, and renamed to follow maia's conventions.

  Args:
    filename (str) : Path of the file
    comm     (MPIComm) : MPI communicator
    redispatch (bool) : Controls the affectation of the partitions to the available ranks (see above).
      Defaults to False.
  Returns:
    CGNSTree: Partitioned CGNS tree

  """

  # Skeleton
  if legacy:
    import Converter.Filter as Filter
    tree = Filter.convertFile2SkeletonTree(filename, maxDepth=2)
  else:
    from maia.io.cgns_io_tree import load_collective_size_tree
    from h5py import h5f
    from .hdf._hdf_cgns import open_from_path, _load_node_partial
    tree = load_collective_size_tree(filename, comm)

  if redispatch:
    zones_to_read = _read_part_from_size(tree, filename, comm)
  else:
    zones_to_read = _read_part_from_name(tree, filename, comm)

  # Data
  if legacy:
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
      base = PT.get_node_from_path(tree, PT.path_head(path))
      PT.add_child(base, node)
  else:
    # Remove zones not going to this rank
    for base in PT.get_children_from_label(tree, 'CGNSBase_t'):
      _zones_to_read = [PT.path_tail(zpath) for zpath in zones_to_read if PT.path_head(zpath) == PT.get_name(base)]
      PT.rm_children_from_predicate(base, lambda n: PT.get_label(n) == 'Zone_t' and PT.get_name(n) not in _zones_to_read)

    # Now load full data of affected zones
    fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDONLY)
    for base in PT.get_children_from_label(tree, 'CGNSBase_t'):
      zone_names = [PT.get_name(n) for n in PT.get_children_from_label(base, 'Zone_t')]
      PT.rm_children_from_label(base, 'Zone_t')
      for zone_name in zone_names:
        gid = open_from_path(fid, f'{PT.get_name(base)}/{zone_name}')
        _load_node_partial(gid, base, lambda X,Y:True, ([],[]))
        gid.close()
    fid.close()

  # Remove empty bases
  PT.rm_children_from_predicate(tree, lambda n: PT.get_label(n) == 'CGNSBase_t' \
          and len(PT.get_children_from_label(n, 'Zone_t')) == 0)

  if redispatch:
    enforce_maia_naming(tree, comm)

  return tree


def save_part_tree(part_tree, filename, comm, single_file=False, legacy=False):
  """Gather the partitioned zones managed by all the processes and write it in a unique
  hdf container.

  If ``single_file`` is True, one file named *filename* storing all the partitioned
  zones is written.  Otherwise, hdf links are used to produce a main file *filename*
  linking to additional subfiles.
  
  Args:
    part_tree (CGNSTree) : Partitioned tree
    filename (str) : Path of the output file
    comm     (MPIComm) : MPI communicator
    single_file (bool) : Produce a unique file if True; use CGNS links otherwise.

  Example:
      .. literalinclude:: snippets/test_io.py
        :start-after: #save_part_tree@start
        :end-before: #save_part_tree@end
        :dedent: 2
  """
  rank = comm.Get_rank()
  base_name, extension = os.path.splitext(filename)
  subfilename = base_name + f'_sub_{rank}' + extension

  # Recover base data and families
  top_tree = PT.new_CGNSTree()
  discover_nodes_from_matching(top_tree, [part_tree], 'CGNSBase_t', comm, get_value='all', child_list=['Family_t', 'ReferenceState_t'])

  if single_file:
    # Sequential write seems to be faster than collective io -- see 01d84da7 for other methods
    # Create file and write Bases
    if rank == 0:
      write_tree(top_tree, filename, legacy=legacy)
    comm.barrier()
    for i in range(comm.Get_size()):
      if i == rank:
        if legacy:
          from Converter.Distributed import writeZones
          writeZones(part_tree, filename, proc=-1)
        else:
          from h5py import h5f
          from .hdf._hdf_cgns import open_from_path, _write_node_partial
          fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR)
          for zone_path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
            zone = PT.get_node_from_path(part_tree, zone_path)
            gid = open_from_path(fid, zone_path.split('/')[0])
            _write_node_partial(gid, zone, lambda X,Y: True, ([],[]))
            gid.close()
          fid.close()
      comm.barrier()

  else:
    links      = []
    for zone_path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
      links += [['', subfilename, zone_path, zone_path]]

    write_tree(part_tree, subfilename, legacy=legacy) #Use direct API to manage name

    _links = comm.gather(links, root=0)
    if rank == 0:
      links  = [l for proc_links in _links for l in proc_links] #Flatten gather result
      write_tree(top_tree, filename, links=links, legacy=legacy)

