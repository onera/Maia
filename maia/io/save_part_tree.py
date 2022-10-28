import os
import maia
import maia.pytree        as PT

from maia.factory.dist_from_part import discover_nodes_from_matching

from .cgns_io_tree import write_tree

def save_part_tree(part_tree, filename, comm, single_file=False, legacy=False):
  """
  Gather the partitioned zones holded by all the processes and write it in a unique
  hdf container.
  If single_file is True, one file named "filename" storing all the partitions is written.
  Otherwise, hdf links are used to produce a main file "filename" linking to additional subfiles.
  """
  rank = comm.Get_rank()
  base_name, extension = os.path.splitext(filename)
  subfilename = base_name + f'_sub_{rank}' + extension

  # Recover base data and families
  top_tree = PT.new_CGNSTree()
  discover_nodes_from_matching(top_tree, [part_tree], 'CGNSBase_t', comm, get_value='all', child_list=['Family_t'])

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
            _write_node_partial(gid, zone, lambda X,Y: True)
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

