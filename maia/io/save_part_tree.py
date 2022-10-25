import os
import maia
import maia.pytree        as PT

from maia.factory.dist_from_part import discover_nodes_from_matching

from . import cgns_io_tree as IOT

def _tree_to_pathes(node, cur_path, paths):
  for child in PT.get_children(node):
    cur_path += f"/{PT.get_name(child)}"
    paths.append(cur_path[1:])
    _tree_to_pathes(child, cur_path, paths)
    cur_path = PT.path_head(cur_path)

def tree_to_paths(tree):
  paths = []
  _tree_to_pathes(tree, '', paths)
  return paths

# Experimental

from mpi4py import MPI
import numpy as np
from .hdf._hdf_cgns import open_from_path, write_tree_partial, write_data_partial, _write_node_partial, DTYPE_TO_CGNSTYPE
from h5py import h5a, h5s, h5t, h5d, h5f, h5p, h5fd

def seq_queue_write(top_tree, part_tree, filename, comm):
  # Create file and write Bases
  if comm.Get_rank() == 0:
    IOT.write_tree(top_tree, filename)
  comm.barrier()
  for i in range(comm.Get_size()):
    if i == comm.Get_rank():
      fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR)
      for zone_path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
        gid = open_from_path(fid, zone_path.split('/')[0])
        zone = PT.get_node_from_path(part_tree, zone_path)
        _write_node_partial(gid, zone, lambda X,Y: True)
        gid.close()
      fid.close()
    comm.barrier()

  # This works in sequential but not in //, probably because group
  # opening is a collective operation
  # fapl = h5p.create(h5p.FILE_ACCESS)
  # fapl.set_driver(h5fd.MPIO)
  # fapl.set_fapl_mpio(comm, MPI.Info())
  # fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR, fapl)
  # for zone_path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
    # base_name, zone_name = zone_path.split('/')
    # gid = open_from_path(fid, zone_path)
    # node = PT.get_node_from_path(part_tree, zone_path)
    # for child in node[2]:
      # _write_node_partial(gid, child, lambda X,Y: True)
    # gid.close()
  # fid.close()

def distri_and_write(top_tree, part_tree, filename, comm):
  discover_nodes_from_matching(top_tree, [part_tree], 'CGNSBase_t/Zone_t', comm)
  for base in PT.get_children_from_label(top_tree, 'CGNSBase_t'):
    dist_zones = []
    for zone in PT.get_children_from_label(base, 'Zone_t'):
      root = PT.maia.conv.get_part_suffix(PT.get_name(zone))[0]
      true_zone = None
      if comm.Get_rank() == root:
        true_zone =  PT.get_node_from_path(part_tree, base[0] + '/' + zone[0])
      dist_zones.append(maia.factory.distribute_tree(true_zone, comm, owner=root))
    PT.rm_children_from_label(base, 'Zone_t')
    for zone in dist_zones:
      PT.add_child(base, zone)
  maia.io.dist_tree_to_file(top_tree, filename, comm)


def skeleton_then_data_write(top_tree, part_tree, filename, comm):
  datashape_info = {}
  for base_name in PT.predicates_to_paths(top_tree, 'CGNSBase_t'):
    base = PT.get_child_from_name(part_tree, base_name)
    skel_zones = []
    if base is not None:
      for zone in PT.get_children_from_label(base, 'Zone_t'):
        zone_path = base_name + '/' + PT.get_name(zone)
        skel_zone = PT.shallow_copy(zone)
        for path in tree_to_paths(skel_zone):
          node = PT.get_node_from_path(skel_zone, path)
          if node[1] is not None:
            datashape_info[zone_path + '/' + path] = (node[1].dtype, node[1].shape)
            node[1] = None
        skel_zones.append(skel_zone)

    all_skel_zones = comm.gather(skel_zones, root=0)
    if comm.rank == 0:
      skel_base = PT.get_child_from_name(top_tree, base_name)
      for rank_skel_zones in all_skel_zones:
        for rank_skel_zone in rank_skel_zones:
          PT.add_child(skel_base, rank_skel_zone)

  all_datashape_info = comm.allgather(datashape_info)
  if comm.rank == 0:
    #Write structure
    maia.io.cgns_io_tree.write_tree(top_tree, filename)
    #Now write dataset sizes
    buff_S3 = np.empty(1, '|S3')
    fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR)
    for datashape_info in all_datashape_info:
      for path, info in datashape_info.items():
        node_dtype, node_shape = info
        _node_dtype = np.int8 if node_dtype == 'S1' else node_dtype
        gid = open_from_path(fid, path)

        # Carreful : apparently filled with 0, how to disable it ??
        # plist = h5p.create(h5p.DATASET_CREATE)
        # plist.set_fill_time(h5d.FILL_TIME_NEVER)
        # tab = np.empty(0)
        # plist.set_fill_value(tab)

        space = h5s.create_simple(node_shape[::-1])
        tata = h5d.create(gid, b' data', h5t.py_create(_node_dtype), space)
        #tata = h5d.create(gid, b' data', h5t.py_create(_node_dtype), space, dcpl=plist)
        #plist = tata.get_create_plist()
        # tab = np.array([-99])
        # plist.get_fill_value(tab)
        #print(plist.get_fill_time(), plist.get_alloc_time(), plist.fill_value_defined() == h5d.FILL_VALUE_DEFAULT)
        tata.close()
        # Update attr
        buff_S3[0] = DTYPE_TO_CGNSTYPE[node_dtype.name]
        attr_id = h5a.open(gid, b'type')
        attr_id.write(buff_S3)
        attr_id.close()
        gid.close()
    fid.close()


  comm.barrier()

  # Now write data in //
  fapl = h5p.create(h5p.FILE_ACCESS)
  fapl.set_driver(h5fd.MPIO)
  fapl.set_fapl_mpio(comm, MPI.Info())
  fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR, fapl)

  for datashape_info in all_datashape_info:
    for path, _ in datashape_info.items():
      node = PT.get_node_from_path(part_tree, path)
      gid = open_from_path(fid, path)
      dset_id = h5d.open(gid, b' data')
      if node is not None:
        array_view = node[1].T
        if array_view.dtype == 'S1':
          array_view.dtype = np.int8
        dset_id.write(h5s.ALL, h5s.ALL, array_view)
      dset_id.close()
      gid.close()
  fid.close()

  # Pas ok, tt le monde doit ouvrir
  """
  for zone_path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
    zone = PT.get_node_from_path(part_tree, zone_path)
    paths = tree_to_paths(zone)
    for path in tree_to_paths(zone):
      node = PT.get_node_from_path(zone, path)
      if node[1] is not None:
        gid = open_from_path(fid, zone_path + '/' + path)
        array_view = node[1].T
        if array_view.dtype == 'S1':
          array_view.dtype = np.int8
        #dset_id = h5d.open(gid, b' data')
        #dset_id.write(h5s.ALL, h5s.ALL, array_view)
        #dset_id.close()
        gid.close()
  """

# Variante : on échange le squelette puis on crée les noeuds en mode collectif 
# pour passer dans le write_data_partial
#
def skeleton_then_partial_data_write(top_tree, part_tree, filename, comm):
  datashape_info = {}
  for base_name in PT.predicates_to_paths(top_tree, 'CGNSBase_t'):
    base = PT.get_child_from_name(part_tree, base_name)
    skel_zones = []
    if base is not None:
      for zone in PT.get_children_from_label(base, 'Zone_t'):
        zone_path = base_name + '/' + PT.get_name(zone)
        skel_zone = PT.shallow_copy(zone)
        for path in tree_to_paths(skel_zone):
          node = PT.get_node_from_path(skel_zone, path)
          if node[1] is not None:
            datashape_info[zone_path + '/' + path] = (node[1].dtype, node[1].shape)
            #node[1] = None
            node[1] = np.empty(0, dtype=node[1].dtype)
        skel_zones.append(skel_zone)

    all_skel_zones = comm.gather(skel_zones, root=0)
    if comm.rank == 0:
      skel_base = PT.get_child_from_name(top_tree, base_name)
      for rank_skel_zones in all_skel_zones:
        for rank_skel_zone in rank_skel_zones:
          PT.add_child(skel_base, rank_skel_zone)

  all_datashape_info = comm.allgather(datashape_info)
  if comm.rank == 0:
    #Write structure
    #maia.io.cgns_io_tree.write_tree(top_tree, filename)
    write_tree_partial(top_tree, filename, \
        lambda P,N: P[1] in ['Root Node of HDF5 File', 'CGNSBase_t', 'Family_t'])

  comm.barrier()

  fapl = h5p.create(h5p.FILE_ACCESS)
  fapl.set_driver(h5fd.MPIO)
  fapl.set_fapl_mpio(comm, MPI.Info())
  fid = h5f.open(bytes(filename, 'utf-8'), h5f.ACC_RDWR, fapl)

  for datashape_info in all_datashape_info:
    for path, shape in datashape_info.items():
      node = PT.get_node_from_path(part_tree, path)

      zeros = [0 for s in shape[1]]
      ones  = [1 for s in shape[1]]
      glob  = [s for s in shape[1]]
      if node is not None:
        array = node[1]
        filter = [zeros, ones, glob, ones, zeros, ones, glob, ones, glob, [0]]
      else:
        array = np.empty(zeros, dtype=shape[0])
        filter = [zeros, ones, zeros, ones, zeros, ones, zeros, ones, glob, [0]]

      gid = open_from_path(fid, path)
      write_data_partial(gid, array, filter)

      gid.close()
  fid.close()




def save_part_tree(part_tree, filename, comm, single_file=False, legacy=False, method=0):
  """
  Gather the partitioned zones holded by all the processes and write it in a unique
  hdf container.
  If single_file is True, one file named "filename" storing all the partitions is written.
  Otherwise, hdf links are used to produce a main file "filename" linking to additional subfiles.
  """
  base_name, extension = os.path.splitext(filename)
  subfilename = base_name + f'_sub_{comm.Get_rank()}' + extension

  # Recover base data and families
  top_tree = PT.new_CGNSTree()
  discover_nodes_from_matching(top_tree, [part_tree], 'CGNSBase_t', comm, get_value='all', child_list=['Family_t'])

  if single_file:

    if method == 0:
      seq_queue_write(top_tree, part_tree, filename, comm)

    # This works, but lot of exchange (including gather) are done
    if method == 1:
      distri_and_write(top_tree, part_tree, filename, comm)

    # Last approach : rebuild skeleton + write_partial
    if method == 2:
      skeleton_then_data_write(top_tree, part_tree, filename, comm)
    if method == 3:
      skeleton_then_partial_data_write(top_tree, part_tree, filename, comm)
    

  else:
    links      = []
    for zone_path in maia.pytree.predicates_to_paths(part_tree, 'CGNSBase_t/Zone_t'):
      links += [['',subfilename, zone_path, zone_path]]

    IOT.write_tree(part_tree, subfilename, legacy=legacy) #Use direct API to manage name

    _links = comm.gather(links, root=0)
    if(comm.Get_rank() == 0):
      links  = [l for proc_links in _links for l in proc_links] #Flatten gather result
      IOT.write_tree(top_tree, filename, links=links, legacy=legacy)

