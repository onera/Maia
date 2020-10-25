from   mpi4py             import MPI
import Converter.PyTree   as     C
import Converter.Internal as     I


def add_sizes_to_bcdataset_tree(bc, bc_path, size_data):
  """
  """
  for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
    pl_n = I.getNodeFromName1(bcds, 'PointList')
    if pl_n:
      pl_path = bc_path+"/PointList"
      I.newIndexArray('PointList#Size', value=size_data[pl_path][2], parent=bc)


def add_sizes_to_zone_tree(zone, zone_path, size_data):
  """
  """
  for elmt in I.getNodesFromType1(zone, 'Elements_t'):
    elmt_path = zone_path+"/"+elmt[0]
    ec_path   = elmt_path+"/ElementConnectivity"
    I.newIndexArray('ElementConnectivity#Size', value=size_data[ec_path][2], parent=elmt)

  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]
      pl_n = I.getNodeFromName1(bc, 'PointList')
      if pl_n:
        pl_path = bc_path+"/PointList"
        I.newIndexArray('PointList#Size', value=size_data[pl_path][2], parent=bc)
      add_sizes_to_bcdataset_tree(bc, bc_path, size_data)

  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+zone_gc[0]
    gcs = I.getNodesFromType1(zone_gc, 'GridConnectivity_t') + I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t')
    for gc in gcs:
      gc_path = zone_gc_path+"/"+gc[0]
      pl_n = I.getNodeFromName1(gc, 'PointList')
      if pl_n:
        pl_path = gc_path+"/PointList"
        I.newIndexArray('PointList#Size', value=size_data[pl_path][2], parent=gc)
      pld_n = I.getNodeFromName1(gc, 'PointListDonor')
      if pl_n:
        pld_path = gc_path+"/PointListDonor"
        I.newIndexArray('PointListDonor#Size', value=size_data[pld_path][2], parent=gc)

  for zone_subregion in I.getNodesFromType1(zone, 'ZoneSubRegion_t'):
    zone_sub_region_path = zone_path+"/"+zone_subregion[0]
    pl_n = I.getNodeFromName1(bc, 'PointList')
    if pl_n:
      pl_path = zone_sub_region_path+"/PointList"
      I.newIndexArray('PointList#Size', value=size_data[pl_path][2], parent=zone_subregion)

def add_sizes_to_tree(size_tree, size_data):
  """
  """
  for base in I.getNodesFromType1(size_tree, 'CGNSBase_t'):
    base_path = '/'+base[0]
    for zone in I.getZones(base):
      zone_path = base_path+"/"+zone[0]
      add_sizes_to_zone_tree(zone, zone_path, size_data)


def load_collective_size_tree(filename, comm):
  """
    Load on all ranks a "size tree"
    a size tree is a partial tree that contains only the data needed to distribute the tree:
      nb of nodes, nb of elements, size of bcs and gcs...
    Convention:
      when we load the dimensions of an array "MyArray" without loading the array,
      then the dimensions are kept in a "MyArray#Size" node,
      at the same level as the array node would be
  """
  skeleton_depth  = 7
  skeleton_n_data = 3

  # In order to avoid filesystem overload only 1 proc reads the squeleton, then we broadcast
  if(comm.Get_rank() == 0):
    size_data = dict()
    size_tree = C.convertFile2PyTree(filename,
                                     skeletonData=[skeleton_n_data, skeleton_depth],
                                     dataShape=size_data,
                                     format='bin_hdf')
    add_sizes_to_tree(size_tree, size_data)
  else:
    size_tree = None

  size_tree = comm.bcast(size_tree, root=0)

  return size_tree

