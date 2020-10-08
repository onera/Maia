from   mpi4py             import MPI
import Converter.PyTree   as     C
import Converter.Internal as     I

def warn_up_bcdataset_dist_tree(bc, bc_path, data_shape):
  """
  """
  for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
    pl_n = I.getNodeFromName1(bcds, 'PointList')
    if(pl_n):
      pl_path = bc_path+"/PointList"
      I.newDataArray('PointList#Shape', value=data_shape[pl_path][2], parent=bc)


def warm_up_zone_dist_tree(zone, zone_path, data_shape):
  """
  """
  for zone_bc in I.getNodesFromType1(zone, 'ZoneBC_t'):
    zone_bc_path = zone_path+"/"+zone_bc[0]
    for bc in I.getNodesFromType1(zone_bc, 'BC_t'):
      bc_path = zone_bc_path+"/"+bc[0]
      pl_n = I.getNodeFromName1(bc, 'PointList')
      if(pl_n):
        pl_path = bc_path+"/PointList"
        I.newDataArray('PointList#Shape', value=data_shape[pl_path][2], parent=bc)
      warn_up_bcdataset_dist_tree(bc, bc_path, data_shape)

  for zone_gc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
    zone_gc_path = zone_path+"/"+zone_gc[0]
    gcs = I.getNodesFromType1(zone_gc, 'GridConnectivity_t') + I.getNodesFromType1(zone_gc, 'GridConnectivity1to1_t')
    for gc in gcs:
      gc_path = zone_gc_path+"/"+gc[0]
      pl_n = I.getNodeFromName1(gc, 'PointList')
      if(pl_n):
        pl_path = gc_path+"/PointList"
        I.newDataArray('PointList#Shape', value=data_shape[pl_path][2], parent=gc)
      pld_n = I.getNodeFromName1(gc, 'PointListDonor')
      if(pl_n):
        pld_path = gc_path+"/PointListDonor"
        I.newDataArray('PointListDonor#Shape', value=data_shape[pld_path][2], parent=gc)
      warn_up_bcdataset_dist_tree(gc, gc_path, data_shape)


def warm_up_dist_tree(dist_tree, data_shape):
  """
  """
  for base in I.getNodesFromType1(dist_tree, 'CGNSBase_t'):
    base_path = '/'+base[0]
    for zone in I.getZones(base):
      zone_path = base_path+"/"+zone[0]
      warm_up_zone_dist_tree(zone, zone_path, data_shape)


def load_collective_pruned_tree(filename, comm):
  """
  """
  rank = comm.Get_rank()
  size = comm.Get_size()

  skeleton_depth  = 7
  skeleton_n_data = 3

  # > In order to avoid filesystem overload only 1 proc read the squeleton
  if(rank == 0):
    data_shape = dict()
    dist_tree  = C.convertFile2PyTree(filename,
                                      skeletonData=[skeleton_n_data, skeleton_depth],
                                      dataShape=data_shape,
                                      format='bin_hdf')

    # > Well we warn_up the with data_shape before send
    warm_up_dist_tree(dist_tree, data_shape)
  else:
    dist_tree = None

  dist_tree = comm.bcast( dist_tree, root=0)

  return dist_tree

