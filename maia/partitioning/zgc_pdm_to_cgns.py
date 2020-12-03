import Converter.Internal as I
import numpy              as NPY

def zgc_original_pdm_to_cgns(zone, dist_zone, comm):
  """
  Already exist in initial configuration
  """

def zgc_created_pdm_to_cgns(zone, dist_zone, comm, entity='face', zgc_name='ZoneGridConnectivity'):
  """
  Create by splitting
  """
  if(entity == 'face'):
    grid_loc = 'FaceCenter'
  elif(entity == 'vtx'):
    grid_loc = 'Vertex'
  else:
    raise NotImplementedError("Unvalid specified entity")

  dist_zone_name = dist_zone[0]

  ppart_ud                   = I.getNodeFromName1(zone, ':CGNS#Ppart')
  ipart                      = I.getNodeFromName1(ppart_ud, 'ipart')[1][0]
  entity_part_bound_proc_idx = I.getNodeFromName1(ppart_ud, 'np_{0}_part_bound_proc_idx'.format(entity))[1]
  entity_part_bound_part_idx = I.getNodeFromName1(ppart_ud, 'np_{0}_part_bound_part_idx'.format(entity))[1]
  entity_part_bound_tmp      = I.getNodeFromName1(ppart_ud, 'np_{0}_part_bound'         .format(entity))[1]
  print("entity_part_bound_proc_idx= ",entity_part_bound_proc_idx)
  print("entity_part_bound_part_idx= ",entity_part_bound_part_idx)
  print("entity_part_bound_tmp= ",entity_part_bound_tmp)

  entity_part_bound = entity_part_bound_tmp.reshape((4, entity_part_bound_tmp.shape[0]//4), order='F')
  entity_part_bound = entity_part_bound.transpose()

  zgc_n = I.newZoneGridConnectivity(name=zgc_name, parent=zone)

  n_internal_join = entity_part_bound_part_idx.shape[0]-1
  for i_join in range(n_internal_join):

    beg_pl = entity_part_bound_part_idx[i_join  ]
    end_pl = entity_part_bound_part_idx[i_join+1]

    if( beg_pl != end_pl):

      pl_size = end_pl - beg_pl
      pl      = NPY.empty((1, pl_size), order='F', dtype=NPY.int32)
      pl[0]   = NPY.copy(entity_part_bound[beg_pl:end_pl, 0])

      pld    = NPY.empty((1, pl_size), order='F', dtype=NPY.int32)
      pld[0] = NPY.copy(entity_part_bound[beg_pl:end_pl, 3])

      connect_proc = entity_part_bound[beg_pl, 1]
      connect_part = entity_part_bound[beg_pl, 2]-1

      join_n = I.newGridConnectivity(name      = 'JN.P{0}.N{1}.LT.P{2}.N{3}'.format(comm.Get_rank(), ipart, connect_proc, connect_part, i_join),
                                     donorName = dist_zone_name+'.P{0}.N{1}'.format(connect_proc, connect_part),
                                     ctype     = 'Abutting1to1',
                                     parent    = zgc_n)

      I.newGridLocation(grid_loc, parent=join_n)
      I.newPointList(name='PointList'     , value=pl , parent=join_n)
      I.newPointList(name='PointListDonor', value=pld, parent=join_n)
