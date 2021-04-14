import Converter.Internal as I
import numpy as np

import Pypdm.Pypdm as PDM

from maia.sids           import conventions as conv
from maia.utils          import py_utils
from maia.tree_exchange  import utils    as te_utils

def get_pl_donor(dist_zones, part_zones, comm):

  path = 'ZoneGridConnectivity_t/GridConnectivity_t'
  jn_id     = []
  jn_id_opp = []
  for dist_zone in dist_zones:
    for jn in py_utils.getNodesByMatching(dist_zone, path):
      jn_id.append    (I.getNodeFromName1(jn, 'Ordinal')[1][0] - 1)
      jn_id_opp.append(I.getNodeFromName1(jn, 'OrdinalOpp')[1][0] - 1)

  n_jn = len(jn_id)
  n_unique_jn = n_jn // 2

  if n_jn == 0:
    return

  join_to_ref = np.empty(n_jn, dtype=np.int32)
  ref_join_gid = 0
  for i in range(n_jn):
    i_jn     = jn_id[i]
    i_jn_opp = jn_id_opp[i]
    if i_jn < i_jn_opp:
      join_to_ref[i_jn]     = ref_join_gid
      join_to_ref[i_jn_opp] = ref_join_gid
      ref_join_gid += 1

  # Create face join distribution (with unicity)
  nb_face_in_joins = np.zeros(n_unique_jn, np.int32)
  gc_type_path = 'ZoneGridConnectivity_t/GridConnectivity_t'
  for d_zone in dist_zones:
    for gc in py_utils.getNodesFromTypePath(d_zone, gc_type_path):
      gc_id     = I.getNodeFromName1(gc, 'Ordinal'   )[1][0] - 1
      gc_id_opp = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0] - 1
      if (gc_id < gc_id_opp):
        nb_face_in_joins[join_to_ref[gc_id]] = te_utils.get_cgns_distribution(gc, ':CGNS#Distribution/Index')[2]
        
  face_in_join_offset = py_utils.sizes_to_indices(nb_face_in_joins)

  shifted_lntogn = list()
  part_data = {key : [] for key in ['pl', 'irank', 'ipart']}
  for p_zone in part_zones:
    d_zone_name = conv.get_part_prefix(I.getName(p_zone))
    i_proc, i_part = conv.get_part_suffix(I.getName(p_zone))
    for gc in py_utils.getNodesFromTypePath(p_zone, gc_type_path):
      if I.getNodeFromName1(gc, 'Ordinal') is not None: #Skip part joins
        gc_id = I.getNodeFromName1(gc, 'Ordinal')[1][0] - 1
        lngn = I.getNodeFromPath(gc, ':CGNS#GlobalNumbering/Index')[1]
        shifted_lntogn.append(lngn + face_in_join_offset[join_to_ref[gc_id]])
        pl = I.getNodeFromName1(gc, 'PointList')[1][0]
        part_data['pl'].append(pl)
        part_data['irank'].append(i_proc*np.ones(pl.size, dtype=pl.dtype))
        part_data['ipart'].append(i_part*np.ones(pl.size, dtype=pl.dtype))

  PTB = PDM.PartToBlock(comm, shifted_lntogn, pWeight=None, partN=len(shifted_lntogn), 
                        t_distrib=0, t_post=0, t_stride=0)
  distribution = PTB.getDistributionCopy()

  dData = dict()
  PTB.PartToBlock_Exchange(dData, part_data)

  b_stride = 2*np.ones(dData['pl'].size, dtype=np.int32)

  BTP = PDM.BlockToPart(distribution, comm, shifted_lntogn, len(shifted_lntogn))
  part_data = dict()
  BTP.BlockToPart_Exchange2(dData, part_data, 1, b_stride)


  #Post treat
  i_join = 0
  for p_zone in part_zones:
    i_rank, i_part = conv.get_part_suffix(I.getName(p_zone))
    for gc in py_utils.getNodesFromTypePath(p_zone, gc_type_path):
      if I.getNodeFromName1(gc, 'Ordinal') is not None: #Skip part joins
        pl = I.getNodeFromName1(gc, 'PointList')[1][0]
        opp_pl   = np.empty_like(pl)
        opp_rank = np.empty((pl.size,2), order='F', dtype=np.int32)
        #opp_part = np.empty(pl.size, dtype=np.int32)
        for i in range(pl.size):
          if part_data['irank'][i_join][2*i] != i_rank:
            opp_rank[i][0] = part_data['irank'][i_join][2*i]
            opp_rank[i][1] = part_data['ipart'][i_join][2*i]
            opp_pl[i]   = part_data['pl'   ][i_join][2*i]
          elif part_data['irank'][i_join][2*i+1] != i_rank:
            opp_rank[i][0] = part_data['irank'][i_join][2*i+1]
            opp_rank[i][1] = part_data['ipart'][i_join][2*i+1]
            opp_pl[i]   = part_data['pl'   ][i_join][2*i+1]
          # The two joins are on the same proc, look at the parts
          else:
            opp_rank[i][0] = i_rank
            if part_data['ipart'][i_join][2*i] != i_part:
              opp_rank[i][1] = part_data['ipart'][i_join][2*i]
              opp_pl[i]   = part_data['pl'   ][i_join][2*i]
            elif part_data['ipart'][i_join][2*i+1] != i_part:
              opp_rank[i][1] = part_data['ipart'][i_join][2*i+1]
              opp_pl[i]   = part_data['pl'   ][i_join][2*i+1]
            # The two joins have the same proc id / part id, we need to check original pl
            else:
              opp_rank[i][1] = i_part
              if part_data['pl'][i_join][2*i] != pl[i]:
                opp_pl[i] = part_data['pl'][i_join][2*i]
              else:
                opp_pl[i] = part_data['pl'][i_join][2*i+1]

        i_join += 1
        I.newDataArray('PointListDonor', opp_pl.reshape((1,-1), order='F'), parent=gc)
        I.newDataArray('Donor', opp_rank, parent=gc)

