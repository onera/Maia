import Converter.Internal as I
import numpy as np
import Pypdm.Pypdm as PDM

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils     import np_utils
from maia.transfer  import utils    as te_utils

def get_pl_donor(dist_zones, part_zones, comm):

  query = ['ZoneGridConnectivity_t', \
      lambda n : I.getType(n) == 'GridConnectivity_t' and PT.GridConnectivity.is1to1(n)]
  jn_id     = []
  jn_id_opp = []
  for dist_zone in dist_zones:
    for jn in PT.iter_children_from_predicates(dist_zone, query):
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
    for gc in PT.iter_children_from_predicates(d_zone, gc_type_path):
      gc_id     = I.getNodeFromName1(gc, 'Ordinal'   )[1][0] - 1
      gc_id_opp = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0] - 1
      if (gc_id < gc_id_opp):
        nb_face_in_joins[join_to_ref[gc_id]] = te_utils.get_cgns_distribution(gc, 'Index')[2]

  face_in_join_offset = np_utils.sizes_to_indices(nb_face_in_joins)

  shifted_lntogn = list()
  part_data = {key : [] for key in ['pl', 'irank', 'ipart', 'ijoin']}
  part_stride = []
  for p_zone in part_zones:
    d_zone_name = MT.conv.get_part_prefix(I.getName(p_zone))
    i_proc, i_part = MT.conv.get_part_suffix(I.getName(p_zone))
    for gc in PT.iter_children_from_predicates(p_zone, gc_type_path):
      if I.getNodeFromName1(gc, 'Ordinal') is not None: #Skip part joins
        gc_id = I.getNodeFromName1(gc, 'Ordinal')[1][0] - 1
        lngn = I.getVal(MT.getGlobalNumbering(gc, 'Index'))
        shifted_lntogn.append(lngn + face_in_join_offset[join_to_ref[gc_id]])
        pl = I.getNodeFromName1(gc, 'PointList')[1][0]
        part_data['pl'].append(pl)
        part_data['irank'].append(i_proc*np.ones(pl.size, dtype=pl.dtype))
        part_data['ipart'].append(i_part*np.ones(pl.size, dtype=pl.dtype))
        part_data['ijoin'].append( gc_id*np.ones(pl.size, dtype=pl.dtype))
        part_stride.append(np.ones(pl.size, np.int32))

  PTB = PDM.PartToBlock(comm, shifted_lntogn, pWeight=None, partN=len(shifted_lntogn),
                        t_distrib=0, t_post=2)
  distribution = PTB.getDistributionCopy()

  dData = dict()
  PTB.PartToBlock_Exchange(dData, part_data, pStrid=part_stride)

  BTP = PDM.BlockToPart(distribution, comm, shifted_lntogn, len(shifted_lntogn))
  part_data = dict()
  BTP.BlockToPart_Exchange2(dData, part_data, BlkStride=dData['pl#PDM_Stride'])


  #Post treat
  i_join = 0
  for p_zone in part_zones:
    i_rank, i_part = MT.conv.get_part_suffix(I.getName(p_zone))
    for gc in PT.iter_children_from_predicates(p_zone, gc_type_path):
      if I.getNodeFromName1(gc, 'Ordinal') is not None: #Skip part joins
        gc_id = I.getNodeFromName1(gc, 'Ordinal')[1][0] - 1
        pl_node = I.getNodeFromName1(gc, 'PointList')
        lngn_node = MT.getGlobalNumbering(gc, 'Index')
        lngn = lngn_node[1]
        pl = pl_node[1][0]
        r_idx = 0
        ini_size = pl.size
        # First pass to count the number of matchs for each entity
        n_matches = np.zeros(pl.size, np.int32)
        for i in range(ini_size):
          n_candidates = part_data['pl#PDM_Stride'][i_join][i]
          myself_found = False
          for ic in range(n_candidates):
            #For debug, check that myself is in list
            if part_data['irank'][i_join][r_idx + ic] == i_rank and \
               part_data['ipart'][i_join][r_idx + ic] == i_part and \
               part_data['ijoin'][i_join][r_idx + ic] == gc_id  and \
               part_data['pl'   ][i_join][r_idx + ic] == pl[i]:
                 myself_found = True
            # All the received tuples come from one of the two elements 
            # of same original interface entity (because working by interface -- gnum was shifted)
            # We have to select only the ones coming from the opposite side of the interface
            if part_data['ijoin'][i_join][r_idx + ic] != gc_id:
               n_matches[i] += 1
          r_idx += n_candidates
          assert n_matches[i] >= 1
          assert (myself_found)

        # Reallocate arrays
        new_size = n_matches.sum()
        if new_size != ini_size:
          pl     = np.resize(pl, new_size)
          lngn   = np.resize(lngn, new_size)

        opp_rank = np.empty((pl.size,2), order='F', dtype=np.int32)
        opp_pl   = np.empty_like(pl)
        w_idx = ini_size #To write at end of array
        r_idx = 0
        for i in range(ini_size):
          n_candidates = part_data['pl#PDM_Stride'][i_join][i]
          first_match = True
          for ic in range(n_candidates):
            if part_data['ijoin'][i_join][r_idx + ic] != gc_id:
               if first_match: #Register first inplace
                 opp_rank[i][0] = part_data['irank'][i_join][r_idx+ic]
                 opp_rank[i][1] = part_data['ipart'][i_join][r_idx+ic]
                 opp_pl[i]      = part_data['pl'   ][i_join][r_idx+ic]
                 first_match = False
               else: #Register other at end of array
                 pl[w_idx]     = pl[i]
                 lngn[w_idx]   = lngn[i]
                 opp_pl[w_idx]      = part_data['pl'   ][i_join][r_idx+ic]
                 opp_rank[w_idx][0] = part_data['irank'][i_join][r_idx+ic]
                 opp_rank[w_idx][1] = part_data['ipart'][i_join][r_idx+ic]
                 w_idx += 1

          r_idx += n_candidates

        assert w_idx == new_size
        I.newDataArray('PointListDonor', opp_pl.reshape((1,-1), order='F'), parent=gc)
        I.newDataArray('Donor', opp_rank, parent=gc)
        I.setValue(pl_node, pl.reshape((1,-1), order='F'))
        I.setValue(lngn_node, lngn)
        i_join += 1

