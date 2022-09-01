import Converter.Internal as I
import numpy as np
import Pypdm.Pypdm as PDM

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils     import np_utils
from maia.transfer  import utils    as te_utils
from maia.algo.dist import matching_jns_tools as MJT

def get_pl_donor(dist_tree, part_tree, comm):

  _dist_jn_pairs = MJT.get_matching_jns(dist_tree)
  # Skip joins w/ PL (structured)
  dist_jn_pairs = [pathes for pathes in _dist_jn_pairs \
      if I.getNodeFromPath(dist_tree, pathes[0] + '/PointList') is not None]
  if len(dist_jn_pairs) == 0:
    return

  join_to_ref = {}
  nb_face_in_joins = np.empty(len(dist_jn_pairs), np.int32)
  for i, pathes in enumerate(dist_jn_pairs):
    join_to_ref[pathes[0]] = i
    join_to_ref[pathes[1]] = i
    gc = I.getNodeFromPath(dist_tree, pathes[0])
    nb_face_in_joins[i] = te_utils.get_cgns_distribution(gc, 'Index')[2]

  face_in_join_offset = np_utils.sizes_to_indices(nb_face_in_joins)

  ini_gc_query = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', 
      lambda n : I.getType(n) == 'GridConnectivity_t' and PT.GridConnectivity.is1to1(n) and not MT.conv.is_intra_gc(n[0])]


  shifted_lntogn = list()
  part_data = {key : [] for key in ['pl', 'irank', 'ipart', 'ijoin']}
  part_stride = []
  for p_gc_path in PT.predicates_to_paths(part_tree, ini_gc_query):
    d_gc_path = PT.update_path_elt(p_gc_path, 1, lambda name: MT.conv.get_part_prefix(name))
    p_zone_name = p_gc_path.split('/')[1]
    i_proc, i_part = MT.conv.get_part_suffix(p_zone_name)

    itrf_id = join_to_ref[d_gc_path]
    gc_id = 2*itrf_id + int(d_gc_path < MJT.get_jn_donor_path(dist_tree, d_gc_path))

    gc = I.getNodeFromPath(part_tree, p_gc_path)
    lngn = I.getVal(MT.getGlobalNumbering(gc, 'Index'))
    shifted_lntogn.append(lngn + face_in_join_offset[itrf_id])
    pl = PT.get_child_from_name(gc, 'PointList')[1][0]
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
  for p_gc_path in PT.predicates_to_paths(part_tree, ini_gc_query):
    d_gc_path = PT.update_path_elt(p_gc_path, 1, lambda name: MT.conv.get_part_prefix(name))
    p_zone_name = p_gc_path.split('/')[1]
    i_rank, i_part = MT.conv.get_part_suffix(p_zone_name)

    itrf_id = join_to_ref[d_gc_path]
    gc_id = 2*itrf_id + int(d_gc_path < MJT.get_jn_donor_path(dist_tree, d_gc_path))

    gc = I.getNodeFromPath(part_tree, p_gc_path)
    pl_node = PT.get_child_from_name(gc, 'PointList')
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

