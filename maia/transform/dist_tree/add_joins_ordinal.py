import Converter.Internal as I
import mpi4py.MPI as MPI
import numpy as np

from maia.sids import Internal_ext as IE

def _compare_pointrange(gc1, gc2):
 """
 Compare a couple of grid_connectivity nodes and return True
 if the PointList and PointListDonor are symmetrically equals
 """
 gc1_pr  = I.getNodeFromName1(gc1, 'PointRange')[1]
 gc1_prd = I.getNodeFromName1(gc1, 'PointRangeDonor')[1]
 gc2_pr  = I.getNodeFromName1(gc2, 'PointRange')[1]
 gc2_prd = I.getNodeFromName1(gc2, 'PointRangeDonor')[1]
 if gc1_pr.shape != gc2_prd.shape or gc2_pr.shape != gc1_prd.shape:
   return False
 return (np.all(gc1_pr == gc2_prd) and np.all(gc2_pr == gc1_prd))

def _compare_pointlist(gc1, gc2):
  """  
  Compare a couple of grid_connectivity nodes and return True
  if the PointList and PointListDonor are symmetrically equals
  """
  gc1_pl  = np.asarray(I.getNodeFromName1(gc1, 'PointList')[1])
  gc1_pld = np.asarray(I.getNodeFromName1(gc1, 'PointListDonor')[1])
  gc2_pl  = np.asarray(I.getNodeFromName1(gc2, 'PointList')[1])
  gc2_pld = np.asarray(I.getNodeFromName1(gc2, 'PointListDonor')[1])
  if gc1_pl.shape != gc2_pld.shape or gc2_pl.shape != gc1_pld.shape:
    return False
  return (np.all(gc1_pl == gc2_pld) and np.all(gc2_pl == gc1_pld))

def _create_local_match_table(gc_list, gc_paths):
  """
  Iterate over a list of joins to compare the PointList / PointListDonor
  and retrieve the pairs of matching joins
  """
  nb_joins = len(gc_list)
  local_match_table = np.zeros((nb_joins, nb_joins), dtype=bool)

  for igc, gc in enumerate(gc_list):
    current_path = gc_paths[igc]
    current_base = current_path.split('/')[0]
    opp_path = IE.getZoneDonorPath(current_base, gc)
    #print('current', gc[0], gc_paths[igc], 'opp', opp_path)
    candidates = [i for i,path in enumerate(gc_paths) if
        (path==opp_path and IE.getZoneDonorPath(path.split('/')[0], gc_list[i]) == current_path)]
    #print('  candidates', candidates)
    gc_has_pl = I.getNodeFromName1(gc, 'PointList') is not None
    for j in candidates:
      candidate_has_pl = I.getNodeFromName1(gc_list[j], 'PointList') is not None
      if gc_has_pl and candidate_has_pl:
        local_match_table[igc][j] = _compare_pointlist(gc, gc_list[j])
      elif not gc_has_pl and not candidate_has_pl:
        local_match_table[igc][j] = _compare_pointrange(gc, gc_list[j])
  #print('  check_candidates', local_match_table)
  return local_match_table

def add_joins_ordinal(dist_tree, comm):
  """
  For each GridConnectivity_t node found in the dist_tree, find the
  opposite GC node and create for each pair of GCs the Ordinal and
  OrdinalOpp nodes allowing to identify them
  GC Node must have either PointList/PointListDonor arrays or
  PointRange/PointRangeDonor arrays, not both.
  """
  gc_list  = []
  gc_paths = []
  # > First pass to collect joins
  for base in I.getBases(dist_tree):
    for zone in I.getZones(base):
      for zgc in I.getNodesFromType1(zone, 'ZoneGridConnectivity_t'):
        gcs = I.getNodesFromType1(zgc, 'GridConnectivity_t') + I.getNodesFromType1(zgc, 'GridConnectivity1to1_t')
        for gc in gcs:
          gc_list.append(gc)
          gc_paths.append(base[0] + '/' + zone[0])

  local_match_table = _create_local_match_table(gc_list, gc_paths)

  global_match_table = np.empty(local_match_table.shape, dtype=bool)
  comm.Allreduce(local_match_table, global_match_table, op=MPI.LAND)
  #print('  check_candidates\n', global_match_table)
  assert(np.all(np.sum(global_match_table, axis=0) == 1))

  opp_join_id = np.where(global_match_table)[1]
  for gc_id, (gc, opp_id) in enumerate(zip(gc_list, opp_join_id)):
    I.createNode('Ordinal'   , 'UserDefinedData_t',  gc_id+1, parent=gc)
    I.createNode('OrdinalOpp', 'UserDefinedData_t', opp_id+1, parent=gc)

def rm_joins_ordinal(dist_tree):
  """
  Remove Ordinal and OrdinalOpp nodes created on GC_t
  """
  gc_query = lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  for gc in IE.getNodesByMatching(dist_tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', gc_query]):
    I._rmNodesByName1(gc, 'Ordinal')
    I._rmNodesByName1(gc, 'OrdinalOpp')
