import Converter.Internal as I
import mpi4py.MPI as MPI
import numpy as np

import maia.pytree as PT

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
    opp_path = PT.getZoneDonorPath(current_base, gc)
    #print('current', gc[0], gc_paths[igc], 'opp', opp_path)
    candidates = [i for i,path in enumerate(gc_paths) if
        (path==opp_path and PT.getZoneDonorPath(path.split('/')[0], gc_list[i]) == current_path)]
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

def add_joins_ordinal(dist_tree, comm, force=False):
  """
  For each GridConnectivity_t node found in the dist_tree, find the
  opposite GC node and create for each pair of GCs the Ordinal and
  OrdinalOpp nodes allowing to identify them
  GC Node must have either PointList/PointListDonor arrays or
  PointRange/PointRangeDonor arrays, not both.
  If force=True, ordinals are recomputed, else the computation is
  made only if there is no ordinal
  """
  
  gc_list  = []
  gc_paths = []
  # > First pass to collect joins
  match1to1 = lambda n : I.getType(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t'] \
      and PT.GridConnectivity.is1to1(n)
  query = ["CGNSBase_t", "Zone_t", "ZoneGridConnectivity_t", match1to1]
  if force:
    rm_joins_ordinal(dist_tree)
    compute_ordinal = True
    for nodes in PT.iter_children_from_predicates(dist_tree, query, ancestors=True):
      gc_list.append(nodes[-1])
      gc_paths.append('/'.join([I.getName(node) for node in nodes[:2]]))
  else:
    compute_ordinal = False
    for nodes in PT.iter_children_from_predicates(dist_tree, query, ancestors=True):
      gc_list.append(nodes[-1])
      gc_paths.append('/'.join([I.getName(node) for node in nodes[:2]]))
      ordinal_n     = I.getNodeFromName1(nodes[-1], 'Ordinal')
      ordinal_opp_n = I.getNodeFromName1(nodes[-1], 'OrdinalOpp')
      if (ordinal_n is None) or (ordinal_opp_n is None):
        compute_ordinal = True
  
  if not compute_ordinal:
    return

  local_match_table = _create_local_match_table(gc_list, gc_paths)

  global_match_table = np.empty(local_match_table.shape, dtype=bool)
  comm.Allreduce(local_match_table, global_match_table, op=MPI.LAND)
  #print('  check_candidates\n', global_match_table)
  assert(np.all(np.sum(global_match_table, axis=0) == 1))

  opp_join_id = np.where(global_match_table)[1]
  for gc_id, (gc, opp_id) in enumerate(zip(gc_list, opp_join_id)):
    I.createUniqueChild(gc, 'Ordinal'   , 'UserDefinedData_t',  gc_id+1)
    I.createUniqueChild(gc, 'OrdinalOpp', 'UserDefinedData_t', opp_id+1)

def pl_donor_from_ordinals(dist_tree):
  """
  TODO Generalize for PointRangeDonor
  Retrieve for each GridConnectivity_t node the opposite
  pointlist in the tree. This assume that ordinal were added and index distribution
  was identical for two related gc nodes
  """
  ordinal_to_pl = dict()
  gc_t_path = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity_t'
  gc_nodes = PT.get_children_from_predicates(dist_tree, gc_t_path)

  for gc in gc_nodes:
    ordinal = I.getNodeFromName1(gc, 'Ordinal')[1][0]
    ordinal_to_pl[ordinal] = I.getNodeFromName1(gc, 'PointList')[1]

  for gc in gc_nodes:
    ordinal_opp = I.getNodeFromName1(gc, 'OrdinalOpp')[1][0]
    donor_pl = ordinal_to_pl[ordinal_opp]
    I.newPointList('PointListDonor', donor_pl, parent=gc)

def rm_joins_ordinal(dist_tree):
  """
  Remove Ordinal and OrdinalOpp nodes created on GC_t
  """
  gc_query = lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  for gc in PT.iter_children_from_predicates(dist_tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', gc_query]):
    I._rmNodesByName1(gc, 'Ordinal')
    I._rmNodesByName1(gc, 'OrdinalOpp')
    I._rmNodesByName1(gc, 'InterfaceId')
    I._rmNodesByName1(gc, 'InterfacePos')

def ordinals_to_interfaces(dist_tree):
  """
  Attribute to each 1to1 pair a unique interace id. Ordinals must have been added in the tree.
  Store this id and the position (first or second) in disttree.
  """
  gc_query = lambda n: I.getType(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  ordinal_to_id = {}
  next_id = 0
  for gc in PT.iter_children_from_predicates(dist_tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', gc_query]):
    ordinal     = I.getNodeFromName1(gc, "Ordinal")
    ordinal_opp = I.getNodeFromName1(gc, "OrdinalOpp")
    if ordinal is not None: #Include only 1to1 jns
      key = min(ordinal[1][0], ordinal_opp[1][0])
      interface_pos = 1
      if not key in ordinal_to_id:
        next_id += 1
        ordinal_to_id[key] = next_id
        interface_pos = 0
      I.newDataArray("InterfaceId", ordinal_to_id[key],  parent=gc)
      I.newDataArray("InterfacePos", interface_pos,      parent=gc)
