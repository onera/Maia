import mpi4py.MPI as MPI
import numpy as np

import maia.pytree as PT

def _compare_pointrange(gc1, gc2):
 """
 Compare a couple of grid_connectivity nodes and return True
 if the PointList and PointListDonor are symmetrically equals
 """
 gc1_pr  = PT.get_child_from_name(gc1, 'PointRange')[1]
 gc1_prd = PT.get_child_from_name(gc1, 'PointRangeDonor')[1]
 gc2_pr  = PT.get_child_from_name(gc2, 'PointRange')[1]
 gc2_prd = PT.get_child_from_name(gc2, 'PointRangeDonor')[1]
 if gc1_pr.shape != gc2_prd.shape or gc2_pr.shape != gc1_prd.shape:
   return False
 return (np.all(gc1_pr == gc2_prd) and np.all(gc2_pr == gc1_prd))

def _compare_pointlist(gc1, gc2):
  """  
  Compare a couple of grid_connectivity nodes and return True
  if the PointList and PointListDonor are symmetrically equals
  """
  gc1_pl  = np.asarray(PT.get_child_from_name(gc1, 'PointList')[1])
  gc1_pld = np.asarray(PT.get_child_from_name(gc1, 'PointListDonor')[1])
  gc2_pl  = np.asarray(PT.get_child_from_name(gc2, 'PointList')[1])
  gc2_pld = np.asarray(PT.get_child_from_name(gc2, 'PointListDonor')[1])
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
    gc_has_pl = PT.get_child_from_name(gc, 'PointList') is not None
    for j in candidates:
      candidate_has_pl = PT.get_child_from_name(gc_list[j], 'PointList') is not None
      if gc_has_pl and candidate_has_pl:
        local_match_table[igc][j] = _compare_pointlist(gc, gc_list[j])
      elif not gc_has_pl and not candidate_has_pl:
        local_match_table[igc][j] = _compare_pointrange(gc, gc_list[j])
  #print('  check_candidates', local_match_table)
  return local_match_table

def add_joins_donor_name(dist_tree, comm, force=False):
  """
  For each GridConnectivity_t node found in the dist_tree, find the
  opposite GC node and create the GridConnectivityDonorName node
  GC Node must have either PointList/PointListDonor arrays or
  PointRange/PointRangeDonor arrays, not both.
  If force=True, donor are recomputed, else the computation is
  made only if there are not in tree
  """
  
  gc_list  = []
  gc_paths = []
  # > First pass to collect joins
  match1to1 = lambda n : PT.get_label(n) in ['GridConnectivity1to1_t', 'GridConnectivity_t'] \
      and PT.GridConnectivity.is1to1(n)
  query = ["CGNSBase_t", "Zone_t", "ZoneGridConnectivity_t", match1to1]

  if force:
    for gc in PT.iter_children_from_predicates(dist_tree, query):
      PT.rm_children_from_name(gc, 'GridConnectivityDonorName')

  need_compute = False
  for nodes in PT.iter_children_from_predicates(dist_tree, query, ancestors=True):
    gc_list.append(nodes[-1])
    gc_paths.append('/'.join([PT.get_name(node) for node in nodes[:2]]))
    if PT.get_child_from_name(nodes[-1], 'GridConnectivityDonorName') is None:
      need_compute = True
  
  if not need_compute:
    return

  #gc_paths -> chemin des zones
  local_match_table = _create_local_match_table(gc_list, gc_paths)

  global_match_table = np.empty(local_match_table.shape, dtype=bool)
  comm.Allreduce(local_match_table, global_match_table, op=MPI.LAND)
  #print('  check_candidates\n', global_match_table)
  assert(np.all(np.sum(global_match_table, axis=0) == 1))

  opp_join_id = np.where(global_match_table)[1]
  for gc_id, (gc, opp_id) in enumerate(zip(gc_list, opp_join_id)):
    PT.new_node("GridConnectivityDonorName", "Descriptor_t", PT.get_name(gc_list[opp_id]), parent=gc)

def get_jn_donor_path(dist_tree, jn_path):
  """
  Return the patch of the matching jn in the tree. GridConnectivityDonorName must exists.
  """
  cur_jn = PT.get_node_from_path(dist_tree, jn_path)
  base_name, zone_name, zgc_name, jn_name = jn_path.split('/')
  opp_zone_path = PT.getZoneDonorPath(base_name, cur_jn)
  opp_gc_name   = PT.get_value(PT.get_child_from_name(cur_jn, "GridConnectivityDonorName"))

  opp_zone      = PT.get_node_from_path(dist_tree, opp_zone_path)
  opp_zgc       = PT.get_child_from_label(opp_zone, "ZoneGridConnectivity_t")
  return f"{opp_zone_path}/{PT.get_name(opp_zgc)}/{opp_gc_name}"

def update_jn_name(dist_tree, jn_path, new_name):
  """
  Rename a 1to1 GC and update the opposite GridConnectivityDonorName.
  """
  cur_jn = PT.get_node_from_path(dist_tree, jn_path)
  opp_jn = PT.get_node_from_path(dist_tree, get_jn_donor_path(dist_tree, jn_path))
  opp_gc_name_n = PT.get_child_from_name(opp_jn, "GridConnectivityDonorName")
  PT.set_name(cur_jn, new_name)
  PT.set_value(opp_gc_name_n, new_name)
  
def get_matching_jns(dist_tree, filter_loc=None):
  """
  Return the list of pairs of matching jns
  """
  if filter_loc is None:
    gc_query = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
                         and PT.GridConnectivity.is1to1(n)
  else:
    gc_query = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t'] \
                         and PT.GridConnectivity.is1to1(n) \
                         and PT.Subset.GridLocation(n) == filter_loc

  query = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', gc_query]

  # Retrieve interfaces pathes and call function
  jn_pairs = []
  for jn_path in PT.predicates_to_paths(dist_tree, query):
    opp_jn_path   = get_jn_donor_path(dist_tree, jn_path)
    pair = tuple(sorted([jn_path, opp_jn_path]))
    if not pair in jn_pairs:
      jn_pairs.append(pair)
  return jn_pairs

def copy_donor_subset(dist_tree):
  """
  Retrieve for each GridConnectivity_t node the opposite
  pointlist in the tree. This assume that GridConnectivityDonorName were added and index distribution
  was identical for two related gc nodes
  """
  gc_predicates = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', \
      lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']]

  for jn_path in PT.predicates_to_paths(dist_tree, gc_predicates):
    opp_jn_path = get_jn_donor_path(dist_tree, jn_path)
    cur_jn = PT.get_node_from_path(dist_tree, jn_path)
    opp_jn = PT.get_node_from_path(dist_tree, opp_jn_path)
    opp_patch = PT.deep_copy(PT.Subset.getPatch(opp_jn))
    PT.set_name(opp_patch, PT.get_name(opp_patch) + 'Donor')
    PT.rm_children_from_name(cur_jn, PT.get_name(opp_patch))
    PT.add_child(cur_jn, opp_patch)


def store_interfaces_ids(dist_tree):
  """
  Attribute to each 1to1 pair a unique interace id. GridConnectivityDonorName must have been added in the tree.
  Store this id and the position (first or second) in disttree.
  Note : this function does not manage (for now?) location: two jns at different interface
  will have a different id
  """
  matching_pairs = get_matching_jns(dist_tree)
  for i, matching_pair in enumerate(matching_pairs):
    for j,jn_path in enumerate(matching_pair):
      jn = PT.get_node_from_path(dist_tree, jn_path)
      PT.new_DataArray("DistInterfaceId",  i+1, parent=jn)
      PT.new_DataArray("DistInterfaceOrd", j,   parent=jn)

def clear_interface_ids(dist_tree):
  """
  Remove DistInterfaceId nodes created on GC_t
  """
  gc_query = lambda n: PT.get_label(n) in ['GridConnectivity_t', 'GridConnectivity1to1_t']
  for gc in PT.iter_children_from_predicates(dist_tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', gc_query]):
    PT.rm_children_from_name(gc, 'DistInterfaceId')
    PT.rm_children_from_name(gc, 'DistInterfaceOrd')

