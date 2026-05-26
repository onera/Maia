import mpi4py.MPI as MPI
import numpy as np

from maia.typing import *
from maia.pytree.typing import Predicates

import maia.pytree      as PT
import maia.pytree.maia as MT

from maia.utils.parallel import algo as par_algo
from maia.utils import s_numbering

IS_GC_MATCH = PT.pred.is_gc_of_kind(is_1to1=True)

def gc_is_reference(gc_s, zone_path):
  """
  Check if a structured 1to1 GC is the reference of its pair or not
  The opposite GC is not needed to do that (paths and pointrange are
  compared)
  """
  zone_path_opp = PT.GridConnectivity.ZoneDonorPath(gc_s, PT.utils.path_head(zone_path))
  if zone_path < zone_path_opp:
    return True
  elif zone_path > zone_path_opp:
    return False
  else: #Same zone path
    pr  = PT.get_child_from_name(gc_s, "PointRange")[1]
    prd = PT.get_child_from_name(gc_s, "PointRangeDonor")[1]
    bnd_axis   = PT.Subset.normal_axis(gc_s)
    bnd_axis_d = PT.Subset.normal_axis(PT.new_GridConnectivity1to1(point_range=prd))
    if bnd_axis < bnd_axis_d:
      return True
    elif bnd_axis > bnd_axis_d:
      return False
    else: #Same boundary axis
      bnd_axis_val = np.abs(pr[bnd_axis,0])
      bnd_axis_val_d = np.abs(prd[bnd_axis_d,0])
      if bnd_axis_val < bnd_axis_val_d:
        return True
      elif bnd_axis_val > bnd_axis_val_d:
        return False
      else: #Same position in boundary axis
        if np.sum(pr) < np.sum(prd):
          return True
        elif np.sum(pr) > np.sum(prd):
          return False
  raise ValueError("Unable to determine if node is reference")

def _compare_pointrange(gc1, gc2):
 """
 Compare a couple of grid_connectivity nodes and return True
 if the PointList and PointListDonor are equals, even
 if the symmetry is not respected
 """
 gc1_pr  = PT.get_child_from_name(gc1, 'PointRange')[1]
 gc1_prd = PT.get_child_from_name(gc1, 'PointRangeDonor')[1]
 gc2_pr  = PT.get_child_from_name(gc2, 'PointRange')[1]
 gc2_prd = PT.get_child_from_name(gc2, 'PointRangeDonor')[1]
 if gc1_pr.shape != gc2_prd.shape or gc2_pr.shape != gc1_prd.shape:
   return False

 return (np.sort(gc1_pr) == np.sort(gc2_prd)).all() and (np.sort(gc2_pr) == np.sort(gc1_prd)).all()

def _compare_pointlist(gc1, gc2, comm):
  """  
  Compare a couple of grid_connectivity nodes and return True
  if the PointList and PointListDonor are equals, even
  if the symmetry is not respected
  """
  gc1_pl  = PT.get_np_value(PT.find_child_from_name(gc1, 'PointList'))[0]
  gc1_pld = PT.get_np_value(PT.find_child_from_name(gc1, 'PointListDonor'))[0]
  gc2_pl  = PT.get_np_value(PT.find_child_from_name(gc2, 'PointList'))[0]
  gc2_pld = PT.get_np_value(PT.find_child_from_name(gc2, 'PointListDonor'))[0]

  # Sort first JN according to PL
  S1 = par_algo.DistSorter(gc1_pl, comm)
  gc1_pl_s  = S1.sorted_key()
  gc1_pld_s = S1.sort(gc1_pld)
  # Sort second JN according to PLd, using same distribution
  # Index error means that some PLd_2 idx are to big regarding to PL_1 => wrong candidate
  try:
    S2 = par_algo.DistSorter(gc2_pld, comm, distri=S1.distri)
  except IndexError:
    return False
  gc2_pl_s  = S2.sort(gc2_pl)
  gc2_pld_s = S2.sorted_key()
  
  if (gc1_pl_s.size != gc2_pld_s.size) or (gc1_pld_s.size != gc2_pl_s.size):
    return False
  return (gc1_pl_s == gc2_pld_s).all() and (gc1_pld_s == gc2_pl_s).all()

def _jn_is_symmetric_loc(gc1, gc2):
  """ Return True if two matching jns are symmetrically equal """
  gc1_patch = PT.Subset.getPatch(gc1)
  gc2_patch = PT.Subset.getPatch(gc2)
  
  gc1_dpatch = PT.find_child_from_name(gc1, PT.get_name(gc2_patch) + 'Donor')
  gc2_dpatch = PT.find_child_from_name(gc2, PT.get_name(gc1_patch) + 'Donor')

  # Sizes are not checked because jns are supposed to be matching
  return np.array_equal(gc1_patch[1], gc2_dpatch[1]) and np.array_equal(gc1_dpatch[1], gc2_patch[1])

def _as_unst_gc(dist_tree, gc, gc_path, opp_path):
  """ Destructure structured PointList (IJK) for easier PL comparison
  Returns a shallow copy (input node is preserved) """
  for is_donor, path in enumerate([gc_path, opp_path]):
    name = 'PointListDonor' if is_donor else 'PointList'
    pl  = PT.get_np_value(PT.find_child_from_name(gc, name))
    if (s:=pl.shape[0]) != 1:
      zone = PT.find_node_from_path(dist_tree, PT.utils.path_head(path, 2))
      assert PT.Zone.Type(zone) == 'Structured'
      fn = s_numbering.ij_to_index if s == 2 else s_numbering.ijk_to_index
      pl_u = fn(*[pl[i,:] for i in range(s)], PT.Zone.VertexSize(zone))
      PT.update_child(gc, name, value=pl_u.reshape((1,-1), order='F'))      

def _create_local_match_table(dist_tree, gc_list, gc_paths, comm):
  """
  Iterate over a list of joins to compare the PointList / PointListDonor
  and retrieve the pairs of matching joins
  """
  nb_joins = len(gc_list)
  local_match_table = np.zeros((nb_joins, nb_joins), dtype=bool)

  gc_n_elem = lambda n: PT.Subset.n_elem(n) if PT.get_child_from_name(n, 'PointRange') is not None \
                                            else MT.Subset.n_elem(n)

  for igc, gc in enumerate(gc_list):
    current_path = gc_paths[igc]
    current_base = current_path.split('/')[0]
    opp_path = PT.GridConnectivity.ZoneDonorPath(gc, current_base)
    candidates = [i for i,path in enumerate(gc_paths) if
        (path==opp_path and 
         PT.GridConnectivity.ZoneDonorPath(gc_list[i], path.split('/')[0]) == current_path and
         gc_n_elem(gc) == gc_n_elem(gc_list[i]))]
    gc_has_pl = PT.get_child_from_name(gc, 'PointList') is not None
    for j in candidates:
      candidate_has_pl = PT.get_child_from_name(gc_list[j], 'PointList') is not None
      if gc_has_pl and candidate_has_pl:
        _gc     = PT.shallow_copy(gc)
        _gc_opp = PT.shallow_copy(gc_list[j])
        _as_unst_gc(dist_tree, _gc,     current_path, opp_path)
        _as_unst_gc(dist_tree, _gc_opp, opp_path, current_path)
        local_match_table[igc][j] = _compare_pointlist(_gc, _gc_opp, comm)
      elif not gc_has_pl and not candidate_has_pl:
        local_match_table[igc][j] = _compare_pointrange(gc, gc_list[j])
  return local_match_table

def find_joins_donor_name(dist_tree:CGNSDistTree, comm:MPIComm):
  """ Retrieve the related matching GridConnectivity(1to1)_t nodes.
  
  The purpose of this function is to complement the standard description
  of these nodes, which store the path of their related (*Donor*)
  Zone, by also storing the name of the associated join 
  for all GridConnectivity_t nodes of :func:`~maia.pytree.GridConnectivity.Type` Abutting1to1.
  
  For each input join, this name is stored in a ``Descriptor_t``
  node named ``GridConnectivityDonorName``.

  This function requires the matching joins to be defined on the two connected zones,
  which, by the way, is a widely used assumption in Maia.

  Args:
    dist_tree  (CGNSDistTree) : Input distributed tree
    comm           (MPIComm)  : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #find_joins_donor_name@start
        :end-before: #find_joins_donor_name@end
        :dedent: 2
  """
  
  gc_list  = []
  gc_paths = []
  # > First pass to collect joins
  query = ["CGNSBase_t", "Zone_t", "ZoneGridConnectivity_t", IS_GC_MATCH]

  for nodes in PT.iter_children_from_predicates(dist_tree, query, ancestors=True): #get_node_from_path is slower, dont use it
    gc_node = nodes[-1]
    if PT.get_child_from_name(gc_node, 'GridConnectivityDonorName') is None:
      # Skip nodes that already have their DonorName
      gc_list.append(gc_node)
      gc_paths.append('/'.join([PT.get_name(node) for node in nodes[:2]]))

  if len(gc_list) == 0:
    return

  local_match_table = _create_local_match_table(dist_tree, gc_list, gc_paths, comm)

  global_match_table = np.empty(local_match_table.shape, dtype=bool)
  comm.Allreduce(local_match_table, global_match_table, op=MPI.LAND)
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
  opp_zone_path = PT.GridConnectivity.ZoneDonorPath(cur_jn, base_name)
  gc_donor_name = PT.get_child_from_name(cur_jn, "GridConnectivityDonorName")
  if gc_donor_name is None:
    raise RuntimeError(f"No GridConnectivityDonorName found in GC {jn_path}")
  opp_gc_name   = PT.get_value(gc_donor_name)

  opp_zone      = PT.get_node_from_path(dist_tree, opp_zone_path)
  if opp_zone is None:
    raise RuntimeError(f"GridConnectivity {jn_name} connects to zone {opp_zone_path}, who does not exist")
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
  
def get_matching_jns(dist_tree:CGNSTree, 
                     select_func:Callable[[CGNSTree], bool] = lambda n: True) -> List[Tuple[CGNSPath, CGNSPath]]:
  """
  Return the list of pairs of matching jns
  """
  gc_query = IS_GC_MATCH
  if select_func is not None:
    gc_query = gc_query & PT.pred.NodePredicate(select_func)

  query:Predicates = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', gc_query]

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
  Retrieve for each 1to1 GridConnectivity_t node the opposite
  pointlist in the tree. This assume that GridConnectivityDonorName were added and index distribution
  was identical for two related gc nodes
  """
  gc_predicates = ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', IS_GC_MATCH]

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
  Attribute to each 1to1 pair a unique interface id. GridConnectivityDonorName must have been added in the tree.
  Store this id and the position (first or second) in disttree.
  Note : this function does not manage (for now?) location: two jns at different interface
  will have a different id
  """
  matching_pairs = get_matching_jns(dist_tree)
  for i, matching_pair in enumerate(matching_pairs):
    for j,jn_path in enumerate(matching_pair):
      jn = PT.get_node_from_path(dist_tree, jn_path)
      PT.new_Descriptor("DistInterfaceId",  str(i+1), parent=jn)
      PT.new_Descriptor("DistInterfaceOrd", str(j),   parent=jn)

def clear_interface_ids(dist_tree):
  """
  Remove DistInterfaceId nodes created on GC_t
  """
  for gc in PT.iter_children_from_predicates(dist_tree, ['CGNSBase_t', 'Zone_t', 'ZoneGridConnectivity_t', PT.pred.IS_GC]):
    PT.rm_children_from_name(gc, 'DistInterfaceId')
    PT.rm_children_from_name(gc, 'DistInterfaceOrd')

  
def _has_related_subset(zone, jn_name):
  for zsr in PT.get_children_from_label(zone, 'ZoneSubRegion_t'):
    if (rname := PT.get_child_from_name(zsr, 'GridConnectivityRegionName')) is not None:
      if PT.get_str_value(rname) == jn_name:
        return True
  return False

def enforce_symmetric_joins(dist_tree:CGNSDistTree, comm:MPIComm):
  """ Permute subsets of matching joins to enforce symmetry.

  Two matching joins ``gc1`` and ``gc2`` are said to be symmetric if
  the PointList (resp. PointRange) of ``gc1`` is element wise equal to the
  PointListDonor (resp. PointRangeDonor) of ``gc2`` and vice versa.
  The CGNS standard does not impose this symmetry, but some
  algorithms or solvers rely on it.

  Args:
    dist_tree  (CGNSDistTree) : Input distributed tree
    comm       (MPIComm)      : MPI communicator

  Example:
      .. literalinclude:: snippets/test_algo.py
        :start-after: #enforce_symmetric_joins@start
        :end-before: #enforce_symmetric_joins@end
        :dedent: 2
  """
  #MT.check_cgns_dist_tree(dist_tree) # Function is called by fix_tree before computing distribution
  find_joins_donor_name(dist_tree, comm)
  jn_pairs = get_matching_jns(dist_tree)

  gc_cur_list = list()
  gc_opp_list = list()
  is_symm_l = np.empty(len(jn_pairs), bool)
  is_symm_g = np.empty(len(jn_pairs), bool)
  for i,jn_pair in enumerate(jn_pairs):
    gc_cur_list.append(PT.find_node_from_path(dist_tree, jn_pair[0]))
    gc_opp_list.append(PT.find_node_from_path(dist_tree, jn_pair[1]))
    is_symm_l[i] = _jn_is_symmetric_loc(gc_cur_list[-1], gc_opp_list[-1])

  comm.Allreduce(is_symm_l, is_symm_g, MPI.LAND)

  for i, jn_pair in enumerate(jn_pairs):
    # Skip if already symetric (avoid raise due to ZSR)
    if is_symm_g[i]:
      continue

    gc_cur = gc_cur_list[i]
    gc_opp = gc_opp_list[i]
    zone = PT.find_node_from_path(dist_tree, PT.utils.path_head(jn_pair[0], 2))
    if _has_related_subset(zone, PT.get_name(gc_cur)):
      gc_cur, gc_opp = gc_opp, gc_cur # Try permutation, maybe gc_opp has no ZSR

    zone = PT.find_node_from_path(dist_tree, PT.utils.path_head(jn_pair[0], 2))
    if _has_related_subset(zone, PT.get_name(gc_cur)):
      raise RuntimeError(f"Can not reoder GC_t node {gc_cur[0]} which defines one or more ZoneSubRegion_t nodes")

    # Impose gc order to gc_opp
    # Mathing GCs are either PointList/PointList or PointRange/PointRange
    key = 'PointList' if PT.get_child_from_name(gc_cur, 'PointList') is not None else 'PointRange'
    PT.update_child(gc_opp,
                    f'{key}',
                    value=np.copy(PT.get_np_value(PT.find_node_from_name(gc_cur,f'{key}Donor'))))
    PT.update_child(gc_opp,
                    f'{key}Donor',
                    value=np.copy(PT.get_np_value(PT.find_node_from_name(gc_cur,f'{key}'))))
  
